#!/usr/bin/env python3
import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Any
import gc
from feature_extraction.variance import variance_filter_pipeline

# =============================================================================
# PARAMETER DEFINITIONS (Explanations kept in code, but removed from prints)
# =============================================================================

DEFAULT_PARAMETERS = {
    'variance': {
        'show_plot': True,
        'knee_S': 1.0,
        'outlier_multiplier': 3.0,
        'fallback_percentile': 90,
        'min_clean_ratio': 0.5
    },
    'feature_selection': {
        'fisher_amino': {
            'max_outputs': 5,      # Maps to 'max_outputs' in your pipeline
            'knee_S': 1.0,         # Sensitivity for the KneeLocator
        },
        'bpso': {
            'population_size': 20, 
            'iters_scaling': 0.5, # Controls dynamic iterations
            'alpha': 0.95, 
            'threshold': 0.5, 
            'stride': 5,
            'candidate_limit': 150
        },
        'mpso': {
            'dims': 5,
            'candidate_limit': 250, 
            'mpso_iters': 50, 
            'alpha': 0.9, 
            'threshold': 0.5, 
            'stride': 5,
            'population_size': 20
        },
        'chi_sq_amino': {
            'max_amino': 10, 
            'q_bins': 5, 
            'knee_S': 5.0, 
            'sample_rows': 20000, 
            'stride': 10,  # Increased default stride for Chi-Sq stats
            'show_plots': True
        }
    },
    'dimensionality_reduction': {
        'flda': {'solver': 'eig'},
        'pca': {'num_eigenvector': 2},
        'zhlda': {'num_eigenvector': 5},
        'mhlda': {'num_eigenvector': 5},
        'gdhlda': {'num_eigenvector': 5}
    }
}

# =============================================================================
# MINIMAL INTERACTIVE HELPERS (Hyperparameters only)
# =============================================================================

def get_params_minimal(category: str, subcategory: str = None) -> Dict[str, Any]:
    """Displays only hyperparameters and returns user modifications."""
    if subcategory:
        defaults = DEFAULT_PARAMETERS[category][subcategory]
        header = f"[{category.upper()} : {subcategory.upper()}]"
    else:
        defaults = DEFAULT_PARAMETERS[category]
        header = f"[{category.upper()}]"

    print(f"\n{header}")
    for k, v in defaults.items():
        print(f"  {k}: {v}")

    params = defaults.copy()
    modify = input("\nModify? (y/N): ").strip().lower()
    
    if modify == 'y':
        for k, v in defaults.items():
            user_val = input(f"  {k} [{v}]: ").strip()
            if user_val:
                try:
                    if isinstance(v, bool): params[k] = user_val.lower() in ['true', '1', 'y']
                    elif isinstance(v, int): params[k] = int(user_val)
                    elif isinstance(v, float): params[k] = float(user_val)
                    else: params[k] = user_val
                except ValueError:
                    print(f"  Invalid input. Keeping {v}")
    return params

# =============================================================================
# CORE PIPELINE ENGINE
# =============================================================================

# ... (Keep Imports and DEFAULT_PARAMETERS as they are) ...

# =============================================================================
# CORE PIPELINE ENGINE (AUTOMATED DR PHASE)
# =============================================================================

def run_interactive_pipeline(data_factory, pipeline_configs, class_assignment_func=None):
    cache_dir = Path("pipeline_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # --- PHASE 1: VARIANCE (Keep Interactive) ---
    def var_exec(params):
        print("⚙️ Streaming data into variance filter...")
        return pd.concat(variance_filter_pipeline(data_factory, **params), ignore_index=True)

    variance_df = run_interactive_step_generic(
        "VARIANCE", cache_dir / "variance.pkl", 'variance', None, var_exec
    )
    if variance_df is None: return {}

    # --- PHASE 2: FEATURE SELECTION (Keep Interactive) ---
    fs_methods = list(set(c['feature_selection'] for c in pipeline_configs))
    fs_paths = {}

    if 'class' not in variance_df.columns:
        variance_df['class'] = class_assignment_func(variance_df) if class_assignment_func else \
                               variance_df['construct'] + '_' + variance_df['subconstruct']

    for method in fs_methods:
        path = cache_dir / f"{method}.pkl"
        fs_paths[method] = path
        def fs_exec(params):
            return execute_fs_method(method, lambda: [variance_df], params)

        _ = run_interactive_step_generic(method.upper(), path, 'feature_selection', method, fs_exec)
        gc.collect()

    gc.collect()

    # --- PHASE 3: AUTOMATED DIMENSIONALITY REDUCTION ---
    print("\n" + "="*50)
    print("🤖 AUTOMATED DIMENSIONALITY REDUCTION PHASE")
    
    # ONE-TIME GLOBAL QUESTION
    recompute_all = input("Found existing DR caches. Recompute ALL methods? (y/N): ").strip().lower() == 'y'
    
    print("No further interaction required. Results will be saved automatically.")
    print("="*50)

    final_results = {}
    for config in pipeline_configs:
        name, fs_m, dr_m = config['name'], config['feature_selection'], config['dimensionality_reduction']
        cache_path = cache_dir / f"{name}.pkl"

        # Logic for skipping
        if cache_path.exists() and not recompute_all:
            print(f"⏩ Skipping {name} (Already cached)")
            # Load the cache into final_results so it's available for the return
            try:
                with open(cache_path, 'rb') as f:
                    final_results[name] = pickle.load(f)
            except:
                pass
            continue

        print(f"🚀 Processing: {fs_m.upper()} ➔ {dr_m.upper()}")
        
        try:
            with open(fs_paths[fs_m], 'rb') as f:
                fs_data = pickle.load(f)

            dr_params = DEFAULT_PARAMETERS['dimensionality_reduction'].get(dr_m, {}).copy()
            
            # This now includes the dynamic dimension safety check we discussed
            dr_res = execute_dr_method(dr_m, fs_data, dr_params)
            
            if dr_res is not None:
                with open(cache_path, 'wb') as f:
                    pickle.dump(dr_res, f)
                final_results[name] = dr_res
                print(f"✅ Saved: {cache_path}")

        except Exception as e:
            print(f"❌ Error in pipeline {name}: {e}")
        
        finally:
            if 'fs_data' in locals(): del fs_data
            gc.collect()

    print("\n✨ All automated pipelines complete.")

    return final_results, variance_df


# =============================================================================
# GENERIC INTERACTIVE HELPER
# =============================================================================

def run_interactive_step_generic(name, cache_path, param_category, param_subcategory, execution_fn, preview_fn=None):
    """
    Generic runner for an interactive pipeline step with caching.
    """
    result = None

    # Check cache
    if cache_path.exists():
        load_cache = input(f"Found cached result for {name} ({cache_path}). Load? (Y/n): ").strip().lower()
        if load_cache != 'n':
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                print(f"Loaded cached {name} data.")
                if hasattr(result, 'shape'):
                    print(f"Shape: {result.shape}")
                return result
            except Exception as e:
                print(f"Failed to load cache: {e}")

    # Interactive Loop
    while True:
        params = get_params_minimal(param_category, param_subcategory)
        print(f"Running {name}...")
        
        try:
            temp_result = execution_fn(params)
            
            # Show preview
            if hasattr(temp_result, 'shape'):
                 print(f"{name} Result Shape: {temp_result.shape}")
            
            if preview_fn:
                preview_fn(temp_result)

            accept = input(f"\nAccept {name} results? (y/N): ").strip().lower()
            if accept == 'y':
                result = temp_result
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                print(f"Results accepted and cached to {cache_path}")
                break
            else:
                print("Results rejected. Restarting parameter selection...")
        
        except Exception as e:
            print(f"Error executing {name}: {e}")
            retry = input("Retry with different parameters? (Y/n): ").strip().lower()
            if retry == 'n':
                print(f"Skipping {name}.")
                break
    
    return result

# =============================================================================
# EXECUTION DISPATCHERS
# =============================================================================

def execute_fs_method(method, factory, params):
    if method == 'bpso':
        from feature_selection.BPSO import run_bpso_pipeline
        return run_bpso_pipeline(factory, **params)
    elif method == 'mpso':
        from feature_selection.MPSO import run_mpso_pipeline
        return run_mpso_pipeline(factory, **params)
    elif method == 'fisher_amino':
        from feature_selection.Fisher_AMINO import run_fisher_amino_pipeline
        return run_fisher_amino_pipeline(factory, **params)
    elif method == 'chi_sq_amino':
        from feature_selection.Chi_sq_AMINO import run_feature_selection_pipeline
        return run_feature_selection_pipeline(factory, **params)
    raise ValueError(f"Unknown FS: {method}")

import inspect
import inspect
import pandas as pd
import numpy as np

import inspect
import pandas as pd
import numpy as np

def execute_dr_method(method, fs_result, params):
    """
    Executes dimensionality reduction with dynamic dimension capping.
    Fixes the concatenation error by ensuring METADATA_COLS is treated as a list.
    """
    dr_registry = {
        'flda':   ('dimensionality_reduction.FLDA', 'run_flda'),
        'pca':    ('dimensionality_reduction.PCA', 'run_pca'),
        'zhlda':  ('dimensionality_reduction.ZHLDA', 'run_zhlda'),
        'mhlda':  ('dimensionality_reduction.MHLDA', 'run_mhlda'),
        'gdhlda': ('dimensionality_reduction.GDHLDA', 'run_gdhlda')
    }

    method_str = str(method).lower()
    if method_str not in dr_registry:
        raise ValueError(f"Unknown DR method: {method_str}")

    # 1. Dynamic Module Loading
    module_path, func_name = dr_registry[method_str]
    module = __import__(module_path, fromlist=[func_name])
    func = getattr(module, func_name)

    # 2. Strict Feature Count Calculation
    from data_access import METADATA_COLS
    # FIX: Convert METADATA_COLS to a list before concatenating
    exclude_cols = ['class', 'target'] + list(METADATA_COLS)
    
    numeric_cols = [
        col for col in fs_result.columns 
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(fs_result[col])
    ]
    actual_feat_count = len(numeric_cols)

    # 3. Dimension Capping Logic
    num_classes = fs_result['class'].nunique() if 'class' in fs_result.columns else 2
    max_theoretical_dims = min(actual_feat_count, num_classes - 1)
    
    if method_str == 'pca':
        limit = actual_feat_count
    else:
        # For LDA methods (zhlda, gdhlda, flda, mhlda)
        limit = max_theoretical_dims

    requested_dim = params.get('num_eigenvector', params.get('n_components', 2))

    if requested_dim > limit:
        if limit < 1:
            print(f"🛑 Error: {method_str} cannot run. Only {actual_feat_count} features available.")
            return None
        
        print(f"⚠️  Cap: {method_str} output reduced from {requested_dim} to {limit}")
        params['num_eigenvector'] = limit
        params['n_components'] = limit

    # 4. Parameter Mapping & Signature Filtering
    if 'num_eigenvector' in params: params['n_components'] = params['num_eigenvector']
    if 'n_components' in params: params['num_eigenvector'] = params['n_components']

    sig = inspect.signature(func)
    valid_params = {k: v for k, v in params.items() if k in sig.parameters}

    # 5. Execution
    try:
        result = func(fs_result, **valid_params)
        
        if hasattr(result, '__next__') or (hasattr(result, '__iter__') and not isinstance(result, pd.DataFrame)):
            return next(iter(result))
        return result

    except Exception as e:
        print(f"💥 Failed to execute {method_str}: {e}")
        # Last ditch effort for broadcast errors: drop to 1 dimension
        if "broadcast" in str(e) and valid_params.get('num_eigenvector', 1) > 1:
            print(f"🔄 Retrying {method_str} with 1 dimension...")
            valid_params['num_eigenvector'] = 1
            valid_params['n_components'] = 1
            return func(fs_result, **valid_params)
        raise e
# =============================================================================
# CONFIGURATION CREATOR
# =============================================================================

def create_interactive_pipeline_configs():
    """Create all pipeline configurations for interactive mode."""
    feature_selection_methods = ['bpso', 'mpso', 'fisher_amino', 'chi_sq_amino']
    dimensionality_reduction_methods = ['flda', 'pca', 'zhlda', 'mhlda', 'gdhlda']
    
    configs = []
    for fs in feature_selection_methods:
        for dr in dimensionality_reduction_methods:
            config = {
                'name': f'{fs}_to_{dr}',
                'feature_selection': fs,
                'dimensionality_reduction': dr
            }
            configs.append(config)
    
    return configs

def evaluate_separability(df, target_col='class'):
    features = [c for c in df.columns if c != target_col]
    X = df[features].values
    y = df[target_col].values
    
    overall_mean = np.mean(X, axis=0)
    sb = 0
    sw = 0
    
    for cls in np.unique(y):
        X_c = X[y == cls]
        mean_c = np.mean(X_c, axis=0)
        # Between-class variance
        sb += len(X_c) * np.sum((mean_c - overall_mean)**2)
        # Within-class variance
        sw += np.sum((X_c - mean_c)**2)
    
    return sb / (sw + 1e-9)

def get_feature_importance(original_df, transformed_df, target_col='class'):
    """Maps original feature names back to the new LDs using Correlation."""
    from data_access import METADATA_COLS
    # Keep only numeric columns from the original data for correlation
    original_numeric = original_df.select_dtypes(include=[np.number])
    original_numeric = original_numeric.drop(columns=list(METADATA_COLS), errors='ignore')
    
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    loadings = {}
    
    for ld in ld_cols:
        # Align indices in case of row drops
        common_idx = original_numeric.index.intersection(transformed_df.index)
        corrs = original_numeric.loc[common_idx].corrwith(transformed_df.loc[common_idx, ld])
        # Get top 3 strongest original features for this component
        loadings[ld] = corrs.abs().sort_values(ascending=False).head(3).to_dict()
        
    return loadings

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

def summarize_and_evaluate(results, original_df, corr_threshold=0.5):
    if not results:
        print("⚠️ No results available to evaluate.")
        return

    print("\n" + "🏆 DIMENSIONALITY REDUCTION LEADERBOARD 🏆")
    print("=" * 100)
    print(f"{'Rank':<5} | {'Pipeline Name':<25} | {'Fisher Score':<12} | {'Dims'}")
    print("-" * 100)

    scored_list = []
    for name, df in results.items():
        score = evaluate_separability(df)
        scored_list.append({'name': name, 'score': score, 'df': df})

    scored_list.sort(key=lambda x: x['score'], reverse=True)

    for rank, item in enumerate(scored_list, 1):
        name, total_score, df = item['name'], item['score'], item['df']
        ld_cols = [c for c in df.columns if c != 'class']
        
        fs_method = name.split('_to_')[0]
        cache_path = Path("pipeline_cache") / f"{fs_method}.pkl"
        
        used_features = []
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    fs_data = pickle.load(f)
                    # FILTER: Keep only columns that are in original_df AND are numeric
                    potential_feats = [c for c in fs_data.columns if c not in ['class', 'target']]
                    # This ensures we don't try to correlate strings like 'calmodulin-compact'
                    used_features = original_df[potential_feats].select_dtypes(include=[np.number]).columns.tolist()
            except Exception:
                pass
        
        print(f"{rank:<5} | {name:<25} | {total_score:<12.4f} | {len(ld_cols)}")
        print(f"      📉 Selection Stats: {len(used_features)} numeric features available for mapping.")

        try:
            if used_features:
                original_numeric = original_df[used_features] 
                common_idx = original_numeric.index.intersection(df.index)
                
                for ld in ld_cols:
                    # Calculate correlation only for numeric slices
                    corrs = original_numeric.loc[common_idx].corrwith(df.loc[common_idx, ld])
                    sig_feats = corrs[corrs.abs() > corr_threshold].sort_values(ascending=False)
                    
                    if not sig_feats.empty:
                        feat_info = ", ".join([f"{k} ({corrs[k]:.2f})" for k in sig_feats.index])
                        print(f"      ↳ {ld}: {feat_info}")
            else:
                print("      ↳ Info: No numeric survivors found for feature mapping.")
        except Exception as e:
            print(f"      ↳ Feature Mapping Error: {e}")
            
        if rank <= 3:
            if len(ld_cols) >= 3:
                visualize_cluster_biplot_3d(original_df, df, name, survivors=used_features)
            elif len(ld_cols) == 2:
                visualize_cluster_biplot(original_df, df, name, survivors=used_features)
        print("-" * 100)

    import gc
    gc.collect()

def visualize_cluster_biplot_3d(original_df, transformed_df, pipeline_name, target_col='class', stride=20, survivors=None):
    import plotly.graph_objects as go
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    centered_df = transformed_df.copy()
    centered_df[ld_cols] = transformed_df[ld_cols] - transformed_df[ld_cols].mean()
    df_plot = centered_df.iloc[::stride]
    
    fig = go.Figure()

    # 1. Plot Clusters
    for cls in df_plot[target_col].unique():
        cls_subset = df_plot[df_plot[target_col] == cls]
        fig.add_trace(go.Scatter3d(
            x=cls_subset[ld_cols[0]], y=cls_subset[ld_cols[1]], z=cls_subset[ld_cols[2]],
            mode='markers', name=str(cls),
            marker=dict(size=3, opacity=0.5),
            hovertemplate=f"Class: {cls}<extra></extra>"
        ))

    # 2. Add Feature Arrows (Loadings) - Numeric Only!
    if survivors:
        df_sampled = original_df.iloc[::stride]
        # Strict numeric enforcement
        original_numeric = df_sampled[survivors].select_dtypes(include=[np.number])
        common_idx = original_numeric.index.intersection(df_plot.index)
        
        if not original_numeric.empty:
            loadings = original_numeric.loc[common_idx].apply(
                lambda x: df_plot.loc[common_idx, ld_cols[:3]].corrwith(x)
            ).T
            
            scale = df_plot[ld_cols[:3]].abs().max().max() * 0.9
            top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(10).index

            for feat in top_features:
                vx, vy, vz = loadings.loc[feat] * scale
                fig.add_trace(go.Scatter3d(
                    x=[0, vx], y=[0, vy], z=[0, vz],
                    mode='lines+text', text=["", feat],
                    line=dict(color='red', width=4),
                    legendgroup="arrows", showlegend=False
                ))

    fig.update_layout(
        title=f"3D Biplot: {pipeline_name}",
        scene=dict(xaxis_title=ld_cols[0], yaxis_title=ld_cols[1], zaxis_title=ld_cols[2]),
        template="plotly_white"
    )
    fig.show()