#!/usr/bin/env python3
import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Any
import gc
import inspect
from data_access import METADATA_COLS
from feature_extraction.variance import variance_filter_pipeline

# =============================================================================
# PARAMETER DEFINITIONS (Explanations kept in code, but removed from prints)
# =============================================================================

DEFAULT_PARAMETERS = {
    'variance': {
        'show_plot': {'value': True, 'help': "Show variance plots for analysis"},
        'knee_S': {'value': 1.0, 'help': "Knee detection sensitivity - higher = more features"},
        'outlier_multiplier': {'value': 3.0, 'help': "Outlier detection threshold multiplier"},
        'fallback_percentile': {'value': 90, 'help': "Fallback percentile for threshold"},
        'min_clean_ratio': {'value': 0.5, 'help': "Min fraction of features to keep in 'clean' set before reverting to full data"}
    },
    'feature_selection': {
        'bpso': {
            'candidate_limit': {'value': None, 'help': "Max features for optimization (None = Dynamic selection via Fisher Score Knee)"},
            'knee_S': {'value': 2.0, 'help': "Sensitivity for the candidate filtering knee - higher = more features"},
            'population_size': {'value': 20, 'help': "Swarm size - more particles = better exploration"},
            'max_iter': {'value': 30, 'help': "Max iterations - higher = more optimization"},
            'w': {'value': 0.729, 'help': "Inertia weight - controls exploration vs exploitation"},
            'c1': {'value': 1.49445, 'help': "Cognitive parameter - individual learning influence"},
            'c2': {'value': 1.49445, 'help': "Social parameter - swarm influence"},
            'stride': {'value': 10, 'help': "Data sampling stride - higher = less data"}
        },
        'mpso': {
            'dims': {'value': None, 'help': "Output dimensions (None = default 5)"},
            'candidate_limit': {'value': None, 'help': "Max features for optimization (None = Dynamic selection via Fisher Score Knee)"},
            'knee_S': {'value': 2.0, 'help': "Sensitivity for the candidate filtering knee - higher = more features"},
            'max_iter': {'value': 10, 'help': "PSO iterations - higher = better optimization"},
            'population_size': {'value': 40, 'help': "Swarm population size - larger = better search"},
            'alpha': {'value': 0.9, 'help': "Accuracy-sparsity tradeoff - 0.0=accuracy, 1.0=sparsity"},
            'threshold': {'value': 0.5, 'help': "Feature selection threshold - higher = fewer features"},
            'redundancy_weight': {'value': 0.2, 'help': "Dimension independence penalty (0.0=none, 1.0=max redundancy reduction)"},
            'stride': {'value': 15, 'help': "Data sampling stride - higher = less data"}
        },
        'fisher_amino': {
            'max_outputs': {'value': None, 'help': "Target features (None = Dynamic selection via AMINO Distortion Jump method)"},
            'knee_S': {'value': 2.0, 'help': "Sensitivity for the initial Fisher candidate knee - higher = more features"}
        },
        'chi_sq_amino': {
            'stride': {'value': 25, 'help': "Data sampling stride - higher = less memory usage"},
            'max_amino': {'value': None, 'help': "Target features (None = Dynamic selection via AMINO Distortion Jump method)"},
            'q_bins': {'value': 5, 'help': "Quantile bins for chi-square - higher = more resolution"},
            'sample_rows': {'value': 20000, 'help': "Sample size for binning - higher = more accurate bins"},
            'knee_S': {'value': 5.0, 'help': "Sensitivity for the initial Chi-Square candidate knee - higher = more features"}
        }
    },
    'dimensionality_reduction': {
        'flda': {
            'num_eigenvector': {'value': None, 'help': "Output dimensions (None = dynamic 95% variance)"},
            'regularization': {'value': 1e-6, 'help': "L1 regularization - prevents overfitting"}
        },
        'pca': {
            'num_eigenvector': {'value': None, 'help': "Output dimensions (None = dynamic 95% variance)"},
            'svd_solver': {'value': 'auto', 'help': "SVD algorithm - 'auto' recommended"}
        },
        'zhlda': {
            'num_eigenvector': {'value': None, 'help': "Output dimensions (None = dynamic 95% variance)"},
            'learning_rate': {'value': 0.0001, 'help': "Gradient descent step size"},
            'num_iteration': {'value': 1000, 'help': "Max iterations - higher = better convergence"},
            'stop_crit': {'value': 30, 'help': "Convergence patience - prevents early stopping"},
            'convergence_threshold': {'value': 1e-5, 'help': "Convergence tolerance - smaller = more precise"}
        },
        'mhlda': {
            'num_eigenvector': {'value': None, 'help': "Output dimensions (None = dynamic 95% variance)"},
            'regularization': {'value': 1e-4, 'help': "L1 regularization - prevents overfitting"},
            'learning_rate': {'value': 0.0001, 'help': "Gradient descent step size"},
            'num_iteration': {'value': 1000, 'help': "Max iterations - higher = better convergence"},
            'stop_crit': {'value': 30, 'help': "Convergence patience - prevents early stopping"},
            'convergence_threshold': {'value': 1e-5, 'help': "Convergence tolerance - smaller = more precise"}
        },
        'gdhlda': {
            'num_eigenvector': {'value': None, 'help': "Output dimensions (None = dynamic 95% variance)"},
            'learning_rate': {'value': 0.0001, 'help': "Gradient descent step size"},
            'num_iteration': {'value': 1000, 'help': "Max iterations - higher = better convergence"},
            'stop_crit': {'value': 30, 'help': "Convergence patience - prevents early stopping"},
            'convergence_threshold': {'value': 1e-5, 'help': "Convergence tolerance - smaller = more precise"}
        }
    },
    'clustering': {
        'gmm': {
            'stride': {'value': 10, 'help': "Data sampling stride - higher = faster but less precise"},
            'max_k': {'value': 15, 'help': "Maximum clusters to test - higher = more options"},
            'S': {'value': 1.0, 'help': "Knee sensitivity - higher = more clusters, lower = fewer"},
            'show_plots': {'value': True, 'help': "Show BIC diagnostics plot"}
        },
        'spectral': {
            'stride': {'value': 10, 'help': "Data sampling stride - higher = faster but less precise"},
            'max_k': {'value': 15, 'help': "Maximum clusters to test - higher = more options"},
            'n_components': {'value': 50, 'help': "Spectral embedding dimensions - higher = more complex"},
            'S': {'value': 1.0, 'help': "Knee sensitivity - higher = more clusters, lower = fewer"},
            'show_plots': {'value': True, 'help': "Show Elbow diagnostics plot"}
        },
        'tica': {
            'stride': {'value': 10, 'help': "Data sampling stride - higher = faster but less precise"},
            'lag_time': {'value': 10, 'help': "TICA lag time - higher = slower kinetic processes"},
            'max_components': {'value': 5, 'help': "Max TICA components to evaluate"},
            'max_k': {'value': 15, 'help': "Maximum clusters to test - higher = more options"},
            'S': {'value': 1.0, 'help': "Knee sensitivity - higher = more clusters, lower = fewer"},
            'show_plots': {'value': True, 'help': "Show TICA landscape plot"}
        }
    }
}

# =============================================================================
# MINIMAL INTERACTIVE HELPERS (Hyperparameters only)
# =============================================================================

def get_params_minimal(category: str, subcategory: str = None) -> Dict[str, Any]:
    """Displays only hyperparameters and returns user modifications."""
    if subcategory:
        defaults_struct = DEFAULT_PARAMETERS[category][subcategory]
        header = f"[{category.upper()} : {subcategory.upper()}]"
        
        print(f"\n{header}")
        print("📋 Parameter Controls:")
        if category == 'feature_selection':
            descriptions = {
                'bpso': "Binary PSO - swarm intelligence for feature selection",
                'mpso': "Multi-objective PSO - balances accuracy vs sparsity", 
                'fisher_amino': "Fisher scoring with AMINO dimensionality reduction",
                'chi_sq_amino': "Chi-square selection with AMINO reduction"
            }
            if subcategory in descriptions:
                print(f"  🔬 Method: {descriptions[subcategory]}")
        elif category == 'dimensionality_reduction':
            descriptions = {
                'flda': "Fisher Linear Discriminant Analysis",
                'pca': "Principal Component Analysis", 
                'zhlda': "Zero-order Hamiltonian LDA",
                'mhlda': "Modified Hamiltonian LDA",
                'gdhlda': "Gradient Descent Hamiltonian LDA"
            }
            if subcategory in descriptions:
                print(f"  📊 Method: {descriptions[subcategory]}")
        elif category == 'clustering':
            descriptions = {
                'gmm': "Gaussian Mixture Model (with BIC visualization)",
                'spectral': "Spectral Clustering (Nystroem + K-means)",
                'tica': "Time-lagged ICA (Kinetic landscape mapping)"
            }
            if subcategory in descriptions:
                print(f"  🎯 Method: {descriptions[subcategory]}")
    else:
        defaults_struct = DEFAULT_PARAMETERS[category]
        header = f"[{category.upper()}]"
        print(f"\n{header}")
        
    print("⚙️  Hyperparameters:")
    params = {}
    for k, info in defaults_struct.items():
        v = info['value']
        blurb = info['help']
        print(f"  {k}: {v} - {blurb}")
        params[k] = v

    modify = input("\nModify? (y/N): ").strip().lower()
    
    if modify == 'y':
        for k, info in defaults_struct.items():
            v = info['value']
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
# CORE PIPELINE ENGINE (AUTOMATED DR PHASE)
# =============================================================================

def flatten_defaults(nested_dict):
    """Flattens {'parm': {'value': val, 'help': ...}} into {'parm': val}"""
    return {k: v['value'] if isinstance(v, dict) and 'value' in v else v 
            for k, v in nested_dict.items()}

def run_interactive_pipeline(data_factory, pipeline_configs, class_assignment_func=None):
    cache_dir = Path("pipeline_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # --- PHASE 1: VARIANCE (Run before all else) ---
    print("\n" + "="*30 + "\nPHASE 1: VARIANCE\n" + "="*30)
    
    from feature_extraction.variance import variance_filter_pipeline
    
    # Use caching for variance filtering
    variance_cache_path = cache_dir / "variance.pkl"
    def variance_exec(params):
        variance_result_gen = variance_filter_pipeline(data_factory, **params)
        return pd.concat(list(variance_result_gen), ignore_index=False)
    
    variance_df = run_interactive_step_generic("VARIANCE", variance_cache_path, 'variance', None, variance_exec)
    print(f"Variance Output: {variance_df.shape}")
    
    # --- CLASS ASSIGNMENT CHOICE ---
    print("\n" + "="*30 + "\nCLASS ASSIGNMENT\n" + "="*30)
    print("Choose class assignment method:")
    print("1. Default: construct + subconstruct")
    print("2. GMM: Gaussian Mixture Model (shared states)")
    print("3. Spectral: Non-linear spectral clustering")
    print("4. TICA: Kinetic landscape state assignment")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice in ['2', '3', '4']:
        # Create factory from variance data
        def variance_factory():
            yield variance_df
            
        if choice == '2':
            from cluster.gmm import run_global_clustering_pipeline
            cluster_params = get_params_minimal('clustering', 'gmm')
            clustered_df = run_global_clustering_pipeline(variance_factory, target_col='construct', **cluster_params)
            method_name = "GMM"
            
        elif choice == '3':
            from cluster.spectral import run_spectral_clustering_pipeline
            cluster_params = get_params_minimal('clustering', 'spectral')
            # Disable internal plot to show high-res one from helper
            cluster_params['show_plots'] = False
            clustered_df = run_spectral_clustering_pipeline(variance_factory, target_col='construct', **cluster_params)
            method_name = "Spectral"
            
        elif choice == '4':
            try:
                from cluster.tica import run_validated_tica_pipeline
                cluster_params = get_params_minimal('clustering', 'tica')
                # Disable internal plot to show high-res one from helper
                cluster_params['show_plots'] = False
                clustered_df = run_validated_tica_pipeline(variance_factory, target_col='construct', **cluster_params)
                method_name = "TICA"
            except ImportError as e:
                # ...
                if 'deeptime' in str(e):
                    print("\n❌ Error: 'deeptime' package is required for TICA but not found.")
                    print("Please install it with: pip install deeptime")
                    return None, None
                else:
                    raise e
        
        # Add cluster labels as class column
        variance_df['class'] = 'cluster_' + clustered_df['global_cluster_id'].astype(str)
        print(f"Assigned {clustered_df['global_cluster_id'].nunique()} {method_name}-based classes")
        
        if choice == '2':
            from cluster.gmm import analyze_cluster_composition
            # High-res labels on-the-fly
            refined_labels = clustered_df['construct'] + ' | ' + clustered_df['subconstruct']
            analyze_cluster_composition(clustered_df, target_col=refined_labels, cluster_col='global_cluster_id')
        elif choice == '3':
            from cluster.spectral import analyze_cluster_composition
            refined_labels = clustered_df['construct'] + ' | ' + clustered_df['subconstruct']
            analyze_cluster_composition(clustered_df, target_col=refined_labels, cluster_col='global_cluster_id')
        elif choice == '4':
            from cluster.tica import analyze_cluster_composition
            refined_labels = clustered_df['construct'] + ' | ' + clustered_df['subconstruct']
            analyze_cluster_composition(clustered_df, target_col=refined_labels, cluster_col='global_cluster_id')
        
    else:
        # Default class assignment
        variance_df['class'] = variance_df['construct'] + '_' + variance_df['subconstruct']
        print(f"Assigned {variance_df['class'].nunique()} construct-based classes")

    # --- PHASE 2: FEATURE SELECTION (Keep Interactive) ---
    fs_methods = list(set(c['feature_selection'] for c in pipeline_configs))
    fs_paths = {}

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

            dr_params = flatten_defaults(DEFAULT_PARAMETERS['dimensionality_reduction'].get(dr_m, {}))
            
            # This now includes the dynamic dimension safety check we discussed
            final_output = execute_dr_method(dr_m, fs_data, dr_params)
            
            # Extract and display selected features
            selected_features = extract_selected_features(fs_data)
            print(f"Selected features ({len(selected_features)}): {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")
            
            final_results[name] = {
                'success': True,
                'data': final_output,
                'config': config,
                'selected_features': selected_features
            }
            
            # --- AUTO-SAVE Result ---
            with open(cache_path, 'wb') as f:
                pickle.dump(final_results[name], f)
                
            print(f"Result: SUCCESS (Saved to {cache_path.name})")

        except Exception as e:
            print(f"❌ Error in pipeline {name}: {e}")
        
        finally:
            if 'fs_data' in locals(): del fs_data
            gc.collect()

    print("\n✨ All automated pipelines complete.")
    
    # Calculate evaluation metrics and create leaderboard
    print("\n📊 CALCULATING EVALUATION METRICS...")
    
    # Use existing evaluation system
    summarize_and_evaluate(final_results, variance_df)
    
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

def extract_selected_features(fs_result):
    """Extract selected feature names from different FS method result formats."""
    # Logic: Favor explicit attrs, then columns, then dict keys
    if hasattr(fs_result, 'attrs') and 'selected_features' in fs_result.attrs:
        return fs_result.attrs['selected_features']
    
    if hasattr(fs_result, 'columns'):
        exclude = ['class', 'target'] + list(METADATA_COLS)
        return [col for col in fs_result.columns if col not in exclude]
    
    if isinstance(fs_result, dict):
        if 'selected_features' in fs_result:
            return fs_result['selected_features']
        if 'data' in fs_result and hasattr(fs_result['data'], 'attrs') and 'selected_features' in fs_result['data'].attrs:
            return fs_result['data'].attrs['selected_features']
        if 'X_selected' in fs_result:
            if hasattr(fs_result['X_selected'], 'columns'):
                exclude = ['class', 'target'] + list(METADATA_COLS)
                return [col for col in fs_result['X_selected'].columns if col not in exclude]
            return [f"feature_{i}" for i in range(fs_result['X_selected'].shape[1])]
        return list(fs_result.keys())
    elif hasattr(fs_result, '__len__'):
        # List or similar
        return list(fs_result)
    else:
        return ["Unknown format"]

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

def execute_dr_method(method, fs_result, params):
    """
    Executes dimensionality reduction with dynamic dimension capping.
    """
    # fs_result is now back to being a DataFrame (MPSO metadata move to .attrs)
    # Ensure it's treated correctly as a DataFrame
    if isinstance(fs_result, dict) and 'X_selected' in fs_result:
        fs_result = fs_result['X_selected']

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
    
    # Target and metadata columns should not be used in DR
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

    # Handle parameters that might still be in {'value': X, 'help': Y} format
    def get_val(p_dict, key, default):
        raw = p_dict.get(key, default)
        if isinstance(raw, dict) and 'value' in raw:
            return raw['value']
        return raw

    requested_dim = get_val(params, 'num_eigenvector', get_val(params, 'n_components', 2))

    # Capping Logic: Only apply if a specific integer was requested
    if requested_dim is not None:
        if requested_dim > limit:
            if limit < 1:
                print(f"🛑 Error: {method_str} cannot run. Only {actual_feat_count} features available.")
                return None
            
            print(f"⚠️  Cap: {method_str} output reduced from {requested_dim} to {limit}")
            requested_dim = limit

    # Update params with the resolved value (could be int or None)
    params['num_eigenvector'] = requested_dim
    params['n_components'] = requested_dim

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

def evaluate_separability(df, target_col='class', selected_features=None, return_individual=False):
    """
    Calculate Fisher separability score (SB/SW ratio).
    SB: Between-class scatter (how far apart class means are)
    SW: Within-class scatter (how tight the clusters are)
    """
    from data_access import METADATA_COLS
    
    if selected_features is None:
        # Auto-detect numeric features, excluding metadata
        exclude = [target_col, 'target'] + list(METADATA_COLS)
        features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    else:
        # Use only the selected features that are actually present
        features = [f for f in selected_features if f in df.columns]
    
    if not features:
        return {} if return_individual else 0
        
    y = df[target_col].values
    unique_y = np.unique(y)
    overall_mean = df[features].mean(axis=0).values
    
    # Pre-calculate class statistics for speed
    class_data = {}
    for cls in unique_y:
        df_c = df[df[target_col] == cls][features]
        class_data[cls] = {
            'n': len(df_c),
            'mean': df_c.mean(axis=0).values,
            'var_sum': ((df_c - df_c.mean(axis=0))**2).sum(axis=0).values
        }
    
    individual_scores = {}
    total_sb = 0
    total_sw = 0
    
    for i, feat in enumerate(features):
        sb_feat = 0
        sw_feat = 0
        for cls in unique_y:
            stats = class_data[cls]
            sb_feat += stats['n'] * (stats['mean'][i] - overall_mean[i])**2
            sw_feat += stats['var_sum'][i]
        
        individual_scores[feat] = sb_feat / (sw_feat + 1e-9)
        total_sb += sb_feat
        total_sw += sw_feat
    
    if return_individual:
        return individual_scores
    
    # Revert to Pooled Ratio (Total SB / Total SW) for the aggregate score
    # to maintain consistency with historical results.
    return total_sb / (total_sw + 1e-9)

def get_feature_importance(original_df, transformed_df, selected_features=None, target_col='class'):
    """Maps selected original features back to new LDs using Correlation."""
    from data_access import METADATA_COLS
    
    if selected_features is None:
        # If no selected features specified, use all numeric features
        original_numeric = original_df.select_dtypes(include=[np.number])
        original_numeric = original_numeric.drop(columns=list(METADATA_COLS), errors='ignore')
        features_to_correlate = original_numeric.columns.tolist()
    else:
        # Use only the selected features
        features_to_correlate = [f for f in selected_features if f in original_df.columns]
        original_numeric = original_df[features_to_correlate]
    
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
    for name, res in results.items():
        # Handle both Dict (pipeline result) and raw DataFrame (old cache)
        if isinstance(res, dict) and 'data' in res:
            df = res['data']
            # Recover features from the dict or the dataframe attributes (MPSO)
            selected_features = res.get('selected_features', [])
            if not selected_features and hasattr(df, 'attrs') and 'selected_features' in df.attrs:
                selected_features = df.attrs['selected_features']
        else:
            df = res
            selected_features = getattr(df, 'attrs', {}).get('selected_features', [])
            
        # Fallback: if selected_features are missing (old cache), try to infer from fs method name
        if not selected_features:
            fs_method = name.split('_to_')[0]
            fs_cache_path = Path("pipeline_cache") / f"{fs_method}.pkl"
            if fs_cache_path.exists():
                try:
                    with open(fs_cache_path, 'rb') as f:
                        fs_obj = pickle.load(f)
                        selected_features = extract_selected_features(fs_obj)
                except:
                    pass

        # We evaluate the quality of the TRANSFORMATION
        score = evaluate_separability(df)
        scored_list.append({
            'name': name, 
            'score': score, 
            'df': df,
            'selected_features': selected_features
        })

    scored_list.sort(key=lambda x: x['score'], reverse=True)

    for rank, item in enumerate(scored_list, 1):
        name, total_score, df = item['name'], item['score'], item['df']
        used_features = item['selected_features']
        ld_cols = [c for c in df.columns if c != 'class']
        
        # Verify used_features against original_df to ensure they are numeric and exist
        used_features = [f for f in used_features if f in original_df.columns and pd.api.types.is_numeric_dtype(original_df[f])]
        print(f"{rank:<5} | {name:<25} | {total_score:<12.4f} | {len(ld_cols)}")
        print(f"      📉 Selection Stats: {len(used_features)} numeric features available for mapping.")
        sys.stdout.flush()

        try:
            if used_features:
                original_numeric = original_df[used_features] 
                common_idx = original_numeric.index.intersection(df.index)
                
                if len(common_idx) < 10:
                    print(f"      ↳ Warning: Low index overlap ({len(common_idx)} rows). Check for filter mismatch.")
                
                # Calculate per-component separability
                comp_scores = evaluate_separability(df, target_col='class', return_individual=True)
                total_comp_score = sum(comp_scores.values()) + 1e-9
                
                for ld in ld_cols:
                    if ld not in df.columns: continue
                    
                    # 1. Separability contribution %
                    c_score = comp_scores.get(ld, 0)
                    c_pct = (c_score / total_comp_score) * 100
                    
                    # 2. Top 5 features by absolute correlation
                    corrs = original_numeric.loc[common_idx].corrwith(df.loc[common_idx, ld])
                    top_5 = corrs.abs().sort_values(ascending=False).head(5)
                    
                    feat_info = ", ".join([f"{k} ({corrs[k]:.2f})" for k in top_5.index])
                    print(f"      ↳ {ld} ({c_pct:.1f}%): {feat_info}")
                sys.stdout.flush()
            else:
                print("      ↳ Info: No numeric survivors found for feature mapping.")
                sys.stdout.flush()
        except Exception as e:
            print(f"      ↳ Feature Mapping Error: {e}")
            sys.stdout.flush()
            
        if rank <= 3:
            try:
                if len(ld_cols) >= 3:
                    visualize_cluster_biplot_3d(original_df, df, name, survivors=used_features)
                elif len(ld_cols) == 2:
                    visualize_cluster_biplot(original_df, df, name, survivors=used_features)
                elif len(ld_cols) == 1:
                    visualize_cluster_1d(df, name) # Add this call!
            except Exception as viz_e:
                print(f"      ↳ Visualization Error: {viz_e}")
                sys.stdout.flush()
        print("-" * 100)
        sys.stdout.flush()

    import gc
    gc.collect()

def visualize_cluster_1d(transformed_df, pipeline_name, target_col='class'):
    import plotly.express as px
    ld_col = [c for c in transformed_df.columns if c != target_col][0]
    
    fig = px.strip(transformed_df, 
                   x=ld_col, 
                   y=target_col, 
                   color=target_col,
                   title=f"1D Distribution: {pipeline_name}",
                   labels={ld_col: "Component 1"},
                   template="plotly_white")
    
    fig.update_layout(showlegend=False)
    fig.show()

def visualize_cluster_biplot_3d(original_df, transformed_df, pipeline_name, target_col='class', stride=20, survivors=None):
    import plotly.graph_objects as go
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    centered_df = transformed_df.copy()
    centered_df[ld_cols] = transformed_df[ld_cols] - transformed_df[ld_cols].mean()
    df_plot = centered_df.iloc[::stride]
    
    fig = go.Figure()

    # 1. Plot Clusters with Metadata Hover
    # Identify metadata: only include standardized metadata columns
    meta_cols = [c for c in original_df.columns if c in METADATA_COLS]
    
    for cls in df_plot[target_col].unique():
        cls_subset = df_plot[df_plot[target_col] == cls]
        
        # Pull metadata for this subset
        common_idx = cls_subset.index.intersection(original_df.index)
        meta_subset = original_df.loc[common_idx, meta_cols]
        
        fig.add_trace(go.Scatter3d(
            x=cls_subset[ld_cols[0]], y=cls_subset[ld_cols[1]], z=cls_subset[ld_cols[2]],
            mode='markers', name=str(cls),
            marker=dict(size=3, opacity=0.8),
            customdata=meta_subset,
            hovertemplate="<b>Class: %{name}</b><br>" + 
                          "<br>".join([f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(meta_cols)]) + 
                          "<extra></extra>"
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

def visualize_cluster_biplot(original_df, transformed_df, pipeline_name, target_col='class', stride=20, survivors=None):
    import plotly.graph_objects as go
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    centered_df = transformed_df.copy()
    centered_df[ld_cols] = transformed_df[ld_cols] - transformed_df[ld_cols].mean()
    df_plot = centered_df.iloc[::stride]
    
    fig = go.Figure()

    # 1. Plot Clusters with Metadata Hover
    meta_cols = [c for c in original_df.columns if c in METADATA_COLS]
    
    for cls in df_plot[target_col].unique():
        cls_subset = df_plot[df_plot[target_col] == cls]
        
        # Pull metadata for this subset
        common_idx = cls_subset.index.intersection(original_df.index)
        meta_subset = original_df.loc[common_idx, meta_cols]

        fig.add_trace(go.Scatter(
            x=cls_subset[ld_cols[0]], y=cls_subset[ld_cols[1]],
            mode='markers', name=str(cls),
            marker=dict(size=8, opacity=0.6),
            customdata=meta_subset,
            hovertemplate="<b>Class: %{name}</b><br>" + 
                          "<br>".join([f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(meta_cols)]) + 
                          "<extra></extra>"
        ))

    # 2. Add Feature Arrows (Loadings)
    if survivors:
        df_sampled = original_df.iloc[::stride]
        original_numeric = df_sampled[survivors].select_dtypes(include=[np.number])
        common_idx = original_numeric.index.intersection(df_plot.index)
        
        if not original_numeric.empty:
            loadings = original_numeric.loc[common_idx].apply(
                lambda x: df_plot.loc[common_idx, ld_cols[:2]].corrwith(x)
            ).T
            
            scale = df_plot[ld_cols[:2]].abs().max().max() * 0.9
            top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(10).index

            for feat in top_features:
                vx, vy = loadings.loc[feat] * scale
                fig.add_trace(go.Scatter(
                    x=[0, vx], y=[0, vy],
                    mode='lines+text', text=["", feat],
                    textposition="top center",
                    line=dict(color='red', width=2),
                    legendgroup="arrows", showlegend=False
                ))

    fig.update_layout(
        title=f"2D Biplot: {pipeline_name}",
        xaxis_title=ld_cols[0], yaxis_title=ld_cols[1],
        template="plotly_white"
    )
    fig.show()