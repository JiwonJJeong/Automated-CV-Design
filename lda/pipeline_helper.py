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
from pipeline_evaluator import summarize_and_evaluate, evaluate_separability, extract_selected_features

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
    
    # --- AUTOMATIC STANDARD SCALING ---
    from feature_scaling.standard import create_standard_scaled_generator
    
    # Create scaled data directly, no need to keep variance_df
    def variance_factory():
        yield variance_df
    
    scaled_factory = create_standard_scaled_generator(variance_factory)
    scaled_df = pd.concat(list(scaled_factory()), ignore_index=False)
    print(f"✅ Applied standard scaling: {scaled_df.shape}")
    
    # Clean up variance_df to save memory
    del variance_df
    gc.collect()
    print("🧹 Cleaned up original variance data to save memory")
    
    # --- CLASS ASSIGNMENT CHOICE ---
    print("\n" + "="*30 + "\nCLASS ASSIGNMENT\n" + "="*30)
    print("Choose class assignment method:")
    print("1. Default: construct + subconstruct")
    print("2. GMM: Gaussian Mixture Model (shared states)")
    print("3. Spectral: Non-linear spectral clustering")
    print("4. TICA: Kinetic landscape state assignment")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice in ['2', '3', '4']:
        # Create factory from scaled data
        def scaled_data_factory():
            yield scaled_df
            
        if choice == '2':
            from cluster.gmm import run_global_clustering_pipeline
            cluster_params = get_params_minimal('clustering', 'gmm')
            clustered_df = run_global_clustering_pipeline(scaled_data_factory, target_col='construct', **cluster_params)
            method_name = "GMM"
            
        elif choice == '3':
            from cluster.spectral import run_spectral_clustering_pipeline
            cluster_params = get_params_minimal('clustering', 'spectral')
            clustered_df = run_spectral_clustering_pipeline(scaled_data_factory, target_col='construct', **cluster_params)
            method_name = "Spectral"
            
        elif choice == '4':
            from cluster.tica import run_tica_clustering_pipeline
            cluster_params = get_params_minimal('clustering', 'tica')
            clustered_df = run_tica_clustering_pipeline(scaled_data_factory, target_col='construct', **cluster_params)
            method_name = "TICA"
        
        # Add cluster labels as class column
        scaled_df['class'] = 'cluster_' + clustered_df['global_cluster_id'].astype(str)
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
        scaled_df['class'] = scaled_df['construct'] + '_' + scaled_df['subconstruct']
        print(f"Assigned {scaled_df['class'].nunique()} construct-based classes")

    # --- PHASE 2: FEATURE SELECTION (Keep Interactive) ---
    fs_methods = list(set(c['feature_selection'] for c in pipeline_configs))
    fs_paths = {}

    for method in fs_methods:
        path = cache_dir / f"{method}.pkl"
        fs_paths[method] = path
        def fs_exec(params):
            return execute_fs_method(method, lambda: [scaled_df], params)

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
            
            # Collect hyperparameters from both steps
            fs_hyperparams = fs_data.get('hyperparameters', {}) if isinstance(fs_data, dict) else {}
            dr_hyperparams = {
                'category': 'dimensionality_reduction',
                'subcategory': dr_m,
                'parameters': dr_params.copy(),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            final_results[name] = {
                'success': True,
                'data': final_output,
                'config': config,
                'selected_features': selected_features,
                'feature_selection_hyperparameters': fs_hyperparams,
                'dimensionality_reduction_hyperparameters': dr_hyperparams
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
    
    # Use existing evaluation system (variance_df unused, pass original_df for feature correlation analysis)
    # Get original data for correlation analysis
    original_df = pd.concat(list(data_factory()), ignore_index=False)
    summarize_and_evaluate(final_results, variance_df=None, original_df=original_df)
    
    return final_results, scaled_df  # Return scaled_df instead of variance_df


# =============================================================================
# RESULTS SAVING UTILITIES
# =============================================================================

def save_pipeline_results(results, base_df, export_dir="results"):
    """
    Save pipeline results and base dataframe to pickle files.
    
    Parameters:
    -----------
    results : dict
        Pipeline results dictionary
    base_df : pd.DataFrame
        Base dataframe with metadata
    export_dir : str or Path
        Directory to save results (default: "results")
    """
    import pickle
    from pathlib import Path
    
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing results and ask for verification
    results_file = export_dir / "pipeline_results.pkl"
    base_df_file = export_dir / "base_df_metadata.pkl"
    
    if results_file.exists() or base_df_file.exists():
        print(f"⚠️  Existing results found in {export_dir}/:")
        if results_file.exists():
            print(f"   📁 pipeline_results.pkl (already exists)")
        if base_df_file.exists():
            print(f"   📁 base_df_metadata.pkl (already exists)")
        
        response = input("Overwrite existing results? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Save cancelled - existing results preserved")
            return False
    
    try:
        # Save pipeline results with highest protocol for better performance
        with open(results_file, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save base dataframe/metadata
        with open(base_df_file, "wb") as f:
            pickle.dump(base_df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"✅ Results saved to {export_dir}/")
        print(f"   📁 pipeline_results.pkl ({len(results)} pipelines)")
        print(f"   📁 base_df_metadata.pkl ({base_df.shape[0]} rows, {base_df.shape[1]} cols)")
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False
    
    return True


# =============================================================================
# GENERIC INTERACTIVE HELPER
# =============================================================================

def enhance_result_with_hyperparameters(result, params, category, subcategory):
    """
    Enhance a result object with hyperparameter information for tracking and reproducibility.
    """
    if isinstance(result, dict):
        enhanced_result = result.copy()
    else:
        enhanced_result = {'data': result}
    
    # Add hyperparameter metadata
    enhanced_result['hyperparameters'] = {
        'category': category,
        'subcategory': subcategory,
        'parameters': params.copy(),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    return enhanced_result

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
                # Automatically enhance result with hyperparameters when user accepts
                enhanced_result = enhance_result_with_hyperparameters(result, params, param_category, param_subcategory)
                print(f"📝 Hyperparameters automatically attached to {name} results")
                with open(cache_path, 'wb') as f:
                    pickle.dump(enhanced_result, f)
                print(f"✅ Results with hyperparameters cached to {cache_path}")
                break
            else:
                print("Results rejected. Restarting parameter selection...")
        
        except Exception as e:
            print(f"Error executing {name}: {e}")
            retry = input("Retry with different parameters? (Y/n): ").strip().lower()
            if retry == 'n':
                print(f"Skipping {name}.")
                break

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


