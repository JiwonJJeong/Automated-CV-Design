#!/usr/bin/env python3
"""
Interactive Pipeline Runner for CV Design

This module provides an interactive pipeline runner that allows users to
run variance → feature selection → dimensionality reduction pipelines
with user intervention and parameter tuning at each step.

Usage:
    from pipeline_helper import run_interactive_pipelines, create_interactive_pipeline_configs
    
    # Create pipeline configurations
    configs = create_interactive_pipeline_configs()
    
    # Run interactive mode
    results = run_interactive_pipelines(data_factory, configs)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Iterator, Callable, Any

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# =============================================================================
# INTERACTIVE PIPELINE RUNNER
# =============================================================================

def get_user_parameters(current_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allow user to modify pipeline parameters interactively.
    
    Args:
        current_params: Current parameter dictionary
        
    Returns:
        Modified parameter dictionary
    """
    print("\n📝 Current Parameters:")
    for key, value in current_params.items():
        print(f"   {key}: {value}")
    
    print("\n💡 Enter new values (press Enter to keep current):")
    modified_params = current_params.copy()
    
    for key, current_value in current_params.items():
        user_input = input(f"   {key} [{current_value}]: ").strip()
        
        if user_input:
            # Try to convert to appropriate type
            try:
                if isinstance(current_value, bool):
                    modified_params[key] = user_input.lower() in ['true', '1', 'yes', 'on']
                elif isinstance(current_value, int):
                    modified_params[key] = int(user_input)
                elif isinstance(current_value, float):
                    modified_params[key] = float(user_input)
                else:
                    modified_params[key] = user_input
            except ValueError:
                print(f"   ⚠️  Could not convert '{user_input}', keeping original value")
    
    return modified_params


def run_single_pipeline(data_factory: Callable, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single pipeline configuration.
    
    Args:
        data_factory: Factory function that yields DataFrames
        config: Pipeline configuration dictionary
        
    Returns:
        Result dictionary with pipeline output
    """
    try:
        # Extract configuration
        fs_method = config['feature_selection']
        dr_method = config['dimensionality_reduction']
        params = config.get('params', {})
        
        print(f"🔄 Running {fs_method} → {dr_method}...")
        
        # Step 1: Variance filtering (always first)
        from variance import variance_filter_pipeline
        variance_result = list(variance_filter_pipeline(
            data_factory(), 
            show_plot=False,
            **params.get('variance', {})
        ))
        
        if not variance_result:
            raise ValueError("Variance filtering returned no results")
        
        variance_df = variance_result[0]
        
        # Create new factory for filtered data
        def filtered_factory():
            yield variance_df
        
        # Step 2: Feature selection
        fs_params = params.get('feature_selection', {})
        if fs_method == 'bpso':
            from BPSO import run_bpso_pipeline
            fs_result = run_bpso_pipeline(filtered_factory, **fs_params)
        elif fs_method == 'mpso':
            from MPSO import run_mpso_pipeline
            fs_result = run_mpso_pipeline(filtered_factory, **fs_params)
        elif fs_method == 'fisher_amino':
            from Fisher_AMINO import run_fisher_amino_pipeline
            fs_result = run_fisher_amino_pipeline(filtered_factory, **fs_params)
        elif fs_method == 'chi_sq_amino':
            from Chi_sq_AMINO import run_feature_selection_pipeline
            fs_result = run_feature_selection_pipeline(filtered_factory, **fs_params)
        else:
            raise ValueError(f"Unknown feature selection method: {fs_method}")
        
        # Step 3: Dimensionality reduction
        dr_params = params.get('dimensionality_reduction', {})
        if dr_method == 'flda':
            from FLDA import run_flda
            dr_result = run_flda(fs_result, **dr_params)
        elif dr_method == 'pca':
            from PCA import run_pca
            dr_result = list(run_pca(fs_result, **dr_params))[0]
        elif dr_method == 'zhlda':
            from ZHLDA import run_zhlda
            dr_result = list(run_zhlda(fs_result, **dr_params))[0]
        elif dr_method == 'mhlda':
            from MHLDA import run_mhlda
            dr_result = run_mhlda(fs_result, **dr_params)
        elif dr_method == 'gdhlda':
            from GDHLDA import run_gdhlda
            dr_result = list(run_gdhlda(fs_result, **dr_params))[0]
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {dr_method}")
        
        return {
            'success': True,
            'pipeline_name': config['name'],
            'variance_result': variance_df,
            'feature_selection_result': fs_result,
            'final_result': dr_result,
            'config': config
        }
        
    except Exception as e:
        return {
            'success': False,
            'pipeline_name': config['name'],
            'error': str(e),
            'config': config
        }


def run_interactive_pipelines(data_factory: Callable, pipeline_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run multiple pipelines with user confirmation at each step.
    
    Args:
        data_factory: Factory function that yields DataFrames
        pipeline_configs: List of pipeline configuration dictionaries
        
    Returns:
        Dictionary of results for completed pipelines
    """
    results = {}
    completed = []
    skipped = []
    failed = []
    
    print(f"🚀 Starting Interactive Pipeline Runner")
    print(f"📊 Total pipelines: {len(pipeline_configs)}")
    print(f"🎯 You'll be prompted for each pipeline")
    
    for i, config in enumerate(pipeline_configs, 1):
        print(f"\n{'='*80}")
        print(f"📍 Pipeline {i}/{len(pipeline_configs)}: {config['name']}")
        print(f"🔗 Method: {config['feature_selection']} → {config['dimensionality_reduction']}")
        
        # Show current parameters
        params = config.get('params', {})
        if params:
            print(f"⚙️  Current Parameters:")
            for category, cat_params in params.items():
                if cat_params:
                    print(f"   {category}:")
                    for param, value in cat_params.items():
                        print(f"     {param}: {value}")
        else:
            print(f"⚙️  Using default parameters")
        
        # User confirmation
        print(f"\n🤔 Options:")
        print(f"   [Enter/Y] - Run with current parameters")
        print(f"   [N]      - Modify parameters before running")
        print(f"   [S]      - Skip this pipeline")
        print(f"   [Q]      - Quit interactive mode")
        
        user_input = input(f"\nYour choice: ").strip().lower()
        
        if user_input == 'q':
            print("🛑 Quitting interactive mode")
            break
        elif user_input == 's':
            print(f"⏭️  Skipping {config['name']}")
            skipped.append(config['name'])
            continue
        elif user_input == 'n':
            # Allow parameter modification
            print(f"\n📝 Modify parameters for {config['name']}:")
            
            if 'params' not in config:
                config['params'] = {}
            
            # Modify each category
            for category in ['variance', 'feature_selection', 'dimensionality_reduction']:
                cat_params = config['params'].get(category, {})
                if cat_params or input(f"Modify {category} parameters? [y/N]: ").strip().lower() == 'y':
                    modified = get_user_parameters(cat_params)
                    config['params'][category] = modified
                    print(f"✅ Updated {category} parameters")
        elif user_input not in ['', 'y', 'yes']:
            print(f"⚠️  Unknown option '{user_input}', skipping...")
            skipped.append(config['name'])
            continue
            
        # Run the pipeline
        print(f"\n🔄 Running {config['name']}...")
        try:
            result = run_single_pipeline(data_factory, config)
            
            if result['success']:
                results[config['name']] = result
                completed.append(config['name'])
                print(f"✅ Completed: {config['name']}")
                
                # Show quick results summary
                final_result = result['final_result']
                print(f"   📏 Final shape: {final_result.shape}")
                feature_cols = [col for col in final_result.columns if col != 'class']
                print(f"   🔧 Features: {len(feature_cols)}")
                if len(feature_cols) <= 10:
                    print(f"   📋 Feature names: {feature_cols}")
                else:
                    print(f"   📋 Feature names: {feature_cols[:5]}...{feature_cols[-3:]}")
            else:
                failed.append(config['name'])
                print(f"❌ Failed: {config['name']}")
                print(f"   🚨 Error: {result['error']}")
                
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted by user")
            break
        except Exception as e:
            failed.append(config['name'])
            print(f"❌ Unexpected error in {config['name']}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 INTERACTIVE PIPELINE RUNNER SUMMARY")
    print(f"✅ Completed: {len(completed)}")
    print(f"⏭️  Skipped: {len(skipped)}")
    print(f"❌ Failed: {len(failed)}")
    
    if completed:
        print(f"\n✅ Successfully completed pipelines:")
        for name in completed:
            result = results[name]
            print(f"   📊 {name}: {result['final_result'].shape}")
    
    if skipped:
        print(f"\n⏭️  Skipped pipelines:")
        for name in skipped:
            print(f"   📋 {name}")
    
    if failed:
        print(f"\n❌ Failed pipelines:")
        for name in failed:
            print(f"   🚨 {name}")
    
    return results


def create_interactive_pipeline_configs() -> List[Dict[str, Any]]:
    """
    Create all pipeline configurations for interactive mode.
    
    Returns:
        List of pipeline configuration dictionaries
    """
    feature_selection_methods = ['bpso', 'mpso', 'fisher_amino', 'chi_sq_amino']
    dimensionality_reduction_methods = ['flda', 'pca', 'zhlda', 'mhlda', 'gdhlda']
    
    configs = []
    for i, fs in enumerate(feature_selection_methods):
        for j, dr in enumerate(dimensionality_reduction_methods):
            config = {
                'name': f'{fs}_to_{dr}',
                'feature_selection': fs,
                'dimensionality_reduction': dr,
                'params': {}  # Will be populated interactively
            }
            configs.append(config)
    
    return configs


if __name__ == "__main__":
    # Example usage for interactive mode
    print("🎮 Interactive Pipeline Runner")
    
    # Create configurations
    configs = create_interactive_pipeline_configs()
    
    # Example data factory (replace with your actual data)
    def example_data_factory():
        # Replace this with your actual data loading
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        data = {}
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        data['class'] = np.random.choice([0, 1, 2], n_samples)
        
        df = pd.DataFrame(data)
        yield df
    
    print("🔧 This example uses sample data. Replace example_data_factory() with your actual data.")
    print(f"📊 Generated {len(configs)} pipeline combinations (4 FS × 5 DR = 20)")
    
    # Run interactive mode
    if input("🚀 Start interactive pipeline runner? [y/N]: ").strip().lower() == 'y':
        results = run_interactive_pipelines(example_data_factory, configs)
        print(f"\n🎉 Interactive session completed!")
        print(f"📊 Results: {len(results)} successful pipelines")
    else:
        print("👋 Exiting...")
