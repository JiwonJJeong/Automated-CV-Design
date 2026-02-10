import numpy as np
import pandas as pd
import gc
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture



# 1. Path Setup (Maintaining your structure)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

# =============================================================================
# VISUALIZATION: THE ELBOW PLOT
# =============================================================================

def plot_bic_curve(k_range, bics, optimal_k):
    """Diagnostic plot to justify the number of components chosen."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, bics, marker='o', color='#2c3e50', lw=2)
    plt.axvline(x=optimal_k, color='#e74c3c', linestyle='--', label=f'Optimal K (Min BIC): {optimal_k}')
    plt.title("GMM BIC Curve: Finding the Optimal Number of States", fontsize=14)
    plt.xlabel("Number of Components (k)")
    plt.ylabel("BIC Score (Lower is Better)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# CORE CLUSTERING PIPELINE
# =============================================================================

def run_global_clustering_pipeline(df_iterator_factory, target_col='class', stride=5, 
                                   max_k=15, S=1.0, show_plots=True):
    """
    Finds shared states across all classes using Gaussian Mixture Models and BIC.
    Assumes features are already reduced.
    """
    
    # --- Pass 1: Sampling for Model Fitting ---
    print(f"Sampling dataset (stride={stride}) to find optimal GMM components...")
    sample_list = []
    for chunk in df_iterator_factory():
        sample_list.append(chunk.iloc[::stride])
    
    df_sample = pd.concat(sample_list, ignore_index=False)
    # Get features excluding target and metadata
    feature_cols = [c for c in get_feature_cols(df_sample) if c != target_col]
    
    # Scaling is mandatory for GMM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample[feature_cols])
    
    # --- Pass 2: The BIC Search ---
    k_range = range(2, max_k + 1)
    bics = []
    for k in k_range:
        # Use higher reg_covar and max_iter for stability during search
        gmm = GaussianMixture(
            n_components=k, 
            random_state=42, 
            n_init=1, 
            covariance_type='diag', 
            reg_covar=1e-4,  # Increased for stability
            max_iter=500,    # Increased
            tol=1e-3         # Slightly relaxed to ensure convergence
        )
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))
    
    # Optimal K Detection using KneeLocator on BIC curve
    # BIC curves are typically 'convex' and 'decreasing' towards the minimum
    kn = KneeLocator(list(k_range), bics, curve='convex', direction='decreasing', S=S)
    optimal_k = kn.knee if kn.knee is not None else k_range[np.argmin(bics)]
    
    print(f"✅ BIC analysis complete. Optimal K (S={S}): {optimal_k}")

    if show_plots:
        plot_bic_curve(k_range, bics, optimal_k)

    # --- Pass 3: Final Global Fit ---
    print(f"Fitting final GMM with {optimal_k} components...")
    final_gmm = GaussianMixture(
        n_components=optimal_k, 
        random_state=42, 
        n_init=10,           # Higher n_init for the final model
        covariance_type='diag', 
        reg_covar=1e-4, 
        max_iter=1000,       # Maximum patience
        tol=1e-4
    )
    final_gmm.fit(X_scaled)

    # --- Pass 4: Projection (Full Dataset) ---
    print("Assigning GMM state labels to the full dataset...")
    final_data_list = []
    
    for chunk in df_iterator_factory():
        # Apply the scaling learned from the sample
        X_chunk = scaler.transform(chunk[feature_cols])
        
        # Predict which component each point belongs to
        cluster_labels = final_gmm.predict(X_chunk)
        
        # Append the new labels to the original dataframe
        chunk_res = chunk.copy()
        chunk_res['global_cluster_id'] = cluster_labels
        
        final_data_list.append(chunk_res)
    
    gc.collect() # Tidy up memory
    result_df = pd.concat(final_data_list, ignore_index=False)
    
    # Preserve metadata attributes (selected_features, etc.)
    # We pull attributes from df_sample which represents the input's metadata
    if hasattr(df_sample, 'attrs'):
        result_df.attrs.update(df_sample.attrs)
        
    return result_df

def analyze_cluster_composition(df, target_col='class', cluster_col='global_cluster_id'):
    """Calculates and visualizes which classes make up each cluster."""
    
    # 1. Create a contingency table (Counts)
    if isinstance(target_col, str):
        if target_col not in df.columns:
            print(f"⚠️  Cannot analyze composition: '{target_col}' not found in results. Available: {df.columns.tolist()}")
            return
        target_values = df[target_col]
        target_name = target_col
    else:
        # Handle case where target_col is a Series or array passed directly
        target_values = target_col
        target_name = "Refined Labels"
        
    composition_counts = pd.crosstab(df[cluster_col], target_values)
    
    # 2. Convert to percentages (Normalized by Cluster)
    # This tells you: "Of the points in Cluster 5, X% are Class 1"
    composition_perc = composition_counts.div(composition_counts.sum(axis=1), axis=0) * 100
    
    print("\n--- Cluster Composition (%) ---")
    print(composition_perc.round(2))

    # 3. Visualization: Stacked Bar Chart (Using Raw Counts)
    sns.set_theme(style="white")
    composition_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    
    plt.title("Cluster Population and Composition", fontsize=14)
    plt.xlabel("Global Cluster ID")
    plt.ylabel("Number of Points")
    plt.legend(title=target_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return composition_perc

if __name__ == "__main__":
    # Example usage:
    # df_factory = create_dataframe_factory("data.csv")
    # df_clustered = run_global_clustering_pipeline(df_factory)
    pass