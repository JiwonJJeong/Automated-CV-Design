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

# 1. Path Setup (Maintaining your structure)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

# =============================================================================
# VISUALIZATION: THE ELBOW PLOT
# =============================================================================

def plot_elbow_curve(k_range, inertias, optimal_k):
    """Diagnostic plot to justify the number of clusters chosen."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, marker='o', color='#2c3e50', lw=2)
    plt.axvline(x=optimal_k, color='#e74c3c', linestyle='--', label=f'Optimal K: {optimal_k}')
    plt.title("The Elbow Method: Finding the 'True' Number of States", fontsize=14)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-Cluster Variance)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# CORE CLUSTERING PIPELINE
# =============================================================================

def run_global_clustering_pipeline(df_iterator_factory, target_col='class', stride=5, 
                                   max_k=15, show_plots=True):
    """
    Finds shared states across all classes using K-Means and the Elbow Method.
    Assumes features are already reduced.
    """
    
    # --- Pass 1: Sampling for Model Fitting ---
    print(f"Sampling dataset (stride={stride}) to find optimal cluster count...")
    sample_list = []
    for chunk in df_iterator_factory():
        sample_list.append(chunk.iloc[::stride])
    
    df_sample = pd.concat(sample_list, ignore_index=True)
    # Get features excluding target and metadata
    feature_cols = [c for c in get_feature_cols(df_sample) if c != target_col]
    
    # Scaling is mandatory for K-Means (distance-based)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample[feature_cols])
    
    # --- Pass 2: The Elbow Search ---
    print(f"Evaluating K from 2 to {max_k}...")
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        # n_init=auto for modern sklearn efficiency
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    # Automatic detection of the 'elbow' point
    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    optimal_k = kn.knee if kn.knee is not None else 5
    print(f"✅ Elbow detected at K = {optimal_k}")

    if show_plots:
        plot_elbow_curve(k_range, inertias, optimal_k)

    # --- Pass 3: Final Global Fit ---
    print(f"Fitting final K-Means with {optimal_k} clusters...")
    final_kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    final_kmeans.fit(X_scaled)

    # --- Pass 4: Projection (Full Dataset) ---
    print("Projecting cluster labels onto the full dataset...")
    final_data_list = []
    
    for chunk in df_iterator_factory():
        # Apply the scaling learned from the sample
        X_chunk = scaler.transform(chunk[feature_cols])
        
        # Predict which cluster each point belongs to
        cluster_labels = final_kmeans.predict(X_chunk)
        
        # Append the new labels to the original dataframe
        chunk_res = chunk.copy()
        chunk_res['global_cluster_id'] = cluster_labels
        
        final_data_list.append(chunk_res)
    
    gc.collect() # Tidy up memory
    return pd.concat(final_data_list, ignore_index=True)

def analyze_cluster_composition(df, target_col='class', cluster_col='global_cluster_id'):
    """Calculates and visualizes which classes make up each cluster."""
    
    # 1. Create a contingency table (Counts)
    composition_counts = pd.crosstab(df[cluster_col], df[target_col])
    
    # 2. Convert to percentages (Normalized by Cluster)
    # This tells you: "Of the points in Cluster 5, X% are Class 1"
    composition_perc = composition_counts.div(composition_counts.sum(axis=1), axis=0) * 100
    
    print("\n--- Cluster Composition (%) ---")
    print(composition_perc.round(2))

    # 3. Visualization: Stacked Bar Chart
    sns.set_theme(style="white")
    composition_perc.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    
    plt.title("Cluster Composition by Class", fontsize=14)
    plt.xlabel("Global Cluster ID")
    plt.ylabel("Percentage of Cluster")
    plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return composition_perc

if __name__ == "__main__":
    # Example usage:
    # df_factory = create_dataframe_factory("data.csv")
    # df_clustered = run_global_clustering_pipeline(df_factory)
    pass