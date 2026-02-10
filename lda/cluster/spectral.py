import numpy as np
import pandas as pd
import gc
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

# 1. Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_elbow_curve(k_range, inertias, optimal_k):
    """Diagnostic plot to justify the number of clusters chosen."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, marker='o', color='#8e44ad', lw=2) # Purple for Spectral
    plt.axvline(x=optimal_k, color='#e74c3c', linestyle='--', label=f'Optimal K: {optimal_k}')
    plt.title("Spectral Clustering Elbow Method (in Embedded Space)", fontsize=14)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-Cluster Variance)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_cluster_composition(df, target_col='class', cluster_col='global_cluster_id'):
    """Calculates and visualizes which classes make up each cluster."""
    
    # 1. Create a contingency table (Counts)
    if isinstance(target_col, str):
        if target_col not in df.columns:
            print(f"⚠️  Cannot analyze composition: '{target_col}' not found. Available: {df.columns.tolist()}")
            return
        target_values = df[target_col]
        target_name = target_col
    else:
        target_values = target_col
        target_name = "Refined Labels"
        
    composition_counts = pd.crosstab(df[cluster_col], target_values)
    
    # 2. Convert to percentages
    composition_perc = composition_counts.div(composition_counts.sum(axis=1), axis=0) * 100
    
    print("\n--- Cluster Composition (%) ---")
    print(composition_perc.round(2))

    # 3. Visualization (Using Raw Counts)
    sns.set_theme(style="white")
    composition_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title("Cluster Population and Composition", fontsize=14)
    plt.xlabel("Global Cluster ID")
    plt.ylabel("Number of Points")
    plt.legend(title=target_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return composition_perc

# =============================================================================
# CORE SPECTRAL PIPELINE
# =============================================================================

def run_spectral_clustering_pipeline(df_iterator_factory, target_col='class', stride=5, 
                                     max_k=15, n_components=100, S=1.0, show_plots=True):
    """
    Performs Spectral Clustering via Nystroem Approximation + K-Means.
    
    Args:
        n_components (int): Dimensions of the spectral embedding. 
                            Higher = more complex non-linear shapes captured.
    """
    
    # --- Pass 1: Sampling for Embedding Learning ---
    print(f"Sampling dataset (stride={stride}) to learn Spectral Embedding...")
    sample_list = []
    for chunk in df_iterator_factory():
        sample_list.append(chunk.iloc[::stride])
    
    df_sample = pd.concat(sample_list, ignore_index=False)
    feature_cols = [c for c in get_feature_cols(df_sample) if c != target_col]
    
    # Standard Scaling is critical before Kernel methods
    scaler = StandardScaler()
    X_sample = scaler.fit_transform(df_sample[feature_cols])
    
    # --- Pass 2: Nystroem Approximation (The "Spectral" Magic) ---
    # This approximates the eigenvalues/vectors of the RBF kernel matrix
    # without calculating the full N^2 matrix.
    print(f"Computing Nystroem Spectral Embedding (n_components={n_components})...")
    spectral_map = Nystroem(kernel='rbf', gamma=None, n_components=n_components, random_state=42)
    X_embedded = spectral_map.fit_transform(X_sample)
    
    # --- Pass 3: The Elbow Search (in Embedded Space) ---
    print(f"Evaluating K from 2 to {max_k} in spectral space...")
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        km.fit(X_embedded)
        inertias.append(km.inertia_)
    
    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing', S=S)
    optimal_k = kn.knee if kn.knee is not None else 5
    print(f"✅ Optimal Spectral Clusters: {optimal_k}")

    if show_plots:
        plot_elbow_curve(k_range, inertias, optimal_k)

    # --- Pass 4: Final Fit in Embedded Space ---
    print(f"Fitting final K-Means model on spectral components...")
    final_kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    final_kmeans.fit(X_embedded)

    # --- Pass 5: Projection (Full Dataset) ---
    print("Projecting FULL dataset into Spectral Space and assigning labels...")
    
    # 1. Clear out Pass 1-4 objects we no longer need
    del X_embedded
    gc.collect()

    final_data_list = []
    
    # 2. Process chunks with explicit memory management
    for chunk in df_iterator_factory():
        # Transform using float32 to save 50% memory immediately
        X_chunk = scaler.transform(chunk[feature_cols]).astype(np.float32)
        
        # Mapping into spectral space is memory-intensive
        X_chunk_embedded = spectral_map.transform(X_chunk).astype(np.float32)
        
        # Predict clusters (int16 is plenty for cluster IDs)
        cluster_labels = final_kmeans.predict(X_chunk_embedded).astype(np.int16)
        
        # 3. Attach labels and drop the spectral components to free RAM
        chunk_res = chunk.copy()
        chunk_res['global_cluster_id'] = cluster_labels
        
        # Append the labeled chunk, then kill the temp matrices
        final_data_list.append(chunk_res)
        
        del X_chunk, X_chunk_embedded, cluster_labels
        # We don't gc.collect() inside the loop (it slows things down), 
        # but we do it if memory is tight.
        
    # 4. Final Concatenation - The most dangerous part for memory
    print("📦 Combining results...")
    full_df = pd.concat(final_data_list, ignore_index=False)
    
    # Immediately clear the list pointers
    del final_data_list
    gc.collect()

    # 5. Handle Metadata
    if hasattr(df_sample, 'attrs'):
        full_df.attrs.update(df_sample.attrs)
    del df_sample
    gc.collect()
    
    if show_plots:
        analyze_cluster_composition(full_df, target_col=target_col)
        
    return full_df