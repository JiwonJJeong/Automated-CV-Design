import numpy as np
import pandas as pd
import gc
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from deeptime.decomposition import TICA, VAMP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend

# 1. Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

# =============================================================================
# VISUALIZATION & VALIDATION
# =============================================================================

def plot_vamp_scores(n_range, scores):
    """Plots VAMP scores to justify the number of TICA components."""
    plt.figure(figsize=(8, 4))
    plt.plot(n_range, scores, 'D-', color='#16a085')
    plt.title("VAMP-2 Scores (Kinetic Information Content)")
    plt.xlabel("Number of Components")
    plt.ylabel("VAMP-2 Score")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_tica_landscape(df, cluster_col='global_cluster_id'):
    """Visualizes the TICA 'Energy Landscape'."""
    if 'TICA_1' in df.columns and 'TICA_2' in df.columns:
        plt.figure(figsize=(10, 7))
        # Use hexbin or kde for large MD datasets to see density
        plt.hexbin(df['TICA_1'], df['TICA_2'], gridsize=50, cmap='Blues', bins='log', alpha=0.3)
        sns.scatterplot(data=df.sample(min(5000, len(df))), 
                        x='TICA_1', y='TICA_2', hue=cluster_col, 
                        palette='bright', s=15, edgecolor='black', linewidth=0.5)
        plt.title("Metastable States on the TICA Kinetic Landscape")
        plt.show()

def analyze_cluster_composition(df, target_col='class', cluster_col='global_cluster_id'):
    """Calculates and visualizes which classes make up each cluster."""
    
    # 1. Create a contingency table (Counts)
    if isinstance(target_col, str):
        if target_col not in df.columns:
            print(f"⚠️  Cannot analyze composition: {target_col} not found. Available: {df.columns.tolist()}")
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

# =============================================================================
# IMPROVED TICA + VAMP PIPELINE
# =============================================================================

def run_validated_tica_pipeline(df_iterator_factory, target_col='class', stride=5, 
                                 lag_time=10, max_components=10, max_k=15, S=1.0, 
                                 show_plots=True):
    """
    Industry-standard TICA pipeline with VAMP validation.
    """
    
    # --- Pass 1: Sampling & VAMP Validation ---
    print(f"Sampling for VAMP validation (stride={stride})...")
    sample_list = []
    for chunk in df_iterator_factory():
        sample_list.append(chunk.iloc[::stride])
    
    df_sample = pd.concat(sample_list, ignore_index=False)
    feature_cols = [c for c in get_feature_cols(df_sample) if c != target_col]
    
    scaler = StandardScaler()
    X_sample = scaler.fit_transform(df_sample[feature_cols])
    
    # Calculate VAMP-2 scores for different numbers of components
    print("Calculating VAMP-2 scores...")
    vamp_scores = []
    comp_range = range(2, max_components + 1)
    
    for n in comp_range:
        try:
            # 1. Initialize estimator
            v_est = VAMP(lagtime=lag_time, dim=n)
            
            # 2. Fit and Fetch (Define v_model here)
            v_model = v_est.fit(X_sample).fetch_model()
            
            # 3. Score (Using the positional r=2 we identified)
            s = v_model.score(2)
            vamp_scores.append(s)
            
        except Exception as e:
            print(f"⚠️ VAMP failed for dim={n}: {e}")
            vamp_scores.append(0)
    
    plot_vamp_scores(comp_range, vamp_scores)

    # --- Pass 2: TICA Projection ---
    # We choose the number of components using Knee Detection on VAMP-2 scores
    kn_vamp = KneeLocator(list(comp_range), vamp_scores, curve='concave', direction='increasing', S=S)
    optimal_n = kn_vamp.knee if kn_vamp.knee is not None else 4
    print(f"✅ Optimal TICA Dimensions: {optimal_n}")
    print(f"Fitting TICA with dim={optimal_n} and lag={lag_time}...")
    tica_estimator = TICA(lagtime=lag_time, dim=optimal_n)
    tica_model = tica_estimator.fit(X_sample).fetch_model()
    X_tica = tica_model.transform(X_sample)

    # --- Pass 3: Elbow for Clustering ---
    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X_tica)
        inertias.append(km.inertia_)
    
    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    optimal_k = kn.knee if kn.knee is not None else 5
    print(f"✅ K-Means Elbow: {optimal_k} clusters.")

    # --- Pass 4: Final Labeling ---
    final_km = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42).fit(X_tica)
    
    # CRITICAL: Drop references to sampling data immediately
    del X_sample, X_tica
    gc.collect()

    final_data_list = []
    for chunk in df_iterator_factory():
        # Cast to float32 to save 50% RAM
        X_chunk = scaler.transform(chunk[feature_cols]).astype(np.float32)
        X_chunk_tica = tica_model.transform(X_chunk).astype(np.float32)
        
        chunk_res = chunk.copy()
        # Use category type for cluster IDs to save space
        chunk_res['global_cluster_id'] = final_km.predict(X_chunk_tica).astype(np.int16)
        chunk_res['TICA_1'] = X_chunk_tica[:, 0]
        chunk_res['TICA_2'] = X_chunk_tica[:, 1]
        
        final_data_list.append(chunk_res)
    
    print("📦 Concatenating final results...")
    full_df = pd.concat(final_data_list, ignore_index=False)
    
    # Wipe the list immediately
    del final_data_list
    gc.collect()

    # Transfer metadata BEFORE plotting
    if hasattr(df_sample, 'attrs'):
        full_df.attrs.update(df_sample.attrs)
    del df_sample # Final big object gone
    gc.collect()
    
    if show_plots:
        plot_tica_landscape(full_df)
    
    # Preserve metadata attributes (selected_features, etc.)
    if hasattr(df_sample, 'attrs'):
        full_df.attrs.update(df_sample.attrs)
        
    # --- Show Landscape & Composition ---
    if show_plots:
        plot_tica_landscape(full_df)
        analyze_cluster_composition(full_df, target_col=target_col)
    
    return full_df

if __name__ == "__main__":
    # Example:
    # df_factory = create_dataframe_factory("md_data.csv")
    # df_results = run_validated_tica_pipeline(df_factory, lag_time=25)
    pass