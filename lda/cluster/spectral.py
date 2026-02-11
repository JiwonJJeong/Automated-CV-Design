import numpy as np
import pandas as pd
import gc
from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

def run_memory_efficient_spectral(df_iterator_factory, target_col='class', 
                                 n_components=100, max_k=15, sample_rows=10000):
    """
    Highly memory-optimized Spectral Clustering.
    Uses MiniBatchKMeans and aggressive garbage collection.
    """
    
    # --- 1. Pass 1: Fit Scaler & Nystroem on a small fixed sample ---
    print(f"Pass 1: Learning mapping from {sample_rows} rows...")
    iterator = df_iterator_factory()
    first_chunk = next(iterator)
    feature_cols = [c for c in get_feature_cols(first_chunk) if c != target_col]
    
    # Take a small sample without concatenating everything
    X_sample = first_chunk[feature_cols].iloc[:sample_rows].values
    
    scaler = StandardScaler().fit(X_sample)
    X_sample_scaled = scaler.transform(X_sample).astype(np.float32)
    
    spectral_map = Nystroem(kernel='rbf', n_components=n_components, random_state=42)
    X_embedded_sample = spectral_map.fit_transform(X_sample_scaled)
    
    del X_sample, X_sample_scaled
    gc.collect()

    # --- 2. Pass 2: Finding K (Elbow) using MiniBatchKMeans ---
    # We use MiniBatch here because it's significantly faster and lighter
    print("Pass 2: Evaluating K in spectral space...")
    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        mbk = MiniBatchKMeans(n_clusters=k, batch_size=1000, n_init=3, random_state=42)
        mbk.fit(X_embedded_sample)
        inertias.append(mbk.inertia_)
    
    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    optimal_k = kn.knee if kn.knee else 5
    print(f"✅ Optimal K: {optimal_k}")

    # --- 3. Pass 3: Fit Final MiniBatchKMeans ---
    # Fit on the embedded sample
    final_km = MiniBatchKMeans(n_clusters=optimal_k, batch_size=1000, random_state=42)
    final_km.fit(X_embedded_sample)
    
    del X_embedded_sample
    gc.collect()

    # --- 4. Pass 4: The Streaming Projection ---
    print("Pass 4: Projecting full dataset...")
    
    def labeled_generator():
        for chunk in df_iterator_factory():
            # Process in float32
            X_chunk = scaler.transform(chunk[feature_cols]).astype(np.float32)
            X_emb = spectral_map.transform(X_chunk)
            
            # Add label directly to chunk
            chunk['global_cluster_id'] = final_km.predict(X_emb).astype(np.int16)
            
            yield chunk
            
            # Explicit cleanup for the matrices
            del X_chunk, X_emb

    # If you MUST return a single dataframe:
    # We do the concat here, but only after Pass 1-3 memory is purged
    return pd.concat(labeled_generator(), ignore_index=False)