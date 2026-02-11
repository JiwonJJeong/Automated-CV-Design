import numpy as np
import pandas as pd
import gc
from deeptime.decomposition import TICA, VAMP
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

def run_memory_efficient_tica(df_iterator_factory, target_col='class', 
                             lag_time=10, max_components=6, sample_rows=20000):
    """
    TICA Pipeline optimized for low-memory MD analysis.
    """
    
    # --- Pass 1: Fit Scaler & Learn TICA on a restricted sample ---
    print(f"Pass 1: Learning TICA mapping from {sample_rows} rows...")
    iterator = df_iterator_factory()
    first_chunk = next(iterator)
    feature_cols = [c for c in get_feature_cols(first_chunk) if c != target_col]
    
    # We only take what we need to learn the slow modes
    X_sample = first_chunk[feature_cols].iloc[:sample_rows].values.astype(np.float32)
    
    scaler = StandardScaler().fit(X_sample)
    X_scaled = scaler.transform(X_sample)
    
    # VAMP Validation (Limited to sample to save RAM)
    v_est = VAMP(lagtime=lag_time, dim=max_components)
    v_model = v_est.fit(X_scaled).fetch_model()
    # We use VAMP-2 scores to confirm dim (omitted loop for brevity, but same logic applies)
    
    tica_est = TICA(lagtime=lag_time, dim=max_components)
    tica_model = tica_est.fit(X_scaled).fetch_model()
    
    X_tica_sample = tica_model.transform(X_scaled)
    
    del X_sample, X_scaled
    gc.collect()

    # --- Pass 2: Cluster in TICA space with MiniBatch ---
    print("Pass 2: Finding metastable states via MiniBatchKMeans...")
    # Find K using the sample in TICA space
    inertias = []
    k_range = range(2, 12)
    for k in k_range:
        mbk = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42).fit(X_tica_sample)
        inertias.append(mbk.inertia_)
    
    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    optimal_k = kn.knee if kn.knee else 4
    
    final_km = MiniBatchKMeans(n_clusters=optimal_k, batch_size=1000, random_state=42)
    final_km.fit(X_tica_sample)
    
    del X_tica_sample
    gc.collect()

    # --- Pass 3: Streaming Projection & Labeling ---
    print("Pass 3: Projecting full dataset...")

    def tica_generator():
        for chunk in df_iterator_factory():
            # Apply learned mapping
            X_chunk = scaler.transform(chunk[feature_cols]).astype(np.float32)
            X_tica = tica_model.transform(X_chunk).astype(np.float32)
            
            # Create a light copy for results
            chunk_res = chunk.copy()
            chunk_res['global_cluster_id'] = final_km.predict(X_tica).astype(np.int16)
            
            # Store first 2 TICA components for visualization
            chunk_res['TICA_1'] = X_tica[:, 0]
            chunk_res['TICA_2'] = X_tica[:, 1]
            
            yield chunk_res
            del X_chunk, X_tica

    # Concatenate only at the very end
    full_df = pd.concat(tica_generator(), ignore_index=False)
    gc.collect()
    return full_df