import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
import h5py

METADATA_COLS = {'construct', 'subconstruct', 'replica', 'frame_number'}

def h5_chunk_iterator(h5_path, dataset_name='data', chunk_size=10000):
    """
    Generator that yields chunks of an HDF5 dataset as Pandas DataFrames.
    """
    with h5py.File(h5_path, 'r') as f:
        dataset = f[dataset_name]
        total_rows = dataset.shape[0]
        # Assuming the first dimension is rows and column names are stored in attributes 
        # or we use the column index if names aren't available.
        column_names = f[dataset_name].attrs.get('column_names', [f'feature_{i}' for i in range(dataset.shape[1])])
        
        for i in range(0, total_rows, chunk_size):
            end = min(i + chunk_size, total_rows)
            yield pd.DataFrame(dataset[i:end], columns=column_names)

def compute_streaming_variance(df_iterator):
    """
    Calculates variance across multiple DataFrames iteratively using Chan's online algorithm.
    """
    n_a = 0
    mean_a = None
    m2_a = None  

    for df in df_iterator:
        feature_cols = [c for c in df.columns if c not in METADATA_COLS]
        data_b = df[feature_cols].values
        n_b = data_b.shape[0]
        
        if n_b == 0: continue

        mean_b = np.mean(data_b, axis=0)
        m2_b = np.var(data_b, axis=0, ddof=0) * n_b

        if n_a == 0:
            n_a, mean_a, m2_a = n_b, mean_b, m2_b
        else:
            n_ab = n_a + n_b
            delta = mean_b - mean_a
            mean_a = mean_a + delta * (n_b / n_ab)
            m2_a = m2_a + m2_b + (delta ** 2) * (n_a * n_b / n_ab)
            n_a = n_ab

    if n_a == 0: return pd.Series(dtype=float)
    
    variance = m2_a / n_a
    return pd.Series(variance, index=feature_cols)

def get_knee_point(variance_series):
    """Finds the elbow/knee in the variance distribution."""
    y = sorted(variance_series.values, reverse=True)
    x = range(len(y))
    
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    if kn.knee is not None:
        threshold = y[kn.knee]
        return threshold
    return 0.0

def variance_filter_pipeline(df_iterator, return_threshold=False):
    """
    Two-pass memory-efficient variance filtering pipeline.
    Pass 1: Calculate variance from iterator
    Pass 2: Yield filtered data
    
    Args:
        df_iterator: Iterator yielding DataFrames (must be consumable twice or provide fresh iterator)
        return_threshold: If True, returns (filtered_iterator, threshold, selected_features)
    
    Yields:
        pd.DataFrame: Filtered chunks with high-variance features + metadata columns
    """
    # PASS 1: Calculate variance
    print("Pass 1: Analyzing feature variance...")
    
    # Convert to list to allow two passes
    # Note: This stores DataFrames in memory, but they're already chunked
    df_chunks = list(df_iterator)
    
    variance_series = compute_streaming_variance(iter(df_chunks))
    
    if len(variance_series) == 0:
        print("Warning: No features found for variance calculation")
        return
    
    threshold = get_knee_point(variance_series)
    selected_features = variance_series[variance_series > threshold].index.tolist()
    
    print(f"Threshold: {threshold:.6f} | Retained: {len(selected_features)} features.")

    # PASS 2: Yield filtered data
    for chunk in df_chunks:
        # Identify columns to keep (Features > threshold + Metadata)
        keep = [c for c in chunk.columns if c in selected_features or c in METADATA_COLS]
        yield chunk[keep]

def h5_variance_filter_pipeline(h5_path, dataset_name='data', chunk_size=10000):
    """
    Two-pass memory-efficient pipeline for H5 files.
    Pass 1: Calculate Variance
    Pass 2: Yield filtered data
    
    This is a convenience wrapper around variance_filter_pipeline for H5 files.
    """
    # Create iterator from H5 file
    df_iter = h5_chunk_iterator(h5_path, dataset_name, chunk_size)
    
    # Use the general variance filter pipeline
    yield from variance_filter_pipeline(df_iter)

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Using with H5 file directly
    print("Example 1: H5 file")
    processed_stream = h5_variance_filter_pipeline('your_data.h5')
    
    for filtered_chunk in processed_stream:
        print(f"Processed chunk with shape: {filtered_chunk.shape}")
        break  # Just show first chunk
    
    # Example 2: Using with DataFrame iterator (e.g., from data_access.py)
    print("\nExample 2: DataFrame iterator")
    # from data_access import data_iterator
    # df_iter = data_iterator(base_dir='/path/to/data', chunk_size=10000)
    # processed_stream = variance_filter_pipeline(df_iter)
    # 
    # for filtered_chunk in processed_stream:
    #     print(f"Processed chunk with shape: {filtered_chunk.shape}")