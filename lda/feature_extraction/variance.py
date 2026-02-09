import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
import h5py
import gc

METADATA_COLS = {'construct', 'subconstruct', 'replica', 'frame_number', 'time'}

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
        #print(f"Available columns: {list(df.columns)}")
        #print(f"Column dtypes: {dict(df.dtypes)}")
        
        feature_cols = [c for c in df.columns if c not in METADATA_COLS]
        #print(f"Feature columns after filtering: {feature_cols}")
        
        # Double-check for any non-numeric columns
        numeric_cols = []
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols.append(col)
            else:
                print(f"Non-numeric column found: {col} (dtype: {df[col].dtype})")
        
        # print(f"Final numeric columns: {numeric_cols}")
        data_b = df[numeric_cols].values
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

def get_knee_point(variance_series, knee_S=1.0, outlier_multiplier=3.0, fallback_percentile=90, min_clean_ratio=0.5):
    """Finds the elbow/knee in the variance distribution with outlier handling."""
    y = sorted(variance_series.values, reverse=True)
    x = range(len(y))
    
    # Debug information
    print(f"Variance statistics:")
    print(f"  Max variance: {max(y):.6f}")
    print(f"  Min variance: {min(y):.6f}")
    print(f"  Mean variance: {np.mean(y):.6f}")
    print(f"  Median variance: {np.median(y):.6f}")
    print(f"  Total features: {len(y)}")
    
    # Detect outliers using IQR method
    q75, q25 = np.percentile(y, [75, 25])
    iqr = q75 - q25
    outlier_threshold = q75 + outlier_multiplier * iqr  # Configurable outlier detection
    
    outliers = [v for v in y if v > outlier_threshold]
    if outliers:
        print(f"  ⚠️  Detected {len(outliers)} outlier variances > {outlier_threshold:.6f}")
        print(f"  Top 5 outlier values: {outliers[:5]}")
        
        # Remove outliers for knee detection
        clean_values = [v for v in y if v <= outlier_threshold]
        if len(clean_values) < len(y) * min_clean_ratio:  # Configurable clean ratio threshold
            print("  Too many outliers detected, using original data")
            clean_values = y
    else:
        clean_values = y
    
    print(f"  Using {len(clean_values)} values for knee detection")
    
    # Try knee detection on cleaned data
    if len(clean_values) > 1:
        clean_x = range(len(clean_values))
        try:
            kn = KneeLocator(clean_x, clean_values, curve='convex', direction='decreasing', S=knee_S)
            if kn.knee is not None:
                threshold = clean_values[kn.knee]
                print(f"  Knee detected at index {kn.knee} with threshold {threshold:.6f}")
                print(f"  Features above threshold: {sum(1 for v in y if v > threshold)}")
                return threshold
        except (ValueError, RuntimeWarning) as e:
            print(f"  Knee detection failed: {e}")
    
    # Fallback strategies
    print("  No reliable knee point detected, using fallback strategies...")
    
    # Strategy 1: Use configurable percentile (more aggressive)
    if len(clean_values) > 10:
        threshold = np.percentile(clean_values, fallback_percentile)
        print(f"  Using {fallback_percentile}th percentile: {threshold:.6f}")
    else:
        # Strategy 2: Use median-based threshold
        median_threshold = np.median(y) * 2
        threshold = median_threshold
        print(f"  Using 2x median: {threshold:.6f}")
    
    print(f"  Features above threshold: {sum(1 for v in y if v > threshold)}")
    return threshold

def plot_variance_distribution(variance_series, threshold=None):
    """
    Plot the variance distribution with knee point.
    
    Args:
        variance_series (pd.Series): Variance values for features
        threshold (float): Threshold value to highlight
    """
    plt.figure(figsize=(12, 8))
    
    # Sort variance values in descending order
    sorted_variance = sorted(variance_series.values, reverse=True)
    x = range(len(sorted_variance))
    
    # Create subplots for both linear and log scale
    plt.subplot(2, 1, 1)
    plt.plot(x, sorted_variance, 'b-', linewidth=2, label='Feature Variance')
    plt.xlabel('Feature Rank (sorted by variance)')
    plt.ylabel('Variance Value')
    plt.title('Feature Variance Distribution (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if threshold is not None:
        threshold_idx = next((i for i, v in enumerate(sorted_variance) if v <= threshold), len(sorted_variance)-1)
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
        plt.axvline(x=threshold_idx, color='r', linestyle=':', alpha=0.7)
        plt.scatter([threshold_idx], [threshold], color='red', s=100, zorder=5)
    
    # Log scale subplot
    plt.subplot(2, 1, 2)
    # Filter out zero values for log scale
    non_zero_variance = [v for v in sorted_variance if v > 0]
    non_zero_x = range(len(non_zero_variance))
    
    plt.semilogy(non_zero_x, non_zero_variance, 'g-', linewidth=2, label='Feature Variance (log scale)')
    plt.xlabel('Feature Rank (sorted by variance)')
    plt.ylabel('Variance Value (log scale)')
    plt.title('Feature Variance Distribution (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if threshold is not None and threshold > 0:
        threshold_idx_log = next((i for i, v in enumerate(non_zero_variance) if v <= threshold), len(non_zero_variance)-1)
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
        plt.axvline(x=threshold_idx_log, color='r', linestyle=':', alpha=0.7)
        plt.scatter([threshold_idx_log], [threshold], color='red', s=100, zorder=5)
    
    plt.tight_layout()
    
    plt.show(block=False)
    # Don't close here - let it stay visible during computation

def identify_problematic_features(variance_series, outlier_threshold=None):
    """Identify features with suspiciously high variance."""
    if outlier_threshold is None:
        q75, q25 = np.percentile(variance_series.values, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 3 * iqr
    
    outliers = variance_series[variance_series > outlier_threshold]
    
    if len(outliers) > 0:
        print(f"\n🔍 Problematic Features (variance > {outlier_threshold:.6f}):")
        for feature, variance in outliers.head(10).items():
            print(f"  {feature}: {variance:.6f}")
        
        if len(outliers) > 10:
            print(f"  ... and {len(outliers) - 10} more")
    
    return outliers

def variance_filter_pipeline(data_factory, return_threshold=False, show_plot=False, **kwargs):
    """
    TRULY memory-efficient variance filtering.
    Instead of a list, we call data_factory() twice to stream from disk.
    """
    # PASS 1: Calculate variance
    print("Pass 1: Analyzing feature variance (Streaming from disk)...")
    
    # We call the factory to get a FRESH iterator for Pass 1
    pass1_iter = data_factory() 
    variance_series = compute_streaming_variance(pass1_iter)
    
    # Immediately clear Pass 1 from memory
    del pass1_iter
    gc.collect()

    if len(variance_series) == 0:
        print("Warning: No features found.")
        return data_factory() # Return fresh stream of original data

    # --- NEW: SANITY CHECK STEP ---
    median_var = variance_series.median()
    # Any feature with variance > 100x median is likely an error/metadata leak
    sanity_threshold = median_var * 100 
    
    outliers = variance_series[variance_series > sanity_threshold]
    
    if not outliers.empty:
        print("\n" + "!"*60)
        print(f"⚠️  SANITY CHECK WARNING: {len(outliers)} problematic features detected!")
        print(f"Median variance: {median_var:.6f} | Sanity threshold: {sanity_threshold:.6f}")
        for name, val in outliers.items():
            print(f"   -> FEATURE: {name} | VARIANCE: {val:.2e} (REMOVED)")
        print("!"*60 + "\n")
        
        # Remove these from the series before knee detection and plotting
        variance_series = variance_series.drop(outliers.index)

    # ... (Knee detection logic remains the same) ...
    threshold = get_knee_point(variance_series, **kwargs)
    selected_features = variance_series[variance_series > threshold].index.tolist()

    if show_plot:
        plot_variance_distribution(variance_series, threshold)

    # PASS 2: Yield filtered data
    print(f"Pass 2: Filtering columns (Retaining {len(selected_features)} features)...")
    
    # We call the factory AGAIN to get a FRESH iterator for Pass 2
    pass2_iter = data_factory()
    for chunk in pass2_iter:
        keep = [c for c in chunk.columns if c in selected_features or c in METADATA_COLS]
        yield chunk[keep]
        
    # Final cleanup
    del variance_series
    gc.collect()

def h5_variance_filter_pipeline(h5_path, dataset_name='data', chunk_size=10000):
    # This 'factory' can be called multiple times to get a new iterator
    factory = lambda: h5_chunk_iterator(h5_path, dataset_name, chunk_size)
    
    yield from variance_filter_pipeline(factory)

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