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
    plt.pause(plot_pause)  # Configurable pause duration
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

def variance_filter_pipeline(df_iterator, return_threshold=False, show_plot=False, knee_S=1.0, outlier_multiplier=3.0, fallback_percentile=90, min_clean_ratio=0.5, plot_pause=3.0):
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
    
    if not df_chunks:
        print("Warning: No data chunks found")
        return
    
    # Check if all chunks have the same columns
    first_cols = set(df_chunks[0].columns)
    for i, chunk in enumerate(df_chunks):
        if set(chunk.columns) != first_cols:
            print(f"Warning: Chunk {i} has different columns. Skipping variance filtering.")
            # Just yield the original chunks without filtering
            for chunk in df_chunks:
                yield chunk
            return
    
    variance_series = compute_streaming_variance(iter(df_chunks))
    
    if len(variance_series) == 0:
        print("Warning: No features found for variance calculation")
        # Just yield the original chunks without filtering
        for chunk in df_chunks:
            yield chunk
        return
    
    threshold = get_knee_point(variance_series, knee_S=knee_S, outlier_multiplier=outlier_multiplier, fallback_percentile=fallback_percentile, min_clean_ratio=min_clean_ratio)
    selected_features = variance_series[variance_series > threshold].index.tolist()
    
    # Identify and report problematic features
    outliers = identify_problematic_features(variance_series)
    
    # Ensure we keep at least some features (minimum 5% or 50 features, whichever is smaller)
    min_features = max(5, min(50, int(len(variance_series) * 0.05)))
    
    if len(selected_features) < min_features:
        print(f"\nWarning: Only {len(selected_features)} features above threshold.")
        print(f"Adjusting threshold to keep at least {min_features} features...")
        
        # Use percentile to ensure minimum features
        adjusted_threshold = np.percentile(variance_series, (1 - min_features/len(variance_series)) * 100)
        selected_features = variance_series[variance_series >= adjusted_threshold].index.tolist()
        threshold = adjusted_threshold
        print(f"New threshold: {threshold:.6f} | Retained: {len(selected_features)} features.")
    else:
        print(f"\nThreshold: {threshold:.6f} | Retained: {len(selected_features)} features.")
    
    # Plot variance distribution if requested
    if show_plot:
        plot_variance_distribution(variance_series, threshold)

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