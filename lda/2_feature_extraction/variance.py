### STEP 0. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
import sys
import os

# Add parent directory to path to import data_access
# This ensures we can import data_access when running this script directly or from '2_feature_extraction'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from data_access import data_iterator
except ImportError:
    print("Warning: Could not import data_iterator from data_access. Make sure the 'lda' directory is in your Python path.")

def compute_streaming_variance(df_iterator):
    """
    Calculates variance across multiple DataFrames iteratively.
    
    Args:
        df_iterator: An iterator yielding pandas DataFrames.
        
    Returns:
        pd.Series: Column-wise variances, excluding metadata.
    """
    # State variables for Chan's algorithm
    n_a = 0
    mean_a = None
    m2_a = None  # Sum of squares of differences from the mean (SSE)

    metadata_cols = {'construct', 'subconstruct', 'replica', 'frame_number'}

    first_chunk = True

    if df_iterator is None:
        return pd.Series(dtype=float)

    for df in df_iterator:
        # Filter feature columns
        if first_chunk:
            feature_cols = [c for c in df.columns if c not in metadata_cols]
            # Initialize state with zeros for appropriate shape
            num_features = len(feature_cols)
            mean_a = np.zeros(num_features)
            m2_a = np.zeros(num_features)
            first_chunk = False
        
        # Get chunk data
        data_b = df[feature_cols].values
        n_b = data_b.shape[0]
        
        if n_b == 0:
            continue

        # Calculate statistics for the current chunk (B)
        mean_b = np.mean(data_b, axis=0)
        # M2_b = sum((x - mean_b)^2)
        m2_b = np.var(data_b, axis=0, ddof=0) * n_b

        # Combine with existing statistics (A) using Chan's algorithm
        # Refer: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        n_ab = n_a + n_b
        
        if n_a == 0:
            # First non-empty chunk becomes the base
            n_a = n_b
            mean_a = mean_b
            m2_a = m2_b
        else:
            delta = mean_b - mean_a
            
            # Update mean
            mean_a = mean_a + delta * (n_b / n_ab)
            
            # Update M2 (Sum of Squares)
            m2_a = m2_a + m2_b + (delta ** 2) * (n_a * n_b / n_ab)
            
            # Update count
            n_a = n_ab

    if n_a == 0:
        return pd.Series(dtype=float)

    # Final variance = M2 / N (Population Variance) or M2 / (N-1) (Sample Variance)
    # Using Population Variance (ddof=0) to match standard numpy output usually expected for large data
    variance = m2_a / n_a
    
    return pd.Series(variance, index=feature_cols)

def get_knee_point(values):
    """
    Finds the knee point in the sorted variance values using the Kneedle algorithm.
    Returns the variance threshold corresponding to the knee.
    """
    if values.empty:
        return 0.0
        
    y = sorted(values.values, reverse=True)
    x = range(len(y))
    
    # Use KneeLocator to find the "elbow"
    # curve='convex' and direction='decreasing' are typical for scree plots / variance distributions
    knee_locator = KneeLocator(x, y, curve='convex', direction='decreasing')
    knee_idx = knee_locator.knee
    
    if knee_idx is not None:
        threshold = y[knee_idx]
        print(f"Knee found at index {knee_idx}, variance threshold: {threshold:.4f}")
        return threshold
    else:
        print("No knee point found. Returning 0.0.")
        return 0.0

def plot_variance(variance_series, threshold):
    """
    Plots the variance of features and marks the threshold.
    """
    variance_values = variance_series.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(variance_values)), variance_values.values, label='Variance')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    
    plt.xlabel('Number of Features (sorted)', fontsize=14)
    plt.ylabel('Variance (Å²)', fontsize=14)
    plt.title('Variance vs. Number of Features', fontweight='bold', fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # plt.show() # Blocking in some environments, save or show depending on context
    # saving might be safer if running headless
    plt.savefig('variance_plot.png')
    print("Variance plot saved to variance_plot.png")

def remove_low_variance_features(threshold=None):
    """
    Computes variance from data_iterator, finds threshold (if None), plots,
    and returns the list of feature names with variance > threshold.
    """
    # 1. Get Data Iterator
    try:
        iterator = data_iterator()
    except NameError:
        print("Error: data_iterator is not defined. Check imports.")
        return []

    # 2. Compute Variance
    print("Computing streaming variance...")
    variance_series = compute_streaming_variance(iterator)
    
    if variance_series.empty:
        print("Variance calculation failed or returned empty.")
        return []

    # 3. Determine Threshold
    if threshold is None:
        threshold = get_knee_point(variance_series)
    
    # 4. Plot
    plot_variance(variance_series, threshold)
    
    # 5. Filter Features
    high_var_features = variance_series[variance_series > threshold].index.tolist()
    print(f"Retaining {len(high_var_features)} features with variance > {threshold}")
    
    return high_var_features

if __name__ == "__main__":
    # Example usage
    features = remove_low_variance_features()
    if features:
        print(f"Selected {len(features)} features. First 5: {features[:5]}")
