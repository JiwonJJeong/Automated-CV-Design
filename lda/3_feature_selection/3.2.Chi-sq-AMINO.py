import numpy as np
import pandas as pd
import gc
from kneed import KneeLocator
import sys
import os

# Try to import amino_fast_mod with proper path handling
try:
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    import amino_fast_mod as amino
except ImportError as e:
    print(f"Warning: Could not import amino_fast_mod: {e}")
    print("Falling back to basic amino module")
    try:
        import amino as amino
    except ImportError:
        print("Error: No amino module available")
        amino = None

import numpy as np
import pandas as pd
import gc
from kneed import KneeLocator
import matplotlib.pyplot as plt

# Import refactored helpers
from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

def plot_chi_squared_scores(chi_scores, bin_edges=None):
    """
    Plot Chi-Squared scores for features.
    
    Args:
        chi_scores (pd.Series): Chi-Squared scores for features
        bin_edges (list): Bin edges used for discretization
    """
    print(f"Debug: chi_scores shape = {chi_scores.shape}")
    print(f"Debug: chi_scores non-zero count = {(chi_scores > 0).sum()}")
    print(f"Debug: chi_scores max = {chi_scores.max()}")
    print(f"Debug: chi_scores min = {chi_scores.min()}")
    
    plt.figure(figsize=(12, 8))
    
    # Sort scores in descending order
    sorted_scores = chi_scores.sort_values(ascending=False)
    x = range(len(sorted_scores))
    
    plt.subplot(2, 1, 1)
    plt.plot(x, sorted_scores.values, 'b-', linewidth=2)
    plt.xlabel('Feature Rank')
    plt.ylabel('Chi-Squared Score')
    plt.title('Chi-Squared Feature Scores')
    plt.grid(True, alpha=0.3)
    
    # Show top features
    top_n = min(10, len(sorted_scores))
    plt.subplot(2, 1, 2)
    plt.barh(range(top_n), sorted_scores.values[:top_n][::-1])
    plt.yticks(range(top_n), sorted_scores.index[:top_n][::-1])
    plt.xlabel('Chi-Squared Score')
    plt.title(f'Top {top_n} Features by Chi-Squared Score')
    plt.tight_layout()
    
    # Show plot but don't close immediately
    plt.show(block=False)
    plt.pause(2.0)  # Keep visible for 2 seconds
    # Don't close here - let it stay visible during AMINO computation results.

def plot_amino_results(selected_features, amino_ops=None):
    """
    Plot AMINO redundancy reduction results.
    
    Args:
        selected_features (list): Final selected features
        amino_ops (list): AMINO order parameters (if available)
    """
    if amino_ops is None or len(amino_ops) == 0:
        print("No AMINO results to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Extract distortion values if available
    distortions = []
    names = []
    for op in amino_ops:
        if hasattr(op, 'name'):
            names.append(str(op.name))
        else:
            names.append(f"Feature_{len(names)}")
        
        # Try to get distortion value
        if hasattr(op, 'distortion'):
            distortions.append(op.distortion)
        else:
            distortions.append(0.0)
    
    if distortions and max(distortions) > 0:
        plt.bar(range(len(names)), distortions)
        plt.xlabel('Selected Features')
        plt.ylabel('Distortion')
        plt.title('AMINO Redundancy Reduction - Feature Distortions')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
    else:
        # Simple feature count plot
        plt.bar(['Selected Features'], [len(selected_features)])
        plt.ylabel('Number of Features')
        plt.title(f'AMINO Selected {len(selected_features)} Features')
    
    plt.tight_layout()
    
    plt.show()
    plt.close()

# =============================================================================
# CORE PROCESSING (SEQUENTIAL & OPTIMIZED)
# =============================================================================

def estimate_bin_edges_and_classes(df_iterator, target_col, q_bins=5, sample_rows=20000):
    """
    Pass 0: Estimates bin edges AND discovers all target classes.
    """
    print(f"Sampling to estimate Chi-Squared bin edges and classes...")
    collected = 0
    sample_list = []
    unique_classes = set() # Track classes
    
    for chunk in df_iterator:
        # 1. Track Classes (Scan all chunks for safety, or just sample)
        unique_classes.update(chunk[target_col].unique())
        
        # 2. Collect sample for bins
        if collected < sample_rows:
            feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
            needed = sample_rows - collected
            sample_list.append(chunk[feature_cols].iloc[:needed])
            collected += len(sample_list[-1])

    full_sample = pd.concat(sample_list)
    X = full_sample.values
    quantiles = np.linspace(0, 1, q_bins + 1)
    
    all_edges = np.quantile(X, quantiles, axis=0)
    bin_edges = {col: np.unique(all_edges[:, i]) for i, col in enumerate(full_sample.columns)}
    
    # Return sorted list of classes ensuring consistent ordering
    return bin_edges, sorted(list(unique_classes))

def compute_sequential_chi(df_iterator, target_col, bin_edges, known_classes):
    """
    Pass 1: Sequentially builds contingency tables to calculate Chi-Squared scores.
    Correctly pre-initializes storage to avoid the 'skip all' bug.
    """
    print("Building Chi-Squared scores sequentially...")
    
    # 1. PRE-INITIALIZE global_counts
    # We use known_classes and bin_edges to set the shape before looping
    global_counts = {}
    num_classes = len(known_classes)
    
    # Use the first chunk's schema just to identify which columns we have
    # but initialize based on our global bin_edges
    for col, edges in bin_edges.items():
        num_bins = len(edges) - 1
        global_counts[col] = np.zeros((num_bins, num_classes))

    total_samples = 0

    for chunk in df_iterator:
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        
        for col in feature_cols:
            if col not in global_counts:
                continue
                
            # Discretize using the global edges for this specific column
            try:
                discretized = pd.cut(chunk[col], bins=bin_edges[col], 
                                     include_lowest=True, labels=False)
            except Exception as e:
                continue
            
            # 2. UPDATE CONTINGENCY TABLE
            # We iterate over known_classes directly
            for i, class_label in enumerate(known_classes):
                mask = (chunk[target_col] == class_label) & (~discretized.isna())
                if mask.any():
                    # Get the bin indices for rows belonging to this class
                    bin_indices = discretized[mask].values.astype(int)
                    
                    # Vectorized update: count occurrences of each bin index
                    counts = np.bincount(bin_indices, minlength=global_counts[col].shape[0])
                    global_counts[col][:, i] += counts
        
        total_samples += len(chunk)

    # 3. CALCULATE CHI-SQUARED SCORE
    # Formula: sum((O - E)^2 / E)
    chi_scores = {}
    for col, counts in global_counts.items():
        if counts.sum() == 0:
            chi_scores[col] = 0.0
            continue
            
        col_sums = counts.sum(axis=0) # Sum per class
        row_sums = counts.sum(axis=1) # Sum per bin
        total = counts.sum()
        
        # Expected frequencies matrix
        expected = np.outer(row_sums, col_sums) / total
        
        mask = expected > 0
        if not mask.any():
            chi_scores[col] = 0.0
        else:
            # The actual Chi-Square calculation
            chi_sq = np.sum((counts[mask] - expected[mask])**2 / expected[mask])
            chi_scores[col] = chi_sq
            
    return pd.Series(chi_scores).sort_values(ascending=False)

def extract_candidates_only(df_iterator, target_col, candidates):
    """
    Pass 2: Extracts only high-signal features and essential metadata for AMINO.
    """
    print(f"Loading {len(candidates)} candidate features into memory for AMINO...")
    data_list = []
    # Keep target_col and time/frame for tracking if needed
    cols_to_keep = candidates + [target_col] + [c for c in ['time', 'frame_number'] if c in METADATA_COLS]
    
    for chunk in df_iterator:
        # Only grab existing columns (safety)
        available = [c for c in cols_to_keep if c in chunk.columns]
        data_list.append(chunk[available])
        
    return pd.concat(data_list, ignore_index=True)

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_feature_selection_pipeline(df_iterator_factory, target_col='class', max_amino=10, show_plots=False, q_bins=5, sample_rows=20000, knee_S=5.0, bins=None, distortion_filename=None, use_numpy_arrays=True):
    """
    Integrated Pipeline accepting a callable factory to avoid generator exhaustion.
    
    Args:
        df_iterator_factory: Callable that returns fresh iterator
        target_col: Target column name
        max_amino: Maximum features for AMINO output
        show_plots: Whether to generate and display plots
    """
    # Pass 0: Quantiles AND Class Discovery
    bin_edges, discovered_classes = estimate_bin_edges_and_classes(df_iterator_factory(), target_col, q_bins=q_bins, sample_rows=sample_rows)
    
    # Pass 1: Statistics
    chi_s = compute_sequential_chi(df_iterator_factory(), target_col, bin_edges, discovered_classes)
    
    # Knee Detection
    y_vals = sorted(chi_s.values, reverse=True)
    kn = KneeLocator(range(1, len(y_vals) + 1), y_vals, curve='convex', direction='decreasing', S=knee_S)
    threshold = kn.knee_y if kn.knee_y is not None else 0.0
    candidate_features = chi_s[chi_s >= threshold].index.tolist()
    
    print(f"Knee Point: {threshold:.4f} | Candidates: {len(candidate_features)}")
    
    # Plot Chi-Squared scores if requested
    if show_plots:
        plot_chi_squared_scores(chi_s, bin_edges)

    # Stability Cap for AMINO
    # Removed automatic truncation to allow all qualified candidates
    
    # Pass 2: Load Candidates
    df_amino_input = extract_candidates_only(df_iterator_factory(), target_col, candidate_features)
    gc.collect()

    # Pass 3: Redundancy reduction (AMINO)
    if amino is None:
        print("AMINO module not available. Returning top features without redundancy reduction.")
        # Return top features without AMINO processing
        final_names = candidate_features[:max_amino] if len(candidate_features) > max_amino else candidate_features
        return df_amino_input[final_names + [target_col]]
    
    print(f"Running AMINO on {len(candidate_features)} candidates...")
    
    # Add progress tracking
    import time
    start_time = time.time()
    
    # Choose data format based on parameter
    if use_numpy_arrays:
        ops = [amino.OrderParameter(name, df_amino_input[name].values) for name in candidate_features]
    else:
        ops = [amino.OrderParameter(name, df_amino_input[name].tolist()) for name in candidate_features]
    print(f"Created {len(ops)} order parameters in {time.time() - start_time:.2f}s")
    
    # Use provided bins or adaptive calculation
    if bins is None:
        bins = min(10, max(5, len(candidate_features) // 5))  # Adaptive bins
    print(f"Using {bins} bins for AMINO clustering")
    
    final_ops = amino.find_ops(ops, max_outputs=max_amino, bins=bins, distortion_filename=distortion_filename)
    
    elapsed = time.time() - start_time
    print(f"AMINO completed in {elapsed:.2f}s")
    
    if not final_ops:
        raise ValueError("AMINO returned no operations - this indicates a problem with the input data or parameters")
    else:
        final_names = [str(op) for op in final_ops]
    
    # Plot AMINO results if requested
    if show_plots:
        plot_amino_results(final_names, final_ops)
    
    # Return the selected features + class + metadata for downstream analysis
    return df_amino_input[final_names + [target_col]]

if __name__ == "__main__":
    # Example usage:
    # factory = create_dataframe_factory(base_dir="/path/to/data", chunk_size=5000)
    # selected_df = run_feature_selection_pipeline(factory, target_col='construct')
    pass