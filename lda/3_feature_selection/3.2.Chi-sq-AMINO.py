import numpy as np
import pandas as pd
import gc
from kneed import KneeLocator
import amino_fast_mod as amino

# Import your new refactored helper
from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

# =============================================================================
# CORE PROCESSING (SEQUENTIAL & OPTIMIZED)
# =============================================================================

def estimate_bin_edges(df_iterator, target_col, q_bins=5, sample_rows=20000):
    """
    Pass 0: Reads a sample to estimate global quantile-based bin edges.
    """
    print(f"Sampling to estimate Chi-Squared bin edges...")
    collected = 0
    sample_list = []
    
    for chunk in df_iterator:
        # Dynamically identify feature columns using the helper
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        needed = sample_rows - collected
        if needed <= 0: break
        
        sample_list.append(chunk[feature_cols].iloc[:needed])
        collected += len(sample_list[-1])

    if not sample_list: return {}

    full_sample = pd.concat(sample_list)
    X = full_sample.values
    quantiles = np.linspace(0, 1, q_bins + 1)
    
    print(f"Computing quantiles for {X.shape[1]} features...")
    all_edges = np.quantile(X, quantiles, axis=0)
    
    bin_edges = {col: np.unique(all_edges[:, i]) for i, col in enumerate(full_sample.columns)}
    return bin_edges

def compute_sequential_chi(df_iterator, target_col, bin_edges):
    """
    Pass 1: Sequentially builds contingency tables to calculate Chi-Squared scores.
    Matches sklearn.feature_selection.chi2 behavior.
    """
    print("Building Chi-Squared scores sequentially...")
    
    global_counts = {} # {feature: 2D_array[bins, classes]}
    classes = None
    num_classes = 0

    for chunk in df_iterator:
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        
        # Initialize categories on the first chunk
        if classes is None:
            y_categories = pd.Categorical(chunk[target_col])
            classes = y_categories.categories
            num_classes = len(classes)
            
        y_codes = pd.Categorical(chunk[target_col], categories=classes).codes
        
        for col in feature_cols:
            edges = bin_edges[col]
            # Digitize: map continuous distance to discrete bin index
            binned = np.digitize(chunk[col].values, edges[1:-1], right=True)
            num_bins = len(edges) - 1
            
            if col not in global_counts:
                global_counts[col] = np.zeros((num_bins, num_classes))
            
            # Vectorized contingency update
            for cls_idx in range(num_classes):
                mask = (y_codes == cls_idx)
                if np.any(mask):
                    bin_counts = np.bincount(binned[mask], minlength=num_bins)
                    global_counts[col][:, cls_idx] += bin_counts

    # Calculate Chi-Square from contingency tables
    chi_scores = {}
    for col, table in global_counts.items():
        row_sums = table.sum(axis=1, keepdims=True)
        col_sums = table.sum(axis=0, keepdims=True)
        total = table.sum()
        
        expected = (row_sums @ col_sums) / total
        mask = expected > 0
        score = np.sum((table[mask] - expected[mask])**2 / expected[mask])
        chi_scores[col] = score
        
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

def run_feature_selection_pipeline(df_iterator_factory, target_col='class', max_amino=10):
    """
    Integrated Pipeline accepting a callable factory to avoid generator exhaustion.
    """
    # Pass 0: Quantiles
    bin_edges = estimate_bin_edges(df_iterator_factory(), target_col)
    
    # Pass 1: Statistics
    chi_s = compute_sequential_chi(df_iterator_factory(), target_col, bin_edges)
    
    # Knee Detection
    y_vals = sorted(chi_s.values, reverse=True)
    kn = KneeLocator(range(1, len(y_vals) + 1), y_vals, curve='convex', direction='decreasing', S=5.0)
    threshold = kn.knee_y if kn.knee_y is not None else 0.0
    candidate_features = chi_s[chi_s >= threshold].index.tolist()
    
    print(f"Knee Point: {threshold:.4f} | Candidates: {len(candidate_features)}")

    # Stability Cap for AMINO
    if len(candidate_features) > 250:
        candidate_features = chi_s.head(250).index.tolist()
    
    # Pass 2: Load Candidates
    df_amino_input = extract_candidates_only(df_iterator_factory(), target_col, candidate_features)
    gc.collect()

    # Pass 3: Redundancy reduction (AMINO)
    print(f"Running AMINO...")
    ops = [amino.OrderParameter(name, df_amino_input[name].tolist()) for name in candidate_features]
    final_ops = amino.find_ops(ops, max_outputs=max_amino, bins=20, distortion_filename=None)
    
    if not final_ops:
        return df_amino_input[[target_col]].copy()

    final_names = [str(op) for op in final_ops]
    # Return the selected features + class + metadata for downstream analysis
    return df_amino_input[final_names + [target_col]]

if __name__ == "__main__":
    # Example usage:
    # factory = create_dataframe_factory(base_dir="/path/to/data", chunk_size=5000)
    # selected_df = run_feature_selection_pipeline(factory, target_col='construct')
    pass