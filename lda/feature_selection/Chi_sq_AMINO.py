import numpy as np
import pandas as pd
import gc
from kneed import KneeLocator
import sys
import os
from ..feature_scaling.min_max import create_minmax_scaled_generator
from .visualization import visualize_amino_diagnostics

# 1. Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    try:
        import amino_fast_mod as amino
        print("Successfully imported Optimized AMINO (Parallel)")
    except ImportError:
        import amino
        print("Successfully imported AMINO from:", amino.__file__)
except ImportError as e:
    print(f"FAILED to find amino.py in {script_dir}. Error: {e}")
    amino = None

# Import refactored helpers
from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

# =============================================================================
# =============================================================================
# CORE PROCESSING
# =============================================================================

def estimate_bin_edges_and_classes(df_iterator, target_col, q_bins=5, sample_rows=20000):
    print(f"Sampling to estimate Chi-Squared bin edges and classes...")
    collected = 0
    sample_list = []
    unique_classes = set()
    
    for chunk in df_iterator:
        unique_classes.update(chunk[target_col].unique())
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
    
    return bin_edges, sorted(list(unique_classes))

def compute_sequential_chi(df_iterator, target_col, bin_edges, known_classes, stride=1):
    """Vectorized calculation of Chi-Squared scores to prevent performance lag."""
    print(f"Building Chi-Squared scores sequentially (stride={stride})...")
    global_counts = {}
    
    for chunk in df_iterator:
        if stride > 1:
            chunk = chunk.iloc[::stride]
        
        # Explicitly exclude all metadata columns and the target column
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        
        # Pre-calculate class masks once per chunk for vectorization
        class_masks = {cls: (chunk[target_col] == cls).values for cls in known_classes}

        for col in feature_cols:
            if col not in global_counts: 
                global_counts[col] = np.zeros((len(bin_edges[col]) - 1, len(known_classes)))
            
            try:
                # Use labels=False for speed boost; result is an array of bin indices
                discretized = pd.cut(chunk[col], bins=bin_edges[col], include_lowest=True, labels=False).values
            except Exception: continue
            
            for i, class_label in enumerate(known_classes):
                mask = class_masks[class_label] & (~np.isnan(discretized))
                if np.any(mask):
                    counts = np.bincount(discretized[mask].astype(int), minlength=global_counts[col].shape[0])
                    global_counts[col][:, i] += counts
    
    chi_scores = {}
    for col, counts in global_counts.items():
        if counts.sum() == 0:
            chi_scores[col] = 0.0
            continue
        col_sums = counts.sum(axis=0)
        row_sums = counts.sum(axis=1)
        total = counts.sum()
        expected = np.outer(row_sums, col_sums) / total
        mask = expected > 0
        if not mask.any():
            chi_scores[col] = 0.0
        else:
            chi_scores[col] = np.sum((counts[mask] - expected[mask])**2 / expected[mask])
            
    return pd.Series(chi_scores).sort_values(ascending=False)

def extract_candidates_only(df_iterator, target_col, candidates, stride=1):
    """Extracts specific columns. stride=1 is used for final full-data return."""
    print(f"Loading {len(candidates)} features (stride={stride})...")
    data_list = []
    
    # Metadata we want to carry through
    meta_to_keep = ['time', 'frame_number', 'replica', 'trajectory'] 
    
    for chunk in df_iterator:
        if stride > 1:
            chunk = chunk.iloc[::stride]
            
        existing_meta = [c for c in meta_to_keep if c in chunk.columns]
        cols_to_keep = list(dict.fromkeys(list(candidates) + [target_col] + existing_meta))
        
        available = [c for c in cols_to_keep if c in chunk.columns]
        data_list.append(chunk[available])
        
    return pd.concat(data_list, ignore_index=False)

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_feature_selection_pipeline(df_iterator_factory, target_col='class', stride=5, max_amino=10, 
                                   q_bins=5, sample_rows=20000, knee_S=5.0, 
                                   bins=None, distortion_filename=None, **kwargs):
    
    # Apply min-max scaling to the data
    scaled_df_factory = create_minmax_scaled_generator(df_iterator_factory)
    
    # 1. Pass 0 & 1: Statistics (Strided for Speed)
    bin_edges, discovered_classes = estimate_bin_edges_and_classes(scaled_df_factory(), target_col, 
                                                                  q_bins=q_bins, sample_rows=sample_rows)
    chi_s = compute_sequential_chi(scaled_df_factory(), target_col, bin_edges, discovered_classes, stride=stride)
    
    # 2. Knee Detection
    y_vals = sorted(chi_s.values, reverse=True)
    kn = KneeLocator(range(1, len(y_vals) + 1), y_vals, curve='convex', direction='decreasing', S=knee_S)
    threshold = kn.knee_y if kn.knee_y is not None else 0.0
    candidate_features = chi_s[chi_s >= threshold].index.tolist()
    print(f"Knee Point: {threshold:.4f} | Candidates: {len(candidate_features)}")

    # 3. Pass 2: Load Candidate Data for AMINO (Strided)
    df_strided = extract_candidates_only(scaled_df_factory, target_col, candidate_features, stride=stride)
    gc.collect()

    # 4. AMINO Redundancy Reduction
    if max_amino is None:
        actual_max = len(candidate_features) # Let AMINO decide within this bound
    else:
        actual_max = min(max_amino, len(candidate_features))
        
    final_names = []
    
    if amino is not None and len(candidate_features) > 0:
        print(f"Running AMINO on {len(candidate_features)} candidates...")
        ops = [amino.OrderParameter(name, df_strided[name].values) for name in candidate_features]
        try:
            final_ops = amino.find_ops(ops, max_outputs=max_amino, bins=bins, distortion_filename=distortion_filename)
            final_names = [getattr(op, 'name', str(op)) for op in final_ops]
        except Exception as e:
            print(f"AMINO failed ({e}), falling back to Chi-Squared top features.")
            limit = max_amino if max_amino is not None else 10
            final_names = candidate_features[:limit]
    else:
        limit = max_amino if max_amino is not None else 10
        final_names = candidate_features[:limit]

    valid_final_names = [n for n in final_names if n in df_strided.columns]
    print(f"✅ Selected {len(valid_final_names)} final features.")

    # 5. VISUALIZATION (Uses strided data)
    visualize_amino_diagnostics(chi_s, df_strided, valid_final_names, target_col)

    # 6. Pass 3: Final Extraction (FULL DATASET, stride=1)
    print(f"Pass 3: Extracting all rows for the {len(valid_final_names)} selected features...")
    df_full = extract_candidates_only(df_iterator_factory(), target_col, valid_final_names, stride=1)
    
    return df_full

if __name__ == "__main__":
    pass