import numpy as np
import pandas as pd
import gc
from kneed import KneeLocator
import sys
import os

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
from data_access import get_feature_cols, METADATA_COLS
from .visualization import visualize_amino_diagnostics

# =============================================================================
# LOCAL MIN-MAX SCALER (Internal Definition)
# =============================================================================

class LocalMinMaxWrapper:
    """
    Wraps a Standard Scaled (Z-score) generator and converts it to 0-1 
    on the fly for algorithms that require non-negative inputs (Chi-Sq/AMINO).
    """
    def __init__(self, factory_func):
        self.factory_func = factory_func

    def __call__(self):
        """Yields chunks where features are shifted to 0-1 range."""
        iterator = self.factory_func()
        
        for chunk in iterator:
            # Separate metadata/target so we don't scale them
            feature_cols = [c for c in get_feature_cols(chunk) 
                          if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(chunk[c])]
            
            # Create a copy to avoid modifying the original cache if it exists
            scaled_chunk = chunk.copy()
            
            if feature_cols:
                data = scaled_chunk[feature_cols].values
                # Vectorized Min-Max for the chunk
                # Note: For strict global min-max, you'd need a pre-pass. 
                # For feature selection relative ranking, chunk-local or "Shift" is usually sufficient.
                # Here we use a safe normalization that handles zero variance.
                d_min = data.min(axis=0)
                d_max = data.max(axis=0)
                denom = d_max - d_min
                denom[denom == 0] = 1.0  # Prevent divide by zero
                
                scaled_chunk[feature_cols] = (data - d_min) / denom

            yield scaled_chunk

# =============================================================================
# CORE PROCESSING
# =============================================================================

def estimate_bin_edges_and_classes(df_iterator, target_col, q_bins=5, sample_rows=20000):
    print(f"Sampling to estimate Chi-Squared bin edges and classes...")
    collected = 0
    sample_list = []
    unique_classes = set()
    
    for chunk in df_iterator:
        if target_col in chunk.columns:
            unique_classes.update(chunk[target_col].unique())
        
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        
        # Take a slice of what we need
        needed = sample_rows - collected
        if needed <= 0: break
            
        sample_list.append(chunk[feature_cols].iloc[:needed])
        collected += len(chunk)
        
        if collected >= sample_rows:
            break

    full_sample = pd.concat(sample_list)
    X = full_sample.values
    quantiles = np.linspace(0, 1, q_bins + 1)
    
    # Calculate quantile-based bin edges
    all_edges = np.quantile(X, quantiles, axis=0)
    bin_edges = {col: np.unique(all_edges[:, i]) for i, col in enumerate(full_sample.columns)}
    
    return bin_edges, sorted(list(unique_classes))

def compute_sequential_chi(df_iterator, target_col, bin_edges, known_classes, stride=1):
    """Vectorized calculation of Chi-Squared scores."""
    print(f"Building Chi-Squared scores sequentially (stride={stride})...")
    
    # Initialize counts dictionary
    # Structure: {col_name: matrix of shape (n_bins, n_classes)}
    global_counts = {}
    
    for chunk in df_iterator:
        if stride > 1:
            chunk = chunk.iloc[::stride]
        
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        
        # Pre-calculate class masks once per chunk
        class_masks = {cls: (chunk[target_col] == cls).values for cls in known_classes}

        for col in feature_cols:
            if col not in bin_edges: continue
                
            # Initialize matrix if seen for the first time
            if col not in global_counts: 
                n_bins = len(bin_edges[col]) - 1
                if n_bins < 1: continue # Skip constant columns
                global_counts[col] = np.zeros((n_bins, len(known_classes)))
            
            try:
                # Fast discretization
                discretized = pd.cut(chunk[col], bins=bin_edges[col], include_lowest=True, labels=False).values
            except Exception: 
                continue
            
            # Fill the count matrix
            for i, class_label in enumerate(known_classes):
                mask = class_masks[class_label] & (~np.isnan(discretized))
                if np.any(mask):
                    counts = np.bincount(discretized[mask].astype(int), minlength=global_counts[col].shape[0])
                    # Ensure dimensions match before adding (handles edge case of out-of-bounds bins)
                    lim = min(len(counts), global_counts[col].shape[0])
                    global_counts[col][:lim, i] += counts[:lim]
    
    # Calculate Chi-Square statistic from contingency tables
    chi_scores = {}
    for col, counts in global_counts.items():
        if counts.sum() == 0:
            chi_scores[col] = 0.0
            continue
            
        col_sums = counts.sum(axis=0)
        row_sums = counts.sum(axis=1)
        total = counts.sum()
        
        expected = np.outer(row_sums, col_sums) / total
        
        # Prevent divide by zero in Chi-sq calculation
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
    
    # Create iterator once
    iterator = df_iterator()
    
    # Proactive discovery of existing metadata
    first_chunk = next(iterator) # Peek at first chunk to check columns
    meta_to_keep = [c for c in METADATA_COLS if c in first_chunk.columns and c != target_col]
    
    # Process the first chunk
    if stride > 1:
        first_chunk = first_chunk.iloc[::stride]
        
    cols_to_keep = list(dict.fromkeys(candidates + [target_col] + meta_to_keep))
    available = [c for c in cols_to_keep if c in first_chunk.columns]
    data_list.append(first_chunk[available])
    
    # Process remaining chunks
    for chunk in iterator:
        if stride > 1:
            chunk = chunk.iloc[::stride]
            
        cols_to_keep = list(dict.fromkeys(candidates + [target_col] + meta_to_keep))
        available = [c for c in cols_to_keep if c in chunk.columns]
        data_list.append(chunk[available])
        
    return pd.concat(data_list, ignore_index=False)

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_feature_selection_pipeline(df_iterator_factory, target_col='class', stride=5, max_amino=40, 
                                   q_bins=5, sample_rows=20000, knee_sensitivity=5.0, 
                                   bins=None, distortion_filename=None):
    
    # 1. SETUP: Create the Local Min-Max Wrapper
    # This ensures Chi-Square/AMINO see 0-1 data, but we don't change the original factory
    
    # Ensure factory is callable (handle case where list/iterator is passed)
    if not callable(df_iterator_factory):
        print("Warning: df_iterator_factory is not callable. Caching data to memory.")
        cached_data = list(df_iterator_factory)
        def data_factory():
            return iter(cached_data)
        df_iterator_factory = data_factory

    minmax_factory = LocalMinMaxWrapper(df_iterator_factory)
    
    # 2. PASS 0 & 1: Statistics (Using Min-Max Data)
    bin_edges, discovered_classes = estimate_bin_edges_and_classes(minmax_factory(), target_col, 
                                                                 q_bins=q_bins, sample_rows=sample_rows)
    
    chi_s = compute_sequential_chi(minmax_factory(), target_col, bin_edges, discovered_classes, stride=stride)
    
    if chi_s.empty:
        print("Warning: Chi-Square returned no scores. Returning empty DataFrame.")
        return pd.DataFrame()

    # 3. KNEE DETECTION
    y_vals = sorted(chi_s.values, reverse=True)
    # Protection against tiny datasets
    if len(y_vals) > 0:
        kn = KneeLocator(range(1, len(y_vals) + 1), y_vals, curve='convex', direction='decreasing', S=knee_sensitivity)
        threshold = kn.knee_y if kn.knee_y is not None else 0.0
    else:
        threshold = 0.0
        
    candidate_features = chi_s[chi_s >= threshold].index.tolist()
    print(f"Knee Point: {threshold:.4f} | Candidates: {len(candidate_features)}")

    # 4. PASS 2: Load Candidate Data for AMINO (Using Min-Max Data)
    # AMINO specifically benefits from 0-1 scaling for its order parameters
    df_strided = extract_candidates_only(lambda: minmax_factory(), target_col, candidate_features, stride=stride)
    
    # Apply Min-Max manually here just for the memory-resident dataframe used by AMINO
    # (Since extract_candidates_only uses the raw factory to save overhead, we scale the result)
    feat_cols = [c for c in candidate_features if c in df_strided.columns]
    df_strided[feat_cols] = (df_strided[feat_cols] - df_strided[feat_cols].min()) / (df_strided[feat_cols].max() - df_strided[feat_cols].min() + 1e-12)

    gc.collect()

    # 5. AMINO REDUNDANCY REDUCTION
    final_names = []
    if amino is not None and len(candidate_features) > 0:
        print(f"Running AMINO on {len(candidate_features)} candidates...")
        # Create OrderParameter objects
        ops = [amino.OrderParameter(name, df_strided[name].values) for name in candidate_features if name in df_strided.columns]
        
        try:
            final_ops = amino.find_ops(ops, max_outputs=max_amino, bins=bins, distortion_filename=distortion_filename)
            final_names = [getattr(op, 'name', str(op)) for op in final_ops]
        except Exception as e:
            print(f"AMINO failed ({e}), falling back to Chi-Squared top features.")
            final_names = candidate_features[:max_amino]
    else:
        # Fallback if AMINO not installed or empty candidates
        final_names = candidate_features[:max_amino] if max_amino else candidate_features

    valid_final_names = [n for n in final_names if n in df_strided.columns]
    print(f"✅ Selected {len(valid_final_names)} final features.")

    # 6. VISUALIZATION
    try:
        visualize_amino_diagnostics(chi_s, df_strided, valid_final_names, target_col)
    except Exception as e:
        print(f"Visualization skipped: {e}")

    # 7. PASS 3: FINAL EXTRACTION (CRITICAL STEP)
    # We use df_iterator_factory (Standard Scaled) NOT minmax_factory
    print(f"Pass 3: Extracting all rows (Standard Scaled) for the {len(valid_final_names)} selected features...")
    
    df_full = extract_candidates_only(df_iterator_factory, target_col, valid_final_names, stride=1)
    
    return df_full

if __name__ == "__main__":
    pass