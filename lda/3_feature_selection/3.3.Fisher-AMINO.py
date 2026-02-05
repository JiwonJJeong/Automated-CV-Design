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

# Import refactored helpers
from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

def compute_sequential_fisher(df_iterator_factory, target_col):
    """
    Pass 1: Computes Fisher Scores using a memory-efficient approach.
    Since Fisher Score involves means and variances per class, we accumulate 
    sums and squared sums chunk-by-chunk.
    """
    print("Calculating Fisher scores sequentially...")
    
    stats = {} # {feature: {class_id: [count, sum, sum_sq]}}
    global_sum = {}
    global_sum_sq = {}
    total_count = 0
    classes_found = set()

    # Pass 1: Accumulate Statistics
    for chunk in df_iterator_factory():
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        y = chunk[target_col].astype(int).values
        current_chunk_classes = np.unique(y)
        classes_found.update(current_chunk_classes)
        
        if total_count == 0:
            print(f"Found classes in data: {sorted(current_chunk_classes)}")

        for col in feature_cols:
            # 1. Ensure the feature exists in stats
            if col not in stats:
                stats[col] = {}
                global_sum[col] = 0.0
                global_sum_sq[col] = 0.0

            # 2. Ensure EVERY class in this chunk exists for this feature
            for cls in current_chunk_classes:
                cls_int = int(cls)
                if cls_int not in stats[col]:
                    stats[col][cls_int] = [0, 0.0, 0.0]

            # 3. Accumulate global stats
            x = chunk[col].values
            global_sum[col] += np.sum(x)
            global_sum_sq[col] += np.sum(x**2)

            # 4. Accumulate per-class stats
            for cls in current_chunk_classes:
                cls_int = int(cls)
                mask = (y == cls_int)
                cls_data = x[mask]
                if len(cls_data) > 0:
                    stats[col][cls_int][0] += len(cls_data)
                    stats[col][cls_int][1] += np.sum(cls_data)
                    stats[col][cls_int][2] += np.sum(cls_data**2)
        
        total_count += len(chunk)

    # Pass 2: Compute Scores
    # Fisher Score = sum(n_i * (mean_i - mean_total)^2) / sum(n_i * var_i)
    fisher_scores = {}
    for col in stats:
        m_total = global_sum[col] / total_count
        
        numerator = 0.0
        denominator = 0.0
        
        for cls, (n_i, s_i, ss_i) in stats[col].items():
            if n_i < 2: continue
            m_i = s_i / n_i
            # Sample variance formula
            var_i = (ss_i - (s_i**2 / n_i)) / (n_i - 1)
            
            numerator += n_i * (m_i - m_total)**2
            denominator += n_i * var_i
            
        fisher_scores[col] = numerator / (denominator + 1e-12)

    result = pd.Series(fisher_scores).sort_values(ascending=False)
    print(f"Fisher computation complete. Non-zero scores: {(result > 0).sum()}/{len(result)}")
    print(f"Max Fisher score: {result.max():.6f}")
    return result

def extract_candidates_only(df_iterator_factory, target_col, candidates):
    """
    Pass 2: Optimized extraction. 
    Uses a generator to keep memory overhead low before concatenation.
    """
    print(f"Loading {len(candidates)} candidate features into memory...")
    
    # Define columns to keep
    cols_to_keep = candidates + [target_col]
    for c in ['time', 'frame_number']:
        if c in METADATA_COLS:
            cols_to_keep.append(c)

    def chunk_generator():
        for chunk in df_iterator_factory():
            # Filter columns immediately to drop unneeded data from RAM
            available = [c for c in cols_to_keep if c in chunk.columns]
            yield chunk[available]

    # Concat the generator directly
    full_df = pd.concat(chunk_generator(), ignore_index=True)
    
    # Explicitly clear internal buffers
    gc.collect()
    return full_df

def run_fisher_amino_pipeline(df_iterator_factory, target_col='class', max_outputs=5, knee_S=1.0, fallback_percentile=90, bins=20, distortion_filename=None):
    """
    Integrated Fisher-AMINO Pipeline.
    """
    # 1. Fisher Scores (Sequential)
    fisher_s = compute_sequential_fisher(df_iterator_factory, target_col)
    
    # 2. Knee Detection
    y_vals = fisher_s.values
    try:
        kn = KneeLocator(range(1, len(y_vals) + 1), y_vals, curve='convex', direction='decreasing', S=knee_S)
        if kn.knee_y is not None:
            threshold = kn.knee_y
        else:
            print("Fisher knee detection: No knee point found")
            threshold = np.percentile(y_vals, fallback_percentile)  # Fallback to percentile
    except (ValueError, RuntimeWarning) as e:
        print(f"Fisher knee detection failed: {e}")
        threshold = np.percentile(y_vals, fallback_percentile)  # Fallback to percentile
    
    candidate_features = fisher_s[fisher_s >= threshold].index.tolist()
    
    print(f"Fisher Knee: {threshold:.4f} | Candidates: {len(candidate_features)}")

    # 3. Extract for AMINO
    df_amino_input = extract_candidates_only(df_iterator_factory(), target_col, candidate_features)
    gc.collect()

    # 5. AMINO Redundancy Reduction
    if amino is None:
        print("AMINO module not available. Returning top Fisher features without redundancy reduction.")
        final_names = candidate_features[:max_outputs]
        return df_amino_input[final_names + [target_col]]
    
    print("Running AMINO...")
    try:
        # Pass the numpy array buffer (.values) instead of converting to a list (.tolist())
        ops = [amino.OrderParameter(name, df_amino_input[name].values) for name in candidate_features]
        final_ops = amino.find_ops(ops, max_outputs=max_outputs, bins=bins, distortion_filename=distortion_filename)
        final_names = [str(op) for op in final_ops]
    except Exception as e:
        print(f"AMINO failed: {e}. Falling back to top Fisher features.")
        final_names = candidate_features[:max_outputs]

    return df_amino_input[final_names + [target_col]]

if __name__ == "__main__":
    # Example usage with data_access factory
    # from data_access import create_dataframe_factory
    # factory = create_dataframe_factory(base_dir="./data", chunk_size=10000)
    # result_df = run_fisher_amino_pipeline(factory, target_col='class')
    pass