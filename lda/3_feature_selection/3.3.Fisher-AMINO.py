import amino_fast_mod as amino
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

    # Pass 1: Accumulate Statistics
    for chunk in df_iterator_factory():
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        y = chunk[target_col].astype(int).values
        current_chunk_classes = np.unique(y)

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

    return pd.Series(fisher_scores).sort_values(ascending=False)

def extract_candidates_only(df_iterator, target_col, candidates):
    """
    Pass 2: Extracts only high-signal features and essential metadata for AMINO.
    """
    print(f"Loading {len(candidates)} candidate features into memory...")
    data_list = []
    cols_to_keep = candidates + [target_col] + [c for c in ['time', 'frame_number'] if c in METADATA_COLS]
    
    for chunk in df_iterator:
        available = [c for c in cols_to_keep if c in chunk.columns]
        data_list.append(chunk[available])
        
    return pd.concat(data_list, ignore_index=True)

def run_fisher_amino_pipeline(df_iterator_factory, target_col='class', max_outputs=5):
    """
    Integrated Fisher-AMINO Pipeline.
    """
    # 1. Fisher Scores (Sequential)
    fisher_s = compute_sequential_fisher(df_iterator_factory, target_col)
    
    # 2. Knee Detection
    y_vals = fisher_s.values
    kn = KneeLocator(range(1, len(y_vals) + 1), y_vals, curve='convex', direction='decreasing', S=1.0)
    threshold = kn.knee_y if kn.knee_y is not None else 0.0
    candidate_features = fisher_s[fisher_s >= threshold].index.tolist()
    
    print(f"Fisher Knee: {threshold:.4f} | Candidates: {len(candidate_features)}")

    # 3. Memory Cap
    if len(candidate_features) > 250:
        candidate_features = fisher_s.head(250).index.tolist()

    # 4. Extract for AMINO
    df_amino_input = extract_candidates_only(df_iterator_factory(), target_col, candidate_features)
    gc.collect()

    # 5. AMINO Redundancy Reduction
    print("Running AMINO...")
    ops = [amino.OrderParameter(name, df_amino_input[name].tolist()) for name in candidate_features]
    
    try:
        final_ops = amino.find_ops(ops, max_outputs=max_outputs, bins=20, distortion_filename=None)
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