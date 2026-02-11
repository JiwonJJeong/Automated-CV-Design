import sys
import os
import numpy as np
import pandas as pd
import gc
from kneed import KneeLocator
from .visualization import visualize_amino_diagnostics

# Path handling for amino
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
    print(f"FAILED to find amino.py. Error: {e}")
    amino = None

# Import refactored helpers
from data_access import get_feature_cols, METADATA_COLS

def compute_sequential_fisher(df_iterator_factory, target_col):
    """
    Calculates Fisher scores using a two-pass accumulation method.
    Fisher score formula: $$F = \frac{\sum n_i (\mu_i - \mu_{total})^2}{\sum n_i \sigma_i^2}$$
    """
    print("Calculating Fisher scores sequentially...")
    
    stats = {} # {feature: {class_label: [count, sum, sum_sq]}}
    global_sum = {}
    global_sum_sq = {}
    total_count = 0
    feature_cols = None
    
    for chunk in df_iterator_factory():
        if feature_cols is None:
            feature_cols = [c for c in get_feature_cols(chunk) if c != target_col and c not in METADATA_COLS]
        
        y = chunk[target_col].values
        current_chunk_classes = np.unique(y)
        
        for col in feature_cols:
            if col not in stats:
                stats[col] = {}
                global_sum[col] = 0.0
                global_sum_sq[col] = 0.0

            x = chunk[col].values
            global_sum[col] += np.sum(x)
            global_sum_sq[col] += np.sum(x**2)

            for cls in current_chunk_classes:
                if cls not in stats[col]:
                    stats[col][cls] = [0, 0.0, 0.0]
                
                mask = (y == cls)
                cls_data = x[mask]
                if len(cls_data) > 0:
                    stats[col][cls][0] += len(cls_data)
                    stats[col][cls][1] += np.sum(cls_data)
                    stats[col][cls][2] += np.sum(cls_data**2)
        
        total_count += len(chunk)

    fisher_scores = {}
    for col in stats:
        m_total = global_sum[col] / (total_count + 1e-12)
        numerator = 0.0
        denominator = 0.0
        
        for cls, (n_i, s_i, ss_i) in stats[col].items():
            if n_i < 2: continue
            
            m_i = s_i / n_i
            # Variance calculation: $$s^2 = \frac{\sum x^2 - \frac{(\sum x)^2}{n}}{n-1}$$
            var_i = (ss_i - (s_i**2 / n_i)) / (n_i - 1)
            
            numerator += n_i * (m_i - m_total)**2
            denominator += n_i * var_i
            
        fisher_scores[col] = numerator / (denominator + 1e-12)

    return pd.Series(fisher_scores).sort_values(ascending=False)

def extract_candidates_only(df_iterator_factory, target_col, candidates):
    """
    Properly yields filtered chunks for concatenation.
    """
    print(f"Loading {len(candidates)} candidate features into memory...")
    
    # Peek at metadata once
    first_chunk = next(df_iterator_factory())
    existing_meta = [c for c in METADATA_COLS if c in first_chunk.columns and c != target_col]
    
    cols_to_extract = list(dict.fromkeys(candidates + [target_col] + existing_meta))

    def chunk_generator():
        for chunk in df_iterator_factory():
            yield chunk[cols_to_extract]

    full_df = pd.concat(chunk_generator(), ignore_index=False)
    gc.collect()
    return full_df

def run_fisher_amino_pipeline(df_iterator_factory, target_col='class', max_outputs=5, knee_S=1.0):
    # Pass 1: Fisher Scores (Using input factory directly)
    fisher_s = compute_sequential_fisher(df_iterator_factory, target_col)
    
    # Knee Detection
    y_vals = fisher_s.values
    kn = KneeLocator(range(1, len(y_vals) + 1), y_vals, curve='convex', direction='decreasing', S=knee_S)
    threshold = kn.knee_y if kn.knee_y else np.percentile(y_vals, 90)
    candidate_features = fisher_s[fisher_s >= threshold].index.tolist()

    # Pass 2: Data Extraction
    df_amino_input = extract_candidates_only(df_iterator_factory, target_col, candidate_features)

    # AMINO Reduction
    final_names = []
    if amino and len(candidate_features) > 0:
        print(f"Running AMINO on {len(candidate_features)} candidates...")
        
        # Internal temporary shift to [0, 1] for AMINO's MI calculations
        # This does not affect the final returned dataframe values
        df_shifted = df_amino_input[candidate_features].copy()
        d_min = df_shifted.min()
        d_max = df_shifted.max()
        df_shifted = (df_shifted - d_min) / (d_max - d_min + 1e-12)

        ops = [amino.OrderParameter(name, df_shifted[name].values) for name in candidate_features]
        final_ops = amino.find_ops(ops, max_outputs=max_outputs)
        final_names = [getattr(op, 'name', str(op)) for op in final_ops]
        del df_shifted
    else:
        print("AMINO skipped. Keeping top candidates.")
        limit = max_outputs if max_outputs is not None else 5
        final_names = candidate_features[:limit]

    visualize_amino_diagnostics(fisher_s, df_amino_input, final_names, target_col)

    # Prepare final output columns (keeping original standard scaled values)
    meta_present = [c for c in df_amino_input.columns if c in METADATA_COLS]
    final_columns = list(dict.fromkeys(final_names + [target_col] + meta_present))
    
    return df_amino_input[final_columns]