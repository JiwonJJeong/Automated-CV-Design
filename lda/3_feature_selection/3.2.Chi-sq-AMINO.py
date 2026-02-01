import numpy as np
import pandas as pd
import h5py
import gc
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.feature_selection import chi2
import amino_fast_mod as amino

# --- CONFIGURATION ---
METADATA_COLS = {'construct', 'subconstruct', 'replica', 'frame_number'}

def h5_chunk_iterator(h5_path, dataset_name='data', chunk_size=10000):
    """Generates DataFrame chunks from an H5 file using memory-efficient slicing."""
    with h5py.File(h5_path, 'r') as f:
        dataset = f[dataset_name]
        total_rows = dataset.shape[0]
        # Attempt to get column names from attributes, else generic names
        column_names = f[dataset_name].attrs.get('column_names')
        if column_names is None:
            column_names = [f'feature_{i}' for i in range(dataset.shape[1])]
        
        for i in range(0, total_rows, chunk_size):
            end = min(i + chunk_size, total_rows)
            yield pd.DataFrame(dataset[i:end], columns=column_names)

# --- PASS 1: STREAMING STATISTICS ---

def compute_pass1_stats(df_iterator, target_col, q_bins=5):
    """
    Computes Variance and Chi-Squared contingency tables from a DataFrame iterator.
    
    Args:
        df_iterator: Iterator yielding DataFrames
        target_col: Name of the target column
        q_bins: Number of bins for chi-squared discretization
    
    Returns:
        tuple: (variance_series, chi_series)
    """
    print("Starting Pass 1: Computing Variance and Chi-Squared statistics...")
    
    # Variance (Chan's Algorithm) State
    n_a = 0
    mean_a = None
    m2_a = None
    
    # Chi-Squared (Contingency Tables) State
    global_chi_tables = {}
    feature_cols = None
    
    for chunk in df_iterator:
        # 1. Setup Columns
        if feature_cols is None:
            feature_cols = [c for c in chunk.columns if c not in METADATA_COLS and c != target_col]
        
        data_b = chunk[feature_cols].values
        y_b = chunk[target_col]
        n_b = data_b.shape[0]
        
        # --- Update Variance (Online) ---
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
        
        # --- Update Chi-Squared Counts ---
        for col in feature_cols:
            # Rank-based discretization per chunk
            binned = pd.qcut(chunk[col].rank(method='first'), q=q_bins, labels=False)
            ct = pd.crosstab(binned, y_b)
            if col not in global_chi_tables:
                global_chi_tables[col] = ct
            else:
                global_chi_tables[col] = global_chi_tables[col].add(ct, fill_value=0)

    # Finalize Variance
    variance_series = pd.Series(m2_a / n_a, index=feature_cols)
    
    # Finalize Chi-Squared Scores
    chi_scores = {}
    for col, ct in global_chi_tables.items():
        observed = ct.values
        row_sums, col_sums = observed.sum(axis=1, keepdims=True), observed.sum(axis=0, keepdims=True)
        expected = (row_sums @ col_sums) / observed.sum()
        expected[expected == 0] = 1e-9
        chi_scores[col] = np.sum((observed - expected)**2 / expected)
    
    chi_series = pd.Series(chi_scores).sort_values(ascending=False)
    
    return variance_series, chi_series

# --- UTILITIES ---

def get_threshold_features(series, label="Statistic"):
    """Finds the knee point and returns features above that threshold."""
    y = sorted(series.values, reverse=True)
    kn = KneeLocator(range(len(y)), y, curve='convex', direction='decreasing')
    threshold = y[kn.knee] if kn.knee else 0.0
    selected = series[series >= threshold].index.tolist()
    print(f"{label} Knee: {threshold:.4f} | Kept {len(selected)} features.")
    return selected

# --- MAIN WORKFLOW ---

def run_feature_selection_pipeline(df_iterator, target_col='class', max_amino=10):
    """
    Complete Pipeline:
    1. Pass 1 Variance/Chi2 scan from DataFrame iterator.
    2. Knee Detection to filter noise.
    3. Pass 2 Extract candidate features from collected data.
    4. AMINO redundancy reduction.
    
    Args:
        df_iterator: Iterator yielding DataFrames with features and target column
        target_col: Name of the target column (default: 'class')
        max_amino: Maximum number of features to output from AMINO
    
    Returns:
        pd.DataFrame: Final selected features + target column
    """
    
    # Convert iterator to list for two-pass processing
    print("Collecting data from iterator...")
    df_chunks = list(df_iterator)
    
    # 1. Pass 1: Gather global stats
    var_s, chi_s = compute_pass1_stats(iter(df_chunks), target_col)
    
    # 2. Thresholding
    high_var_features = get_threshold_features(var_s, "Variance")
    high_chi_features = get_threshold_features(chi_s, "Chi-Squared")
    
    # Intersect criteria (Must have variance AND predictive power)
    candidate_features = list(set(high_var_features) & set(high_chi_features))
    print(f"Candidate features for AMINO: {len(candidate_features)}")

    # 3. Pass 2: Extract candidate features from collected chunks
    print("Pass 2: Extracting candidate features for AMINO...")
    
    # Combine all chunks and select candidate features + target
    combined_df = pd.concat(df_chunks, ignore_index=True)
    
    # Select only candidate features and target
    cols_to_keep = candidate_features + [target_col]
    df_amino_input = combined_df[cols_to_keep]
    
    reduced_data = df_amino_input[candidate_features].values
    y_data = df_amino_input[target_col].values
    
    df_amino_input = pd.DataFrame(reduced_data, columns=candidate_features)
    gc.collect()

    # 4. AMINO Optimization
    print(f"Running AMINO (Max outputs: {max_amino})...")
    ops = [amino.OrderParameter(name, df_amino_input[name].tolist()) for name in candidate_features]
    final_ops = amino.find_ops(ops, max_outputs=max_amino, bins=30, distortion_filename=None)
    final_feature_names = [str(op) for op in final_ops]

    # 5. Final Result
    final_df = df_amino_input[final_feature_names].copy()
    final_df[target_col] = y_data
    
    print(f"Pipeline Complete. Final Shape: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    # Example usage with DataFrame iterator:
    # from data_access import data_iterator
    # df_iter = data_iterator(base_dir='/path/to/data', chunk_size=10000)
    # result_df = run_feature_selection_pipeline(df_iter, target_col='class')
    # result_df.to_csv('final_features.csv', index=False)
    pass