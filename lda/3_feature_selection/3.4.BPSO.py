import numpy as np
import pandas as pd
import h5py
import gc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

# --- H5 UTILITIES (Same as previous) ---

def h5_chunk_iterator(h5_path, dataset_name='data', chunk_size=10000):
    with h5py.File(h5_path, 'r') as f:
        dataset = f[dataset_name]
        total_rows = dataset.shape[0]
        column_names = f[dataset_name].attrs.get('column_names')
        if column_names is None:
            column_names = [f'feature_{i}' for i in range(dataset.shape[1])]
        for i in range(0, total_rows, chunk_size):
            end = min(i + chunk_size, total_rows)
            yield pd.DataFrame(dataset[i:end], columns=column_names)

# --- PASS 1: STREAMING FISHER (To narrow the search space for BPSO) ---

def compute_streaming_fisher(df_iterator, target_col):
    """Quickly reduces thousands of features to a manageable few hundred."""
    print("Pass 1: Filtering features via Streaming Fisher Score...")
    stats = {}
    feature_cols = None

    for chunk in df_iterator:
        if feature_cols is None:
            feature_cols = [c for c in chunk.columns if c not in {target_col, 'construct', 'subconstruct', 'replica', 'frame_number'}]
        
        for label, group in chunk.groupby(target_col):
            data = group[feature_cols].values
            if label not in stats:
                stats[label] = {'n': 0, 'sum': 0, 'sum_sq': 0}
            stats[label]['n'] += data.shape[0]
            stats[label]['sum'] += np.sum(data, axis=0)
            stats[label]['sum_sq'] += np.sum(data**2, axis=0)

    # Calculate final scores
    total_n = sum(s['n'] for s in stats.values())
    global_mean = sum(s['sum'] for s in stats.values()) / total_n
    num, den = np.zeros(len(feature_cols)), np.zeros(len(feature_cols))
    
    for label, s in stats.items():
        m_k = s['sum'] / s['n']
        v_k = (s['sum_sq'] / s['n']) - (m_k**2)
        num += s['n'] * (m_k - global_mean)**2
        den += s['n'] * v_k
    
    scores = num / (den + 1e-9)
    return pd.Series(scores, index=feature_cols).sort_values(ascending=False)

# --- BPSO PROBLEM CLASS ---

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99, n_estimators=10):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train, self.y_train = X_train, y_train
        self.alpha = alpha
        self.n_estimators = n_estimators

    def _evaluate(self, x):
        selected = x > 0.5
        if selected.sum() == 0: return 1.0
        
        clf = OneVsRestClassifier(BaggingClassifier(
            LinearSVC(dual=False), n_jobs=-1, n_estimators=self.n_estimators, max_samples=1.0/self.n_estimators
        ))
        
        # Accuracy via 3-fold CV on the narrow RAM-resident slice
        try:
            acc = cross_val_score(clf, self.X_train[:, selected], self.y_train, cv=3).mean()
        except:
            return 1.0
            
        return self.alpha * (1 - acc) + (1 - self.alpha) * (selected.sum() / self.dimension)

# --- MAIN PIPELINE ---

def run_bpso_pipeline(df_iterator, target_col='class', candidate_limit=200, bpso_iters=50):
    """
    1. Streams DataFrames to find top N candidate features (Fisher).
    2. Loads only those N features into RAM.
    3. Runs BPSO to find the best subset.
    
    Args:
        df_iterator: Iterator yielding DataFrames with features and target column
        target_col: Name of the target column (default: 'class')
        candidate_limit: Number of top features to consider for BPSO
        bpso_iters: Number of BPSO iterations
    
    Returns:
        pd.DataFrame: Final selected features + target column
    """
    # Convert iterator to list for two-pass processing
    print("Collecting data from iterator...")
    df_chunks = list(df_iterator)
    
    # 1. Narrow the search space (Filter)
    fisher_scores = compute_streaming_fisher(iter(df_chunks), target_col)
    candidates = fisher_scores.index[:candidate_limit].tolist()
    
    # 2. Load Narrow Slice into RAM (Pass 2)
    print(f"Loading top {candidate_limit} candidates into RAM for BPSO...")
    
    # Combine all chunks and select candidate features + target
    combined_df = pd.concat(df_chunks, ignore_index=True)
    
    # Select only candidate features and target
    cols_to_keep = candidates + [target_col]
    narrow_df = combined_df[cols_to_keep]
    
    X_narrow = narrow_df[candidates].values
    y = narrow_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X_narrow, y, test_size=0.3, stratify=y)

    # 3. Run BPSO
    print(f"Running BPSO on {candidate_limit} candidate features...")
    problem = SVMFeatureSelection(X_train, y_train)
    task = Task(problem, max_iters=bpso_iters)
    algorithm = ParticleSwarmOptimization(population_size=20)
    
    best_x, _ = algorithm.run(task)
    final_mask = best_x > 0.5
    final_features = [candidates[i] for i, m in enumerate(final_mask) if m]
    
    print(f"BPSO selected {len(final_features)} elite features.")

    # 4. Return the Mutated Result
    final_df = pd.DataFrame(X_narrow[:, final_mask], columns=final_features)
    final_df[target_col] = y
    
    return final_df

if __name__ == "__main__":
    # Example usage with DataFrame iterator:
    # from data_access import data_iterator
    # df_iter = data_iterator(base_dir='/path/to/data', chunk_size=10000)
    # df_final = run_bpso_pipeline(df_iter, target_col='class')
    pass