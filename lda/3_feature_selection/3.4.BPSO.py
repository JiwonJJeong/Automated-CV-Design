import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

# Standardized helpers
from data_access import get_feature_cols, METADATA_COLS

# --- PASS 1: STREAMING FISHER ---

def compute_streaming_fisher(df_iterator, target_col):
    """Memory-efficient Fisher scores using a data stream."""
    print("Pass 1: Filtering features via Streaming Fisher Score...")
    stats = {}
    feature_cols = None
    total_n = 0

    for chunk in df_iterator:
        if feature_cols is None:
            # Use same helper logic as Chi-Sq script
            feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        
        y = chunk[target_col].astype(int).values
        for label in np.unique(y):
            mask = (y == label)
            data = chunk.loc[mask, feature_cols].values
            
            if label not in stats:
                stats[label] = {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
            
            stats[label]['n'] += data.shape[0]
            stats[label]['sum'] += np.sum(data, axis=0)
            stats[label]['sum_sq'] += np.sum(data**2, axis=0)
        
        total_n += len(chunk)

    if total_n == 0:
        return pd.Series(dtype=float)

    global_sum = sum(s['sum'] for s in stats.values())
    global_mean = global_sum / total_n
    
    num = np.zeros(len(feature_cols))
    den = np.zeros(len(feature_cols))
    
    for label, s in stats.items():
        m_k = s['sum'] / s['n']
        ss_within = s['sum_sq'] - (s['sum']**2 / s['n'])
        num += s['n'] * (m_k - global_mean)**2
        den += ss_within
    
    scores = num / (den + 1e-12)
    return pd.Series(scores, index=feature_cols).sort_values(ascending=False)

# --- BPSO PROBLEM CLASS ---

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.95):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train, self.y_train = X_train, y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        if selected.sum() == 0: 
            return 1.0
        
        clf = OneVsRestClassifier(BaggingClassifier(
            LinearSVC(dual=False, tol=1e-3), 
            n_jobs=-1, n_estimators=5, max_samples=0.8
        ))
        
        try:
            acc = cross_val_score(clf, self.X_train[:, selected], self.y_train, cv=3).mean()
        except:
            return 1.0
            
        return self.alpha * (1 - acc) + (1 - self.alpha) * (selected.sum() / self.dimension)

# --- MAIN PIPELINE ---

def run_bpso_pipeline(df_iterator_factory, target_col='class', candidate_limit=150, bpso_iters=30, seed=None):
    """
    Integrated Pipeline accepting a callable factory to avoid generator exhaustion.
    Optimized for memory efficiency and total metadata preservation.
    """
    # 1. Pass 1: Statistics (Fisher Filter)
    # We call the factory to get a fresh iterator for Pass 1
    fisher_scores = compute_streaming_fisher(df_iterator_factory(), target_col)
    
    if fisher_scores.empty:
        print("Warning: No data found during Fisher Pass.")
        return pd.DataFrame()

    candidates = fisher_scores.index[:candidate_limit].tolist()
    
    # 2. Pass 2: Selective RAM Load
    print(f"Pass 2: Loading top {len(candidates)} features into memory...")
    
    # DYNAMIC METADATA IDENTIFICATION
    # Peek at one chunk to see which metadata columns are actually present
    try:
        sample_chunk = next(df_iterator_factory())
        existing_metadata = [c for c in METADATA_COLS if c in sample_chunk.columns]
    except StopIteration:
        return pd.DataFrame()

    # Ensure target_col and all available metadata are preserved in the load list
    cols_to_keep = candidates + [target_col] + existing_metadata
    
    data_list = []
    # Call factory again for the second full pass
    for chunk in df_iterator_factory():
        available = [c for c in cols_to_keep if c in chunk.columns]
        data_list.append(chunk[available])
    
    narrow_df = pd.concat(data_list, ignore_index=True)
    del data_list
    gc.collect()

    X = narrow_df[candidates].values
    y = narrow_df[target_col].values

    # 3. Optimization
    print(f"Running BPSO on {X.shape[1]} features...")
    problem = SVMFeatureSelection(X, y)
    task = Task(problem, max_iters=bpso_iters)
    
    algorithm = ParticleSwarmOptimization(population_size=15, seed=seed)
    
    best_x, _ = algorithm.run(task)
    final_mask = best_x > 0.5
    final_features = [candidates[i] for i, m in enumerate(final_mask) if m]
    
    print(f"Final selection: {len(final_features)} features.")
    
    # 4. Final Assembly
    # Grab everything that was in narrow_df but NOT in the candidates list
    # This automatically includes target_col, replica, frame_number, etc.
    other_cols = [c for c in narrow_df.columns if c not in candidates]
    
    return narrow_df[final_features + other_cols]

if __name__ == "__main__":
    pass