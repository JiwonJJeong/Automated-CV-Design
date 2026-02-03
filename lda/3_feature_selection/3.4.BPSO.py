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

# --- PASS 1: STREAMING FISHER ---

def compute_streaming_fisher(df_factory, target_col):
    """Memory-efficient Fisher scores using a factory to stream data."""
    print("Pass 1: Filtering features via Streaming Fisher Score...")
    stats = {}
    feature_cols = None
    total_n = 0

    # Ensure we use the factory to get a fresh generator
    for chunk in df_factory():
        if feature_cols is None:
            # Filter out metadata
            feature_cols = [c for c in chunk.columns if c not in 
                            {target_col, 'construct', 'subconstruct', 'replica', 'frame_number'}]
        
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

    # Global mean and scoring
    global_sum = sum(s['sum'] for s in stats.values())
    global_mean = global_sum / total_n
    
    num = np.zeros(len(feature_cols))
    den = np.zeros(len(feature_cols))
    
    for label, s in stats.items():
        m_k = s['sum'] / s['n']
        # SS_within = sum(x^2) - (sum(x)^2 / n)
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
        
        # Fast LinearSVC with limited samples for iteration speed
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

def run_bpso_pipeline(df_factory, target_col='class', candidate_limit=150, bpso_iters=30, seed=None):
    """
    1. Pass 1: Streams factory to find top N Fisher candidates.
    2. Pass 2: Streams factory to load ONLY top N candidates into RAM.
    3. Run BPSO on the RAM-resident subset.
    """
    # 1. Narrow the search space (Filter)
    fisher_scores = compute_streaming_fisher(df_factory, target_col)
    candidates = fisher_scores.index[:candidate_limit].tolist()
    
    # 2. Pass 2: Selective RAM Load
    print(f"Pass 2: Loading top {len(candidates)} features into memory...")
    data_list = []
    cols_to_keep = candidates + [target_col]
    
    for chunk in df_factory():
        data_list.append(chunk[cols_to_keep])
    
    narrow_df = pd.concat(data_list, ignore_index=True)
    del data_list
    gc.collect()

    X = narrow_df[candidates].values
    y = narrow_df[target_col].values

    # 3. Optimization
    print(f"Running BPSO on {X.shape[1]} features...")
    problem = SVMFeatureSelection(X, y)
    task = Task(problem, max_iters=bpso_iters)
    
    # FIX: Pass the seed directly to the PSO algorithm
    algorithm = ParticleSwarmOptimization(population_size=15, seed=seed)
    
    best_x, _ = algorithm.run(task)
    final_mask = best_x > 0.5
    final_features = [candidates[i] for i, m in enumerate(final_mask) if m]
    
    print(f"Final selection: {len(final_features)} features.")
    
    return narrow_df[final_features + [target_col]]

if __name__ == "__main__":
    pass