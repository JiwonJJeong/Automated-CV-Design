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

def compute_fisher_scores(df_iterator, target_col='class'):
    """Memory-efficient Fisher scores using a data stream."""
    print("Pass 1: Computing Fisher scores for feature filtering...")
    stats = {}
    feature_cols = None
    total_n = 0

    for chunk in df_iterator:
        if feature_cols is None:
            feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        
        y = chunk[target_col].values
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

    global_mean = sum(s['sum'] for s in stats.values()) / total_n
    num, den = np.zeros(len(feature_cols)), np.zeros(len(feature_cols))
    
    for s in stats.values():
        m_k = s['sum'] / s['n']
        ss_within = s['sum_sq'] - (s['sum']**2 / s['n'])
        num += s['n'] * (m_k - global_mean)**2
        den += ss_within
    
    scores = num / (den + 1e-12)
    return pd.Series(scores, index=feature_cols).sort_values(ascending=False)

# --- MPSO PROBLEM DEFINITION ---

class MPSOProjectionProblem(Problem):
    def __init__(self, X_train, y_train, dims, alpha=0.9, threshold=0.5, n_estimators=5, cv=3):
        super().__init__(dimension=X_train.shape[1] * dims, lower=0, upper=1)
        self.X_train, self.y_train = X_train, y_train
        self.dims = dims
        self.alpha = alpha
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.cv = cv
        self.n_feats = X_train.shape[1]

    def _evaluate(self, x):
        # Selection Matrix (Features x Dims)
        sel_matrix = (x > self.threshold).reshape((self.n_feats, self.dims))
        
        if np.any(np.sum(sel_matrix, axis=0) == 0):
            return 1.0

        # Matrix Projection
        projected = np.matmul(self.X_train, sel_matrix) / np.sum(sel_matrix, axis=0)

        clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False, tol=1e-3), n_estimators=self.n_estimators))
        acc = cross_val_score(clf, projected, self.y_train, cv=self.cv).mean()

        sparsity = np.sum(sel_matrix) / (self.n_feats * self.dims)
        return self.alpha * (1 - acc) + (1 - self.alpha) * sparsity

# --- MAIN PIPELINE ---

def run_mpso_pipeline(df_iterator_factory, target_col='class', dims=5, candidate_limit=250, mpso_iters=40, alpha=0.9, threshold=0.5, population_size=None, pop_scaling=0.02, min_pop=40, max_pop=100, n_estimators=5, cv=3, seed=42):
    """
    Integrated Pipeline accepting a callable factory.
    Synchronized with Chi-Sq and BPSO architecture.
    """
    # 1. Pass 1: Filter
    fisher_scores = compute_fisher_scores(df_iterator_factory(), target_col)
    if fisher_scores.empty:
        return pd.DataFrame()
        
    candidates = fisher_scores.index[:candidate_limit].tolist()
    
    # 2. Pass 2: Selective RAM Load
    print(f"Pass 2: Loading top {len(candidates)} features for MPSO...")
    data_list = []
    # Consistent metadata preservation
    meta_to_keep = [c for c in ['time', 'frame_number', 'replica'] if c in METADATA_COLS]
    cols_to_keep = candidates + [target_col] + meta_to_keep
    
    for chunk in df_iterator_factory():
        available = [c for c in cols_to_keep if c in chunk.columns]
        data_list.append(chunk[available])
    
    full_df = pd.concat(data_list, ignore_index=True)
    X = full_df[candidates].values
    y = full_df[target_col].values
    
    del data_list
    gc.collect()

    # 3. Particle Swarm Optimization
    # Scale population for larger search space (1250 variables vs 150 in BPSO)
    if population_size is None:
        dynamic_pop = int(np.clip(len(candidates) * dims * pop_scaling, min_pop, max_pop))
    else:
        dynamic_pop = population_size
    print(f"Running MPSO on {len(candidates)} features x {dims} dims = {len(candidates) * dims} variables (Pop: {dynamic_pop})")
    
    problem = MPSOProjectionProblem(X, y, dims=dims, alpha=alpha, threshold=threshold, n_estimators=n_estimators, cv=cv)
    task = Task(problem, max_iters=mpso_iters)
    algorithm = ParticleSwarmOptimization(population_size=dynamic_pop, seed=seed)
    
    print("Beginning Swarm Optimization...")
    best_x, _ = algorithm.run(task)
    
    # 4. Final Transform
    final_sel = (best_x > threshold).reshape((len(candidates), dims))
    # Normalize projection by number of features contributing to each dimension
    projected_vals = np.matmul(X, final_sel) / (np.sum(final_sel, axis=0) + 1e-12)
    
    # Construct final DataFrame
    res_df = pd.DataFrame(projected_vals, columns=[f'MPSO_Dim_{i+1}' for i in range(dims)])
    
    # Re-attach target and metadata
    res_df[target_col] = y
    for meta in meta_to_keep:
        if meta in full_df.columns:
            res_df[meta] = full_df[meta].values

    print(f"MPSO Complete. Reduced {X.shape[1]} features to {dims} dimensions.")
    return res_df

if __name__ == "__main__":
    pass