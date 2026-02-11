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
from .visualization import visualize_mpso_diagnostics

# Standardized helpers
from data_access import get_feature_cols, METADATA_COLS

# --- PASS 1: STREAMING FISHER ---

def compute_fisher_scores(df_iterator, target_col='class', stride=1):
    """Memory-efficient Fisher scores with stride support. Assumes data is scaled."""
    print(f"Pass 1: Computing Fisher scores (stride={stride})...")
    stats = {}
    feature_cols = None
    total_n = 0

    for chunk in df_iterator:
        if stride > 1:
            chunk = chunk.iloc[::stride]
            
        if feature_cols is None:
            feature_cols = [c for c in get_feature_cols(chunk) 
                           if c != target_col and c not in METADATA_COLS and pd.api.types.is_numeric_dtype(chunk[c])]
        
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
    def __init__(self, X_train, y_train, dims, alpha=0.9, threshold=0.5, n_estimators=5, cv=3, redundancy_weight=0.2):
        super().__init__(dimension=X_train.shape[1] * dims, lower=0, upper=1)
        self.X_train, self.y_train = X_train, y_train
        self.dims = dims
        self.alpha = alpha
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.cv = cv
        self.redundancy_weight = redundancy_weight
        self.n_feats = X_train.shape[1]

    def _evaluate(self, x):
        weights = x.reshape((self.n_feats, self.dims))
        sel_matrix = (weights > self.threshold).astype(float)
        
        if np.any(np.sum(sel_matrix, axis=0) == 0):
            return 1.0

        # Projection: Divide by sum to keep projected values within original feature scale
        projected = np.matmul(self.X_train, sel_matrix) 
        projected /= (np.sum(sel_matrix, axis=0) + 1e-12)

        # Accuracy
        clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False, tol=1e-3), n_estimators=self.n_estimators))
        acc = cross_val_score(clf, projected, self.y_train, cv=self.cv).mean()

        # Redundancy
        if self.dims > 1:
            noise = np.random.normal(0, 1e-10, projected.shape)
            corr_matrix = np.corrcoef(projected + noise, rowvar=False)
            r_sq_matrix = np.square(corr_matrix)
            redundancy = (np.sum(r_sq_matrix) - self.dims) / (self.dims * (self.dims - 1))
            
            # Feature Overlap Penalty
            feature_usage = np.sum(sel_matrix, axis=1) 
            overlap = np.sum(feature_usage[feature_usage > 1]) / (self.n_feats * self.dims)
        else:
            redundancy = 0
            overlap = 0
            
        sparsity = np.sum(sel_matrix) / (self.n_feats * self.dims)
        error_term = self.alpha * (1 - acc) + (1 - self.alpha) * sparsity
        total_redundancy = (0.7 * redundancy) + (0.3 * overlap)
        
        return (1.0 - self.redundancy_weight) * error_term + self.redundancy_weight * total_redundancy

# --- MAIN PIPELINE ---

def run_mpso_pipeline(df_iterator_factory, target_col='class', n_dimensions=None, candidate_limit=250, max_iter=10, 
                        accuracy_sparsity_weight=0.9, feature_threshold=0.5, dimension_independence_penalty=0.2, 
                        sampling_stride=1, knee_sensitivity=2.0, pop_scaling=1.0, min_pop=10, max_pop=100, 
                        max_candidates=None, optimization_iterations=10, random_seed=None):
    
    # 1. Pass 1: Filter (Using raw factory, assuming already scaled)
    fisher_scores = compute_fisher_scores(df_iterator_factory(), target_col, stride=sampling_stride)
    if fisher_scores.empty:
        return pd.DataFrame()
        
    if candidate_limit is None:
        from kneed import KneeLocator
        y_vals = fisher_scores.values
        kn = KneeLocator(range(len(y_vals)), y_vals, curve='convex', direction='decreasing', S=knee_sensitivity)
        cutoff_idx = kn.knee if kn.knee is not None else min(250, len(y_vals))
        candidates = fisher_scores.index[:cutoff_idx+1].tolist()
        print(f"Dynamic candidate selection: {len(candidates)} features selected.")
    else:
        candidates = fisher_scores.index[:candidate_limit].tolist()

    # 2. Pass 2: Selective RAM Load
    print(f"Pass 2: Loading search data (sampling_stride={sampling_stride})...")
    search_data = []
    meta_to_keep = [c for c in METADATA_COLS if c != target_col]
    
    for chunk in df_iterator_factory():
        if sampling_stride > 1:
            chunk = chunk.iloc[::sampling_stride]
        cols_to_extract = list(dict.fromkeys(candidates + [target_col] + meta_to_keep))
        available = [c for c in cols_to_extract if c in chunk.columns]
        search_data.append(chunk[available])
    
    temp_df = pd.concat(search_data, ignore_index=False)
    X_search = temp_df[candidates].values
    y_search = temp_df[target_col].values
    
    del search_data
    gc.collect()

    # 3. Particle Swarm Optimization
    dynamic_pop = max_candidates if max_candidates is not None else 20
        
    print(f"Beginning Swarm Optimization...")
    problem = MPSOProjectionProblem(X_search, y_search, dims=n_dimensions, alpha=accuracy_sparsity_weight, threshold=feature_threshold, 
                                     n_estimators=5, cv=3, redundancy_weight=dimension_independence_penalty)
    task = Task(problem, max_iters=optimization_iterations, seed=random_seed)
    algorithm = ParticleSwarmOptimization(population_size=dynamic_pop, seed=random_seed)
    
    best_x, _ = algorithm.run(task)
    print(f"✅ Optimization complete.")
    
    # Generate Projection Recipe
    best_x_reshaped = best_x.reshape((len(candidates), n_dimensions))
    final_sel = (best_x_reshaped > feature_threshold)
    best_x_reshaped = best_x.reshape((len(candidates), n_dimensions))
    final_sel = (best_x_reshaped > feature_threshold)
    projection_weights = np.sum(final_sel, axis=0) + 1e-12

    # Column Naming Logic
    dim_columns = []
    for dim_idx in range(n_dimensions):
        selected_in_dim = np.where(final_sel[:, dim_idx])[0]
        if len(selected_in_dim) > 0:
            dim_scores = best_x_reshaped[selected_in_dim, dim_idx]
            top_local_idx = np.argsort(dim_scores)[-2:]
            top_features = [candidates[selected_in_dim[i]] for i in top_local_idx]
            dim_name = "_".join(top_features)
        else:
            dim_name = f"dim{dim_idx}"
        dim_columns.append(f'MPSO_{dim_name}')

    # --- PASS 3: Apply to FULL Dataset ---
    print("Pass 3: Recovering all rows and applying projection...")
    full_results = []
    
    for chunk in df_iterator_factory():
        X_chunk = chunk[candidates].values
        projected_chunk = np.matmul(X_chunk, final_sel) / projection_weights
        
        res_chunk = pd.DataFrame(projected_chunk, columns=dim_columns, index=chunk.index)
        res_chunk[target_col] = chunk[target_col].values
        
        current_meta = [c for c in meta_to_keep if c in chunk.columns]
        for meta in current_meta:
            res_chunk[meta] = chunk[meta].values
        
        full_results.append(res_chunk)

    final_df = pd.concat(full_results, ignore_index=False)
    gc.collect()

    # Visualization
    visualize_mpso_diagnostics(final_df, fisher_scores, candidates, final_sel, target_col)

    contributing_mask = np.any(final_sel, axis=1)
    final_df.attrs['selected_features'] = [candidates[i] for i, m in enumerate(contributing_mask) if m]

    return final_df