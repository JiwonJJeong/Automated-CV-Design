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

def compute_fisher_scores(df_iterator, target_col='class', stride=1):
    """Memory-efficient Fisher scores with stride support."""
    print(f"Pass 1: Computing Fisher scores (stride={stride})...")
    stats = {}
    feature_cols = None
    total_n = 0

    for chunk in df_iterator:
        # APPLY STRIDE TO PASS 1
        if stride > 1:
            chunk = chunk.iloc[::stride]
            
        if feature_cols is None:
            # Explicitly exclude all metadata columns including 'time'
            feature_cols = [c for c in get_feature_cols(chunk) 
                           if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(chunk[c])]
        
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
        self.eval_count = 0  # Track progress

    def _evaluate(self, x):
        # Selection Matrix (Features x Dims)
        self.eval_count += 1
        
        # Print progress every 50 evaluations
        if self.eval_count % 50 == 0:
            print(f"  > Evaluations completed: {self.eval_count}...", end='\r')
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
def run_mpso_pipeline(df_iterator_factory, target_col='class', dims=5, candidate_limit=250, mpso_iters=10, alpha=0.9, threshold=0.5, population_size=None, pop_scaling=0.02, min_pop=40, max_pop=100, n_estimators=5, cv=3, seed=42, stride=1):
    """
    Integrated Pipeline: Optimizes on strided data, returns FULL projected dataset.
    """
    # 1. Pass 1: Filter (Strided)
    fisher_scores = compute_fisher_scores(df_iterator_factory(), target_col, stride=stride)
    if fisher_scores.empty:
        return pd.DataFrame()
        
    candidates = fisher_scores.index[:candidate_limit].tolist()
    
    # 2. Pass 2: Selective RAM Load (Strided for MPSO)
    print(f"Pass 2: Loading search data (stride={stride})...")
    search_data = []
    # FIX: Dynamically identify all available metadata from your config
    meta_to_keep = [c for c in METADATA_COLS if c != target_col]
    
    for chunk in df_iterator_factory():
        if stride > 1:
            chunk = chunk.iloc[::stride]
        available = [c for c in candidates + [target_col] + meta_to_keep if c in chunk.columns]
        search_data.append(chunk[available])
    
    temp_df = pd.concat(search_data, ignore_index=True)
    X_search = temp_df[candidates].values
    y_search = temp_df[target_col].values
    
    del search_data
    gc.collect()

    # 3. Particle Swarm Optimization (Still runs on strided data for speed)
    if population_size is None:
        dynamic_pop = int(np.clip(len(candidates) * dims * pop_scaling, min_pop, max_pop))
    else:
        dynamic_pop = population_size
        
    print(f"Running MPSO on {X_search.shape[0]} strided samples...")
    problem = MPSOProjectionProblem(X_search, y_search, dims=dims, alpha=alpha, threshold=threshold, n_estimators=n_estimators, cv=cv)
    task = Task(problem, max_iters=mpso_iters)
    algorithm = ParticleSwarmOptimization(population_size=dynamic_pop, seed=seed)
    
    best_x, _ = algorithm.run(task)
    
    # Generate the "Recipe" matrix
    final_sel = (best_x > threshold).reshape((len(candidates), dims))
    projection_weights = np.sum(final_sel, axis=0) + 1e-12

    # --- FIX: Pass 3 - Apply to FULL Dataset ---
    print("Pass 3: Applying optimized projection to the FULL dataset...")
    full_results = []
    
    for chunk in df_iterator_factory():
        # 1. Project the candidate features
        X_chunk = chunk[candidates].values
        projected_chunk = np.matmul(X_chunk, final_sel) / projection_weights
        
        # 2. Rebuild the chunk with projected dimensions
        # Create meaningful column names based on selected features
        selected_feature_indices = np.where(best_x > threshold)[0]
        selected_feature_names = [candidates[i] for i in selected_feature_indices]
        
        # For each projected dimension, find the most contributing original features
        dim_columns = []
        for dim_idx in range(dims):
            # Get weights for this dimension across all selected features
            dim_weights = final_sel[:, dim_idx]
            # Get top contributing features (up to 3 for naming)
            top_features_idx = np.argsort(dim_weights)[-3:]  # Top 3 contributors
            top_features = [candidates[i] for i in top_features_idx if i < len(candidates)]
            dim_name = "_".join(top_features[:2])  # Use top 2 for column name
            dim_columns.append(f'MPSO_{dim_name}')
        
        res_chunk = pd.DataFrame(
            projected_chunk, 
            columns=dim_columns,
            index=chunk.index
        )
        
        # 3. Carry over target and EVERY available metadata column
        res_chunk[target_col] = chunk[target_col].values
        
        # Only grab metadata that actually exists in this specific chunk
        current_meta = [c for c in meta_to_keep if c in chunk.columns]
        for meta in current_meta:
            res_chunk[meta] = chunk[meta].values
        
        full_results.append(res_chunk)

    final_df = pd.concat(full_results, ignore_index=True)
    
    print(f"✅ MPSO Complete. Full dataset ({len(final_df)} rows) reduced to {dims} dimensions.")
    return final_df

if __name__ == "__main__":
    pass