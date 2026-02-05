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
    """Memory-efficient Fisher scores with numerical stability fixes."""
    print("Pass 1: Filtering features via Streaming Fisher Score...")
    stats = {}
    feature_cols = None
    total_n = 0

    for chunk in df_iterator:
        # Pre-filter: Ensure we only pick numeric columns for features
        if feature_cols is None:
            # Helper to ignore objects/strings which crash Fisher math
            feature_cols = [c for c in get_feature_cols(chunk) 
                           if c != target_col and pd.api.types.is_numeric_dtype(chunk[c])]
        
        y = chunk[target_col].values
        # Optimization: Select columns once per chunk
        X_chunk = chunk[feature_cols].values
        
        for label in np.unique(y):
            mask = (y == label)
            data = X_chunk[mask]
            
            if label not in stats:
                stats[label] = {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
            
            stats[label]['n'] += data.shape[0]
            stats[label]['sum'] += np.sum(data, axis=0)
            stats[label]['sum_sq'] += np.sum(data**2, axis=0)
        
        total_n += len(chunk)

    if total_n == 0:
        return pd.Series(dtype=float)

    # --- Aggregation Step ---
    global_sum = sum(s['sum'] for s in stats.values())
    global_mean = global_sum / total_n
    
    num = np.zeros(len(feature_cols))
    den = np.zeros(len(feature_cols))
    
    for label, s in stats.items():
        n_k = s['n']
        if n_k == 0: continue
            
        m_k = s['sum'] / n_k
        
        # FIX 1: Numerical Stability for Variance
        # Clip negative variance to 0.0 to prevent NaNs
        ss_within = np.maximum(s['sum_sq'] - (s['sum']**2 / n_k), 0.0)
        
        num += n_k * (m_k - global_mean)**2
        den += ss_within
    
    # FIX 2: Handle constant features (den=0)
    # Add epsilon to denominator to prevent division by zero
    scores = num / (den + 1e-12)
    
    return pd.Series(scores, index=feature_cols).sort_values(ascending=False)

# --- BPSO PROBLEM CLASS ---

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.95, threshold=0.5, n_estimators=3, max_samples=0.8, cv=3):
        # FIX: Explicit float bounds
        super().__init__(dimension=X_train.shape[1], lower=0.0, upper=1.0)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.cv = cv

    def _evaluate(self, x):
        # Sigmoid function: maps velocity/position to a probability
        # 1 / (1 + exp(-x))
        probability = 1 / (1 + np.exp(-x))
        
        # In canonical BPSO, we'd use a random flip: np.random.rand() < probability
        # But for deterministic evaluation in NiaPy, thresholding the probability is safer:
        selected = probability > self.threshold
        
        n_selected = selected.sum()
        if n_selected == 0:
            return 1.0
        
        # Optimization: Reduce estimators to 3 for faster convergence during search
        clf = OneVsRestClassifier(BaggingClassifier(
            LinearSVC(dual=False, tol=1e-3), 
            n_jobs=1, # Important: Set inner jobs to 1 to avoid over-subscription if NiaPy parallelizes
            n_estimators=self.n_estimators, 
            max_samples=self.max_samples
        ))
        
        try:
            unique_labels, label_counts = np.unique(self.y_train, return_counts=True)
            min_class_size = np.min(label_counts)
            
            # FIX 3: Robust CV Strategy
            if min_class_size < 2:
                return 1.0  # Penalty for insufficient samples
            
            cv_folds = min(self.cv, min_class_size)
            accuracy = cross_val_score(clf, self.X_train[:, selected], self.y_train, cv=cv_folds, n_jobs=-1).mean()
        except ValueError:
            # Catch "n_splits=2 cannot be greater than the number of members in each class"
            return 1.0
        except Exception:
            return 1.0
            
        score = self.alpha * (1 - accuracy) + (1 - self.alpha) * (n_selected / self.X_train.shape[1])
        return score
        
# --- MAIN PIPELINE ---

def run_bpso_pipeline(df_iterator_factory, target_col='class', candidate_limit=150, seed=None, alpha=0.95, threshold=0.5, n_estimators=3, max_samples=0.8, cv=3, population_size=None, pop_scaling=0.2, min_pop=20, max_pop=60, iters_scaling=0.5, min_iters=30, max_iters=100):
    """
    Integrated Pipeline with Dynamic Scaling for Population and Iterations.
    """
    # 1. Pass 1: Statistics (Fisher Filter)
    fisher_scores = compute_streaming_fisher(df_iterator_factory(), target_col)
    
    if fisher_scores.empty:
        print("Warning: No data found during Fisher Pass.")
        return pd.DataFrame()

    candidates = fisher_scores.index[:candidate_limit].tolist()
    
    # 2. Pass 2: Selective RAM Load
    print(f"Pass 2: Loading top {len(candidates)} features into memory...")
    
    try:
        sample_chunk = next(df_iterator_factory())
        existing_metadata = [c for c in METADATA_COLS if c in sample_chunk.columns]
    except StopIteration:
        return pd.DataFrame()

    cols_to_keep = candidates + [target_col] + existing_metadata
    
    # Optimization 3: Memory-safe generation/extraction
    data_list = []
    for chunk in df_iterator_factory():
        available = [c for c in cols_to_keep if c in chunk.columns]
        data_list.append(chunk[available])
    
    narrow_df = pd.concat(data_list, ignore_index=True)
    del data_list
    gc.collect()

    X = narrow_df[candidates].values
    y = narrow_df[target_col].values

    # 3. Dynamic Optimization Parameters
    n_feats = X.shape[1]
    
    # Population scaling with manual override
    if population_size is None:
        dynamic_pop = int(np.clip(n_feats * pop_scaling, min_pop, max_pop))
    else:
        dynamic_pop = population_size
        
    # Iteration scaling with manual override  
    dynamic_iters = int(np.clip(n_feats * iters_scaling, min_iters, max_iters))

    print(f"Running BPSO on {n_feats} features (Pop: {dynamic_pop}, Iters: {dynamic_iters})...")
    
    problem = SVMFeatureSelection(X, y, alpha=alpha, threshold=threshold, n_estimators=n_estimators, max_samples=max_samples, cv=cv)
    task = Task(problem, max_iters=dynamic_iters)
    
    # Initialize algorithm with dynamic population
    algorithm = ParticleSwarmOptimization(population_size=dynamic_pop, seed=seed)
    
    best_x, _ = algorithm.run(task)
    
    # Final mask logic (matches the sigmoid/clamping in _evaluate)
    final_mask = (1 / (1 + np.exp(-best_x))) > threshold
    final_features = [candidates[i] for i, m in enumerate(final_mask) if m]
    
    print(f"Final selection: {len(final_features)} features.")
    
    # 4. Final Assembly
    other_cols = [c for c in narrow_df.columns if c not in candidates]
    return narrow_df[final_features + other_cols]

if __name__ == "__main__":
    pass