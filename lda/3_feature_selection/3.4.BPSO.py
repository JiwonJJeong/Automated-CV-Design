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
    def __init__(self, X_train, y_train, alpha=0.95):
        # FIX: Explicit float bounds
        super().__init__(dimension=X_train.shape[1], lower=0.0, upper=1.0)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        n_selected = selected.sum()
        
        if n_selected == 0: 
            return 1.0
        
        # Optimization: Reduce estimators to 3 for faster convergence during search
        clf = OneVsRestClassifier(BaggingClassifier(
            LinearSVC(dual=False, tol=1e-3), 
            n_jobs=1, # Important: Set inner jobs to 1 to avoid over-subscription if NiaPy parallelizes
            n_estimators=3, 
            max_samples=0.8
        ))
        
        try:
            unique_labels, label_counts = np.unique(self.y_train, return_counts=True)
            min_class_size = np.min(label_counts)
            
            # FIX 3: Robust CV Strategy
            if min_class_size < 2:
                # Cannot do CV with a singleton class. 
                # Fallback: simple train/test split or penalty.
                return 1.0 # Penalize solutions that rely on insufficient data
            
            # Standard logic: Use 3-fold if possible, otherwise LOO equivalent
            cv_splits = 3 if min_class_size >= 3 else 2
            
            acc = cross_val_score(
                clf, 
                self.X_train[:, selected], 
                self.y_train, 
                cv=cv_splits, 
                scoring='accuracy',
                n_jobs=-1 # Parallelize at the CV level
            ).mean()
            
        except ValueError:
            # Catch "n_splits=2 cannot be greater than the number of members in each class"
            return 1.0
        except Exception:
            return 1.0
            
        # Objective: Minimize Error (1-acc) + Penalty for feature count
        score = self.alpha * (1.0 - acc) + (1.0 - self.alpha) * (n_selected / self.dimension)
        return score
        
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