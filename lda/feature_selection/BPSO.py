import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

# Standardized helpers
from data_access import get_feature_cols, METADATA_COLS

# --- PASS 1: STREAMING FISHER (UNCHANGED) ---
# Your implementation here was solid. 
# Keeping it exactly as is for brevity.
def compute_streaming_fisher(df_iterator, target_col, stride=1):
    # ... (Keep your exact existing code for this function) ...
    print(f"Pass 1: Filtering features via Streaming Fisher Score (stride={stride})...")
    stats = {}
    feature_cols = None
    total_n = 0

    for chunk in df_iterator:
        if stride > 1:
            chunk = chunk.iloc[::stride]
            
        if feature_cols is None:
            # Explicitly exclude metadata and the target column
            feature_cols = [c for c in get_feature_cols(chunk) 
                           if c != target_col and c not in METADATA_COLS and pd.api.types.is_numeric_dtype(chunk[c])]
        
        y = chunk[target_col].values
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

    global_sum = sum(s['sum'] for s in stats.values())
    global_mean = global_sum / total_n
    
    num = np.zeros(len(feature_cols))
    den = np.zeros(len(feature_cols))
    
    for label, s in stats.items():
        n_k = s['n']
        if n_k == 0: continue
            
        m_k = s['sum'] / n_k
        # Variance calculation
        ss_within = np.maximum(s['sum_sq'] - (s['sum']**2 / n_k), 0.0)
        
        num += n_k * (m_k - global_mean)**2
        den += ss_within
    
    scores = num / (den + 1e-12)
    
    return pd.Series(scores, index=feature_cols).sort_values(ascending=False)


import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

# Handle both direct execution and module import
try:
    from .visualization import visualize_bpso_diagnostics
except ImportError:
    # Direct execution - use absolute imports
    from visualization import visualize_bpso_diagnostics

# Standardized helpers
from data_access import get_feature_cols, METADATA_COLS

import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

# Handle both direct execution and module import
try:
    from .visualization import visualize_bpso_diagnostics
except ImportError:
    # Direct execution - use absolute imports
    from visualization import visualize_bpso_diagnostics

# Standardized helpers
from data_access import get_feature_cols, METADATA_COLS

# --- OPTIMIZED BPSO PROBLEM CLASS ---

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.95, threshold=0.5, cv=3):
        # Scaling removed: X_train is assumed to be pre-scaled by the pipeline
        self.X_train = X_train
        self.y_train = y_train
        
        super().__init__(dimension=X_train.shape[1], lower=0.0, upper=1.0)
        self.alpha = alpha
        self.threshold = threshold
        self.cv = cv
        self.eval_count = 0 
        self.cache = {} 

        counts = np.unique(y_train, return_counts=True)[1]
        self.min_class_size = np.min(counts) if len(counts) > 0 else 0

    def _evaluate(self, x):
        self.eval_count += 1
        if self.eval_count % 50 == 0:
            print(f"  > Evaluations completed: {self.eval_count}...")

        selected = x > self.threshold 
        if not np.any(selected) or self.min_class_size < 2:
            return 1.0
        
        feature_key = tuple(selected)
        if feature_key in self.cache:
            return self.cache[feature_key]

        clf = LinearSVC(dual=False, tol=1e-2, max_iter=1000)
        
        try:
            cv_folds = min(self.cv, self.min_class_size)
            accuracy = cross_val_score(clf, self.X_train[:, selected], 
                                     self.y_train, cv=cv_folds).mean()
        except:
            accuracy = 0.0
            
        score = self.alpha * (1 - accuracy) + (1 - self.alpha) * (selected.sum() / self.dimension)
        self.cache[feature_key] = score
        return score

def mrmr_ranker(X_df, fisher_series, n_selected):
    """
    X_df: DataFrame of candidate features (narrow_df)
    fisher_series: Series of pre-computed Fisher scores
    """
    features = fisher_series.index.tolist()
    selected = [features[0]]  # Start with the highest Fisher score
    unselected = features[1:]
    
    # Pre-compute correlation matrix for speed
    corr_matrix = X_df[features].corr().abs().fillna(0)

    while len(selected) < n_selected and unselected:
        mrmr_scores = []
        for f in unselected:
            relevance = fisher_series[f]
            # Average correlation with already selected features
            redundancy = corr_matrix.loc[f, selected].mean()
            
            # FCQ Formula (avoid division by zero)
            score = relevance / (redundancy + 1e-5)
            mrmr_scores.append((score, f))
        
        # Pick the feature with the best mRMR score
        best_feat = max(mrmr_scores, key=lambda x: x[0])[1]
        selected.append(best_feat)
        unselected.remove(best_feat)
        
    return selected

# --- MAIN PIPELINE ---

def run_bpso_pipeline(df_iterator_factory, target_col='class', candidate_limit=150, 
                       seed=None, stride=5, population_size=None, knee_sensitivity=2.0, 
                       n_particles=None, max_iter=None, mrmr_limit=50, w=0.729, c1=1.49445, c2=1.49445,
                       accuracy_sparsity_weight=0.95):
    
    # Ensure factory is callable (handle case where list/iterator is passed)
    if not callable(df_iterator_factory):
        print("Warning: df_iterator_factory is not callable. Caching data to memory.")
        cached_data = list(df_iterator_factory)
        def data_factory():
            return iter(cached_data)
        df_iterator_factory = data_factory

    # --- STEP 1: PROACTIVE DISCOVERY ---
    first_chunk = next(df_iterator_factory())
    all_cols = first_chunk.columns.tolist()
    
    # Identify buckets once to avoid if-statements in loops
    available_features = [c for c in get_feature_cols(first_chunk) if c != target_col]
    actual_metadata = [c for c in METADATA_COLS if c in all_cols and c != target_col]
    
    # --- STEP 2: PASS 1 (FISHER) ---
    fisher_scores = compute_streaming_fisher(df_iterator_factory(), target_col, stride=stride)
    if fisher_scores.empty: return pd.DataFrame()

    # Knee Detection / Candidate Selection
    if candidate_limit is None:
        from kneed import KneeLocator
        y_vals = fisher_scores.values
        kn = KneeLocator(range(len(y_vals)), y_vals, curve='convex', direction='decreasing', S=knee_sensitivity)
        cutoff_idx = kn.knee if kn.knee is not None else min(150, len(y_vals))
        candidates = fisher_scores.index[:cutoff_idx+1].tolist()
    else:
        candidates = fisher_scores.index[:candidate_limit].tolist()
    
    # Step 3: Refine candidates using mRMR
    print(f"Refining {len(candidates)} candidates using mRMR...")

    # We use the narrow_df (or a sample of it) to calculate redundancy
    search_chunks = []
    for chunk in df_iterator_factory():
        search_chunks.append(chunk.iloc[::stride][candidates + [target_col]])
    temp_df = pd.concat(search_chunks)

    # Rank top N features using mRMR
    mrmr_limit = min(mrmr_limit, len(candidates)) 
    candidates = mrmr_ranker(temp_df, fisher_scores.loc[candidates], mrmr_limit)

    # --- STEP 3: PASS 2 (LOAD SEARCH DATA FOR BPSO) ---
    # Now BPSO only works on the mRMR-refined independent features
    X = temp_df[candidates].values
    y = temp_df[target_col].values

    # --- STEP 4: OPTIMIZATION ---
    pop = n_particles or population_size or int(np.clip(X.shape[1] * 0.3, 20, 60))
    iters = max_iter or int(np.clip(X.shape[1] * 0.4, 15, 40))

    # Set global seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    problem = SVMFeatureSelection(X, y, cv=3, alpha=accuracy_sparsity_weight)
    task = Task(problem, max_iters=iters)
    
    # We remove seed=seed here to avoid potential NiaPy version conflicts
    algorithm = ParticleSwarmOptimization(population_size=pop, w=w, c1=c1, c2=c2)
    
    best_x, _ = algorithm.run(task)
    final_features = [candidates[i] for i, val in enumerate(best_x) if val > 0.5]
    
    # --- STEP 5: VISUALIZATION (Using centralized visualization.py) ---
    visualize_bpso_diagnostics(temp_df, fisher_scores, candidates, final_features, target_col)

    # --- STEP 6: PASS 3 (RECOVER FULL DATASET) ---
    print(f"Pass 3: Recovering all rows for {len(final_features)} features...")
    final_out_cols = final_features + [target_col] + actual_metadata
    
    return pd.concat([chunk[final_out_cols] for chunk in df_iterator_factory()])
