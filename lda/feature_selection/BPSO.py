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
            feature_cols = [c for c in get_feature_cols(chunk) 
                           if c != target_col and pd.api.types.is_numeric_dtype(chunk[c])]
        
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
from sklearn.preprocessing import StandardScaler
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

# Standardized helpers
from data_access import get_feature_cols, METADATA_COLS

from functools import lru_cache

import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

# Standardized helpers
from data_access import get_feature_cols, METADATA_COLS

# --- OPTIMIZED BPSO PROBLEM CLASS ---

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.95, threshold=0.5, cv=3):
        # 1. Performance: Scale once so LinearSVC converges instantly
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X_train)
        self.y_train = y_train
        
        super().__init__(dimension=X_train.shape[1], lower=0.0, upper=1.0)
        self.alpha = alpha
        self.threshold = threshold
        self.cv = cv
        self.eval_count = 0 
        self.cache = {} # 2. Performance: Don't re-run CV for same feature sets

        counts = np.unique(y_train, return_counts=True)[1]
        self.min_class_size = np.min(counts)

    def _evaluate(self, x):
        self.eval_count += 1
        
        # 3. Heartbeat: Visual confirmation that the code is running
        if self.eval_count % 50 == 0:
            print(f"  > Evaluations completed: {self.eval_count}...")
        elif self.eval_count % 5 == 0:
            print(".", end="", flush=True)

        selected = x > self.threshold 
        if not np.any(selected) or self.min_class_size < 2:
            return 1.0
        
        # 4. Cache check: Swarms often revisit the same local optima
        feature_key = tuple(selected)
        if feature_key in self.cache:
            return self.cache[feature_key]

        # 5. Fast-solver settings
        clf = LinearSVC(dual=False, tol=1e-2, max_iter=1000)
        
        try:
            cv_folds = min(self.cv, self.min_class_size)
            accuracy = cross_val_score(clf, self.X_scaled[:, selected], 
                                     self.y_train, cv=cv_folds, n_jobs=1).mean()
        except:
            accuracy = 0.0
            
        score = self.alpha * (1 - accuracy) + (1 - self.alpha) * (selected.sum() / self.dimension)
        
        self.cache[feature_key] = score
        return score

# --- MAIN PIPELINE ---

def run_bpso_pipeline(df_iterator_factory, target_col='class', candidate_limit=150, 
                       seed=None, stride=5, population_size=None, **kwargs):
    
    fisher_scores = compute_streaming_fisher(df_iterator_factory(), target_col, stride=stride)
    
    if fisher_scores.empty:
        return pd.DataFrame()

    candidates = fisher_scores.index[:candidate_limit].tolist()
    
    print(f"Pass 2: Loading {len(candidates)} features (stride={stride})...")
    data_list = []
    actual_metadata = []

    for chunk in df_iterator_factory():
        if stride > 1:
            chunk = chunk.iloc[::stride]
            
        if not actual_metadata:
            actual_metadata = [c for c in METADATA_COLS if c != target_col and c in chunk.columns]
            
        cols_to_keep = candidates + [target_col] + actual_metadata
        available = [c for c in cols_to_keep if c in chunk.columns]
        data_list.append(chunk[available])
    
    narrow_df = pd.concat(data_list, ignore_index=True)
    del data_list
    gc.collect()

    X = narrow_df[candidates].values
    y = narrow_df[target_col].values

    print(f"Beginning Swarm Optimization on {X.shape[0]} samples...")
    
    n_feats = X.shape[1]
    # Set a reasonable floor/ceiling for population and iterations for speed
    pop = population_size if population_size else int(np.clip(n_feats * 0.3, 20, 60))
    # We cap iterations to prevent "taking forever"
    iters = int(np.clip(n_feats * 0.4, 15, 40))

    problem = SVMFeatureSelection(X, y, cv=3)
    task = Task(problem, max_iters=iters)
    
    algorithm = ParticleSwarmOptimization(
        population_size=pop, seed=seed,
        w=kwargs.get('w', 0.729), c1=kwargs.get('c1', 1.49445), c2=kwargs.get('c2', 1.49445)
    )
    
    best_x, _ = algorithm.run(task)
    
    final_mask = best_x > 0.5
    final_features = [candidates[i] for i, m in enumerate(final_mask) if m]
    
    print(f"\n✅ Final selection: {len(final_features)} features.")
    
    print(f"Pass 3: Recovering all rows for {len(final_features)} features...")
    full_data_list = []
    for chunk in df_iterator_factory():
        available = [c for c in (final_features + [target_col] + actual_metadata) if c in chunk.columns]
        full_data_list.append(chunk[available])
    
    final_df = pd.concat(full_data_list, ignore_index=True)
    del narrow_df
    gc.collect()
    
    return final_df