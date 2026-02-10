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
                       seed=None, stride=5, population_size=None, knee_S=2.0, **kwargs):
    
    fisher_scores = compute_streaming_fisher(df_iterator_factory(), target_col, stride=stride)
    
    if fisher_scores.empty:
        return pd.DataFrame()

    if candidate_limit is None:
        from kneed import KneeLocator
        y_vals = fisher_scores.values
        # Using S=2.0 for feature selection to be slightly more conservative than variance filtering
        kn = KneeLocator(range(len(y_vals)), y_vals, curve='convex', direction='decreasing', S=knee_S)
        cutoff_idx = kn.knee if kn.knee is not None else min(150, len(y_vals))
        candidates = fisher_scores.index[:cutoff_idx+1].tolist()
        print(f"Dynamic candidate selection: Fisher score knee at index {cutoff_idx} ({len(candidates)} features)")
    else:
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
    
    narrow_df = pd.concat(data_list, ignore_index=False)
    del data_list
    gc.collect()

    X = narrow_df[candidates].values
    y = narrow_df[target_col].values

    print(f"Beginning Swarm Optimization on {X.shape[0]} samples...")
    
    n_feats = X.shape[1]
    # Set a reasonable floor/ceiling for population and iterations for speed
    # RESPECT Overrides from pipeline_helper (n_particles and max_iter)
    pop = kwargs.get('n_particles', kwargs.get('population_size'))
    if pop is None:
        pop = int(np.clip(n_feats * 0.3, 20, 60))
        
    iters = kwargs.get('max_iter')
    if iters is None:
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
    
    # --- STANDARDIZED VISUALIZATION ---
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        sns.set_theme(style="whitegrid")
        
        # 1. Signal Strength (Fisher Scree Plot)
        y_vals = fisher_scores.values
        axes[0].plot(range(len(y_vals)), y_vals, color='grey', alpha=0.5)
        # Highlight selected features on the scree plot
        selected_indices = [fisher_scores.index.get_loc(f) for f in final_features if f in fisher_scores.index]
        axes[0].scatter(selected_indices, fisher_scores.iloc[selected_indices], color='red', s=40, label='BPSO Selected', zorder=5)
        axes[0].set_title("Feature Signal Strength (Fisher)", fontsize=14)
        axes[0].set_xlabel("Feature Rank")
        axes[0].set_ylabel("Fisher Score")
        axes[0].legend()
        
        # 2. Redundancy (Correlation Heatmap)
        if len(final_features) > 1:
            corr = narrow_df[final_features].corr()
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[1], annot=False)
            axes[1].set_title("Feature Redundancy (Correlation)", fontsize=14)
        else:
            axes[1].text(0.5, 0.5, "Need 2+ Features\nfor Heatmap", ha='center')
            
        # 3. State Space Mapping (2D Scatter)
        if len(final_features) >= 2:
            f1, f2 = final_features[0], final_features[1]
            sample_df = narrow_df.sample(min(2000, len(narrow_df)))
            sns.scatterplot(data=sample_df, x=f1, y=f2, hue=target_col, palette="deep", s=20, alpha=0.7, ax=axes[2])
            axes[2].set_title(f"State Space Mapping\n{f1} vs {f2}", fontsize=14)
        else:
            axes[2].text(0.5, 0.5, "Need 2+ Features\nfor Scatter Plot", ha='center')
            
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

    print(f"Pass 3: Recovering all rows for {len(final_features)} features...")
    full_data_list = []
    for chunk in df_iterator_factory():
        available = [c for c in (final_features + [target_col] + actual_metadata) if c in chunk.columns]
        full_data_list.append(chunk[available])
    
    final_df = pd.concat(full_data_list, ignore_index=False)
    del narrow_df
    gc.collect()
    
    return final_df