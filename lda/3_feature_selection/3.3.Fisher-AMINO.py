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

# Import refactored helpers
from data_access import get_feature_cols, METADATA_COLS

# --- PASS 1: STREAMING FISHER ---

def compute_streaming_fisher(df_iterator_factory, target_col):
    """Memory-efficient Fisher scoring using dynamic class initialization."""
    print("Pass 1: Filtering features via Streaming Fisher Score...")
    stats = {} # {feature: {class_id: [count, sum, sum_sq]}}
    global_sum = {}
    global_sum_sq = {}
    total_count = 0

    for chunk in df_iterator_factory():
        feature_cols = [c for c in get_feature_cols(chunk) if c != target_col]
        # Force integer classes to prevent KeyError/Type Mismatch
        y = chunk[target_col].astype(int).values
        current_chunk_classes = np.unique(y)

        for col in feature_cols:
            if col not in stats:
                stats[col] = {}
                global_sum[col] = 0.0
                global_sum_sq[col] = 0.0

            for cls in current_chunk_classes:
                cls_int = int(cls)
                if cls_int not in stats[col]:
                    stats[col][cls_int] = [0, 0.0, 0.0]

            x = chunk[col].values
            global_sum[col] += np.sum(x)
            global_sum_sq[col] += np.sum(x**2)

            for cls in current_chunk_classes:
                cls_int = int(cls)
                mask = (y == cls_int)
                cls_data = x[mask]
                if len(cls_data) > 0:
                    stats[col][cls_int][0] += len(cls_data)
                    stats[col][cls_int][1] += np.sum(cls_data)
                    stats[col][cls_int][2] += np.sum(cls_data**2)
        
        total_count += len(chunk)

    # Compute Final Scores
    fisher_scores = {}
    for col in stats:
        m_total = global_sum[col] / total_count
        num, den = 0.0, 0.0
        for cls, (n_i, s_i, ss_i) in stats[col].items():
            if n_i < 2: continue
            m_i = s_i / n_i
            num += n_i * (m_i - m_total)**2
            den += (ss_i - (s_i**2 / n_i)) # Within-class SS
        fisher_scores[col] = num / (den + 1e-12)

    return pd.Series(fisher_scores).sort_values(ascending=False)

# --- BPSO PROBLEM CLASS ---

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.95):
        # dimension = number of candidate features
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train, self.y_train = X_train, y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        if selected.sum() == 0: 
            return 1.0 # Maximum penalty for selecting zero features
        
        # Use a fast LinearSVC with Bagging to handle variance
        clf = OneVsRestClassifier(BaggingClassifier(
            LinearSVC(dual=False, tol=1e-3), 
            n_jobs=-1, n_estimators=5, max_samples=0.8
        ))
        
        try:
            # Objective: Maximize Accuracy and Minimize feature count
            acc = cross_val_score(clf, self.X_train[:, selected], self.y_train, cv=3).mean()
        except:
            return 1.0
            
        # Fitness: Weighted average of error and feature ratio
        error_rate = 1 - acc
        feature_penalty = selected.sum() / self.dimension
        return self.alpha * error_rate + (1 - self.alpha) * feature_penalty

# --- MAIN PIPELINE ---

def run_bpso_pipeline(df_iterator_factory, target_col='class', candidate_limit=150, bpso_iters=30):
    """
    Three-Pass BPSO Pipeline:
    1. Pass 1: Streaming Fisher Score to find top N candidates.
    2. Pass 2: Extract only candidates into RAM.
    3. Optimization: Run BPSO on the RAM-resident slice.
    """
    
    # 1. Narrow the search space (Pass 1)
    fisher_scores = compute_streaming_fisher(df_iterator_factory, target_col)
    candidates = fisher_scores.index[:candidate_limit].tolist()
    
    # 2. Extract Narrow Slice into RAM (Pass 2)
    print(f"Loading top {len(candidates)} candidates for BPSO optimization...")
    data_list = []
    # Include metadata like 'frame_number' for tracking if needed
    cols_to_load = candidates + [target_col]
    
    for chunk in df_iterator_factory():
        # Force integer target to match Pass 1
        chunk[target_col] = chunk[target_col].astype(int)
        data_list.append(chunk[cols_to_load])
    
    narrow_df = pd.concat(data_list, ignore_index=True)
    del data_list # Free memory
    gc.collect()

    X = narrow_df[candidates].values
    y = narrow_df[target_col].values

    # 3. Run BPSO Optimization
    print(f"BPSO: Optimizing subset of {len(candidates)} features...")
    problem = SVMFeatureSelection(X, y)
    task = Task(problem, max_iters=bpso_iters)
    algorithm = ParticleSwarmOptimization(population_size=15)
    
    best_x, _ = algorithm.run(task)
    final_mask = best_x > 0.5
    final_features = [candidates[i] for i, m in enumerate(final_mask) if m]
    
    print(f"BPSO selected {len(final_features)} elite features.")

    return narrow_df[final_features + [target_col]]

if __name__ == "__main__":
    pass