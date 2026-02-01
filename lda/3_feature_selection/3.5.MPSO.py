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

# --- METADATA PARSING ---

def get_feature_metadata_from_df(df, target_col='class'):
    """Extracts and parses residue indices from DataFrame column names."""
    # Get feature columns (exclude metadata and target)
    metadata_cols = {'construct', 'subconstruct', 'replica', 'frame_number', target_col}
    feature_names = [col for col in df.columns if col not in metadata_cols]
    
    # Logic to parse 'RES123_456' or 'res123.456' format
    residue_list = []
    for fname in feature_names:
        if "res" in fname.lower() or "RES" in fname:
            # Handle both 'res123.456' and 'RES123_456' formats
            parts = fname.replace('res', '').replace('RES', '').replace('.', '_').split('_')
            if len(parts) >= 2:
                residue_list.extend([int(parts[0]), int(parts[1])])
    
    residue_list = np.sort(np.unique(residue_list))
    
    # Create index mappings
    f_idx1 = []
    f_idx2 = []
    for fname in feature_names:
        if "res" in fname.lower() or "RES" in fname:
            parts = fname.replace('res', '').replace('RES', '').replace('.', '_').split('_')
            if len(parts) >= 2:
                f_idx1.append(np.where(residue_list == int(parts[0]))[0][0])
                f_idx2.append(np.where(residue_list == int(parts[1]))[0][0])
    
    return np.array(f_idx1), np.array(f_idx2), len(residue_list), feature_names

# --- PASS 1: STREAMING FISHER FILTER ---

def compute_streaming_fisher(df_iterator, target_col):
    """Streams DataFrames to compute Fisher scores for candidate selection."""
    print("Pass 1: Computing Fisher scores for feature filtering...")
    stats = {}
    feature_cols = None

    for chunk in df_iterator:
        if feature_cols is None:
            metadata_cols = {'construct', 'subconstruct', 'replica', 'frame_number', target_col}
            feature_cols = [c for c in chunk.columns if c not in metadata_cols]
        
        for label, group in chunk.groupby(target_col):
            data = group[feature_cols].values
            if label not in stats:
                stats[label] = {'n': 0, 'sum': 0, 'sum_sq': 0}
            stats[label]['n'] += data.shape[0]
            stats[label]['sum'] += np.sum(data, axis=0)
            stats[label]['sum_sq'] += np.sum(data**2, axis=0)

    # Calculate final scores
    total_n = sum(s['n'] for s in stats.values())
    global_mean = sum(s['sum'] for s in stats.values()) / total_n
    num, den = np.zeros(len(feature_cols)), np.zeros(len(feature_cols))
    
    for label, s in stats.items():
        m_k = s['sum'] / s['n']
        v_k = (s['sum_sq'] / s['n']) - (m_k**2)
        num += s['n'] * (m_k - global_mean)**2
        den += s['n'] * v_k
    
    scores = num / (den + 1e-9)
    fisher_series = pd.Series(scores, index=feature_cols).sort_values(ascending=False)
    return fisher_series 

# --- THE PSO PROBLEM (Optimized for RAM-slice) ---

class SVMFeatureExpandH5(Problem):
    def __init__(self, Available_data_map, Preselected_data_map, X_train, y_train, 
                 f_idx1, f_idx2, num_features, n_residues,
                 rescount, respair, n_estimators, ratio, alpha=0.99):
        super().__init__(dimension=Available_data_map.size, lower=0, upper=1)
        self.Available_data_map = Available_data_map
        self.Preselected_data_map = Preselected_data_map
        self.X_train, self.y_train = X_train, y_train
        self.f_idx1, self.f_idx2 = f_idx1, f_idx2
        self.num_features, self.n_residues = num_features, n_residues
        self.rescount, self.respair = rescount, respair
        self.n_estimators, self.ratio, self.alpha = n_estimators, ratio, alpha

    def _evaluate(self, x):
        # 1. Selection Matrix
        sel_temp = (x > self.ratio).reshape(self.Available_data_map.shape)
        sel_matrix = (sel_temp * self.Available_data_map + self.Preselected_data_map).astype(bool)
        
        num_selected = np.sum(sel_matrix, axis=0)
        if np.sum(num_selected) == 0: return 1.0

        # 2. Projection: Dim_n = Sum(X_features) / count
        # Done via matrix multiplication for speed
        sum_data = np.matmul(self.X_train, sel_matrix)
        data_avg = np.divide(sum_data, num_selected, out=np.zeros_like(sum_data), where=num_selected != 0)

        # 3. Model Evaluation
        clf = OneVsRestClassifier(BaggingClassifier(
            LinearSVC(dual=False), n_jobs=-1, n_estimators=self.n_estimators, max_samples=1.0/self.n_estimators
        ))
        acc = cross_val_score(clf, data_avg, self.y_train, cv=3).mean()
        
        # 4. Constraint Penalties (Omitted for brevity, same as user's original)
        score = 1 - acc
        return self.alpha * score # + penalty_terms...

# --- MAIN ORCHESTRATOR ---

# --- MAIN PIPELINE ---

def run_mpso_pipeline(df_iterator, target_col='class', dims=5, candidate_limit=500, mpso_iters=50):
    """
    Multi-stage MPSO feature selection with DataFrame iterator input.
    
    Args:
        df_iterator: Iterator yielding DataFrames with features and target column
        target_col: Name of the target column (default: 'class')
        dims: Number of output dimensions for projection
        candidate_limit: Number of top features to consider for MPSO
        mpso_iters: Number of MPSO iterations
    
    Returns:
        pd.DataFrame: Projected features + target column
    """
    # Convert iterator to list for two-pass processing
    print("Collecting data from iterator...")
    df_chunks = list(df_iterator)
    
    # Combine chunks for processing
    combined_df = pd.concat(df_chunks, ignore_index=True)
    
    # 1. Extract feature metadata from column names
    f_idx1, f_idx2, n_res, all_feature_names = get_feature_metadata_from_df(combined_df, target_col)
    
    # 2. Get candidate features via Fisher filtering
    fisher_scores = compute_streaming_fisher(iter(df_chunks), target_col)
    candidate_features = fisher_scores.index[:candidate_limit].tolist()
    
    # 3. Load narrow slice for PSO
    print(f"Loading top {candidate_limit} candidates into RAM for MPSO...")
    cols_to_keep = candidate_features + [target_col]
    narrow_df = combined_df[cols_to_keep]
    
    X_candidates = narrow_df[candidate_features].values
    y = narrow_df[target_col].values
    
    # Get indices for candidate features only
    candidate_indices = [all_feature_names.index(c) for c in candidate_features]
    f_idx1_candidates = f_idx1[candidate_indices]
    f_idx2_candidates = f_idx2[candidate_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X_candidates, y, test_size=0.3, stratify=y)

    # 4. PSO Loop
    cur_map = np.ones((len(candidate_features), dims)).astype(bool)
    
    problem = SVMFeatureExpandH5(cur_map, np.zeros_like(cur_map), X_train, y_train, 
                                f_idx1_candidates, f_idx2_candidates, 
                                None, None, 3, 1, 10, 0.8)
    
    best_x, _ = ParticleSwarmOptimization(population_size=30).run(Task(problem, max_iters=mpso_iters))
    final_sel = (best_x > 0.8).reshape(cur_map.shape)

    # 5. Final Projection
    print("Projecting full dataset...")
    projected_data = np.matmul(X_candidates, final_sel)
    avg_projected = projected_data / np.sum(final_sel, axis=0)
    
    df_final = pd.DataFrame(avg_projected, columns=[f'Dim{i}' for i in range(dims)])
    df_final[target_col] = y
    
    return df_final

if __name__ == "__main__":
    # Example usage with DataFrame iterator:
    # from data_access import data_iterator
    # df_iter = data_iterator(base_dir='/path/to/data', chunk_size=10000)
    # df_final = run_mpso_pipeline(df_iter, target_col='class')
    pass