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

def get_feature_metadata_from_h5(h5_path, dataset_name='data'):
    """Extracts and parses residue indices from H5 attributes."""
    with h5py.File(h5_path, 'r') as f:
        feature_names = f[dataset_name].attrs.get('column_names', [])
    
    # Logic to parse 'res123.456'
    residue_list = np.sort(np.unique(','.join([','.join(i.split("res")[1].split(".")) 
                           for i in feature_names if "res" in i]).split(",")).astype(int))
    
    f_idx1 = np.array([np.where(residue_list == int(i.split("res")[1].split(".")[0]))[0][0] 
                       for i in feature_names if "res" in i])
    f_idx2 = np.array([np.where(residue_list == int(i.split("res")[1].split(".")[1]))[0][0] 
                       for i in feature_names if "res" in i])
    
    return f_idx1, f_idx2, len(residue_list), feature_names

# --- PASS 1: STREAMING FILTER (Same as Fisher/Chi2 logic) ---

def get_candidate_features(h5_path, target_col, limit=500):
    """Streams H5 to find top N candidates so the PSO fits in RAM."""
    print(f"Pass 1: Filtering to top {limit} candidates via Fisher Score...")
    # ... (Insert the compute_streaming_fisher logic from previous response) ...
    # For brevity, assume this returns a list of column indices
    return list(range(limit)) 

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

def run_h5_multistage_mpso(h5_path, target_col, dims=5, candidate_limit=500):
    # 1. Setup Metadata & Candidates
    f_idx1, f_idx2, n_res, all_names = get_feature_metadata_from_h5(h5_path)
    candidate_indices = get_candidate_features(h5_path, target_col, limit=candidate_limit)
    
    # 2. Load Narrow Slice for PSO
    with h5py.File(h5_path, 'r') as f:
        X_candidates = f['data'][:, candidate_indices]
        y = f['data'][:, list(all_names).index(target_col)]
        
    X_train, X_test, y_train, y_test = train_test_split(X_candidates, y, test_size=0.3, stratify=y)

    # 3. PSO Loop
    # Available_data_map now refers to the indices of the CANDIDATES
    cur_map = np.ones((len(candidate_indices), dims)).astype(bool)
    
    # (Simulating one iteration for this example)
    problem = SVMFeatureExpandH5(cur_map, np.zeros_like(cur_map), X_train, y_train, 
                                f_idx1[candidate_indices], f_idx2[candidate_indices], 
                                None, None, 3, 1, 10, 0.8)
    
    best_x, _ = ParticleSwarmOptimization(population_size=30).run(Task(problem, max_iters=50))
    final_sel = (best_x > 0.8).reshape(cur_map.shape)

    # 4. Final Projection (Streaming the whole H5 file)
    print("Projecting full dataset in chunks...")
    projected_results = []
    
    with h5py.File(h5_path, 'r') as f:
        dataset = f['data']
        for i in range(0, dataset.shape[0], 10000):
            chunk = dataset[i:i+10000, candidate_indices]
            # Matrix multiply chunk by selection matrix
            sum_chunk = np.matmul(chunk, final_sel)
            avg_chunk = sum_chunk / np.sum(final_sel, axis=0)
            projected_results.append(avg_chunk)
            
    final_data = np.vstack(projected_results)
    df_final = pd.DataFrame(final_data, columns=[f'Dim{i}' for i in range(dims)])
    df_final[target_col] = y
    
    return df_final