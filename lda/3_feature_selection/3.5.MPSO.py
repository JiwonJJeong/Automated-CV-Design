import numpy as np
import pandas as pd
import argparse
import os
import time
import gc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization
from collections.abc import Iterable

class SVMFeatureExpand(Problem):
    def __init__(self, Available_data_map, Preselected_data_map, X_train, y_train, 
                 feature_index1, feature_index2, num_features, n_residues,
                 constrain_aim_rescount, constrain_aim_respair, 
                 n_estimators, feature_excluding_ratio, alpha=0.99):
        super().__init__(dimension=Available_data_map.shape[0]*Available_data_map.shape[1], lower=0, upper=1)
        self.Available_data_map = Available_data_map
        self.Preselected_data_map = Preselected_data_map
        self.X_train = X_train
        self.y_train = y_train
        self.feature_index1 = feature_index1
        self.feature_index2 = feature_index2
        self.num_features = num_features
        self.n_residues = n_residues
        self.constrain_aim_rescount = constrain_aim_rescount
        self.constrain_aim_respair = constrain_aim_respair
        self.n_estimators = n_estimators
        self.feature_excluding_ratio = feature_excluding_ratio
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > self.feature_excluding_ratio
        selected_matrix_temp = selected.reshape(self.Available_data_map.shape[0], self.Available_data_map.shape[1])
        selected_matrix = (selected_matrix_temp * self.Available_data_map + self.Preselected_data_map).astype(bool)
        
        num_selected = np.sum(selected_matrix, axis=0)
        n_selected_residues = np.zeros(self.Available_data_map.shape[1])
        
        for dim in range(self.Available_data_map.shape[1]):
            residue_in_dim = np.concatenate((self.feature_index1[selected_matrix[:, dim]], self.feature_index2[selected_matrix[:, dim]]))
            n_selected_residues[dim] = len(np.unique(residue_in_dim))
            
        sum_selected_data = np.matmul(self.X_train, selected_matrix)
        data_averaged = np.divide(sum_selected_data, num_selected, out=np.zeros_like(sum_selected_data), where=num_selected != 0)
        
        if np.sum(num_selected) == 0:
            return 1.0
            
        clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=-1, n_estimators=self.n_estimators, max_samples=1.0/self.n_estimators), n_jobs=-1)
        accuracy = cross_val_score(clf, data_averaged, self.y_train, cv=3, n_jobs=-1).mean()
        score = 1 - accuracy
        
        n_selected_residues_penalty = (n_selected_residues - self.constrain_aim_respair).clip(min=0)
        num_selected_penalty = (num_selected - self.constrain_aim_rescount).clip(min=0)
        
        objfunc = self.alpha * score + (1 - self.alpha) * (
            np.sum(np.divide(num_selected_penalty, self.num_features, out=np.zeros_like(num_selected_penalty), where=self.num_features != 0, casting='unsafe')) + 
            np.sum(np.divide(n_selected_residues_penalty, self.n_residues, out=np.zeros_like(n_selected_residues_penalty), where=self.n_residues != 0))
        )
        
        print(f"Acc: {accuracy:.4f}, Obj: {objfunc:.4f}")
        return objfunc

def get_feature_metadata(feature_names):
    """
    Parses residue indices from feature names like 'res123.456'.
    """
    residue_list = np.sort(np.unique(','.join([','.join(i.split("res")[1].split(".")) for i in feature_names]).split(",")).astype(int))
    feature_index1 = np.array([np.where(residue_list == int(i.split("res")[1].split(".")[0]))[0][0] for i in feature_names])
    feature_index2 = np.array([np.where(residue_list == int(i.split("res")[1].split(".")[1]))[0][0] for i in feature_names])
    return feature_index1, feature_index2, len(residue_list)

def run_mpso_iteration(X_train, y_train, Available_data_map, Preselected_data_map, 
                       feature_index1, feature_index2, num_features, n_residues,
                       rescount, respair, estimators, ratio, alpha, iters, pop, seed,
                       iteration_idx):
    """
    Performs a single PSO iteration.
    """
    print(f"\n--- Starting MPSO Iteration {iteration_idx} ---")
    problem = SVMFeatureExpand(
        Available_data_map=Available_data_map,
        Preselected_data_map=Preselected_data_map,
        X_train=X_train,
        y_train=y_train,
        feature_index1=feature_index1,
        feature_index2=feature_index2,
        num_features=num_features,
        n_residues=n_residues,
        constrain_aim_rescount=rescount,
        constrain_aim_respair=respair,
        n_estimators=estimators,
        feature_excluding_ratio=ratio,
        alpha=alpha
    )
    
    task = Task(problem, max_iters=iters)
    algorithm = ParticleSwarmOptimization(population_size=pop, seed=seed)
    best_features, _ = algorithm.run(task)
    
    # Calculate final selection matrix for this iteration
    selected_matrix = (((best_features > ratio).reshape(Available_data_map.shape[0], Available_data_map.shape[1])) * Available_data_map + Preselected_data_map).astype(bool)
    
    return selected_matrix

def evaluate_selection(selected_matrix, X_train, X_test, y_train, y_test, feature_names, estimators):
    """
    Evaluates the final selected features on the test set.
    """
    num_selected = np.sum(selected_matrix, axis=0)
    print(f"Number of selected features: {num_selected}")
    
    if np.max(num_selected) < 20:
        for i in range(selected_matrix.shape[1]):
            print(f"Selected features in Dim{i}: {', '.join(feature_names[selected_matrix[:, i]])}")

    model_selected = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=1, n_estimators=estimators, max_samples=1.0/estimators), n_jobs=1)
    
    sum_train = np.matmul(X_train, selected_matrix)
    data_avg_train = np.divide(sum_train, num_selected, out=np.zeros_like(sum_train), where=num_selected != 0)

    sum_test = np.matmul(X_test, selected_matrix)
    data_avg_test = np.divide(sum_test, num_selected, out=np.zeros_like(sum_test), where=num_selected != 0)

    model_selected.fit(data_avg_train, y_train)
    accuracy = model_selected.score(data_avg_test, y_test)
    cv_accuracy = cross_val_score(model_selected, data_avg_test, y_test, cv=5, n_jobs=1).mean()
    
    print(f"Subset test accuracy: {accuracy:.4f}")
    print(f"Subset CV accuracy: {cv_accuracy:.4f}")
    
    return accuracy, data_avg_test

def run_multistage_mpso(df, target_col='class', 
                        dims=5, rescount=3, respair=1, preselected=None, 
                        start_iter=1, num_iters=1, resume_file=None, 
                        ratio=0.8, alpha=0.99, iters=200, pop=30, seed=123, estimators=10):
    """
    Orchestrates multiple stages of MPSO.
    Accepts DataFrame or Iterator[DataFrame].
    Returns Iterator[DataFrame] of the projected/reduced data.
    """

    # 0. Handle Iterator -> Full DataFrame
    if isinstance(df, Iterable) and not isinstance(df, pd.DataFrame):
        print("Consuming DataFrame iterator for MPSO...")
        df = pd.concat(df, ignore_index=True)

    # Validation
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")

    y = df[target_col].to_numpy()
    X = df.drop(target_col, axis=1).values
    feature_names = df.drop(target_col, axis=1).columns.values

    # Pre-computation for residue parsing
    feature_index1, feature_index2, _ = get_feature_metadata(feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    
    if preselected and isinstance(preselected, list):
        # args.preselected should be a list, already split in main
        Preselected_index = [np.where(feature_names == n)[0][0] for n in preselected if n != "" and n in feature_names]
    else:
        Preselected_index = []
        
    Preselected_data_map = np.zeros((len(feature_names), dims)).astype(bool)
    for i, idx in enumerate(Preselected_index):
        if i < dims:
            Preselected_data_map[idx, i] = 1

    # Starting state
    if resume_file:
         current_selected_matrix = np.load(resume_file).astype(bool)
    else:
         current_selected_matrix = np.ones((len(feature_names), dims)).astype(bool)

    for i in range(start_iter, start_iter + num_iters):
        Available_data_map = current_selected_matrix.copy()
        num_features = np.sum(Available_data_map, axis=0)
        n_residues = np.zeros(dims)
        for dim in range(dims):
            res_in_dim = np.concatenate((feature_index1[Available_data_map[:, dim]], feature_index2[Available_data_map[:, dim]]))
            n_residues[dim] = len(np.unique(res_in_dim))

        current_selected_matrix = run_mpso_iteration(
            X_train, y_train, Available_data_map, Preselected_data_map,
            feature_index1, feature_index2, num_features, n_residues,
            rescount=rescount, respair=respair, estimators=estimators, 
            ratio=ratio, alpha=alpha, iters=iters, pop=pop, seed=seed,
            iteration_idx=i
        )
        
        # Save intermediate state (Allowed checkpoint)
        save_path = f"selected_feature_matrix_iter{i}.npy"
        np.save(save_path, current_selected_matrix)
        print(f"Iteration {i} complete. Selection matrix saved to {save_path}")

        evaluate_selection(current_selected_matrix, X_train, X_test, y_train, y_test, feature_names, estimators=estimators)

    # Prepare final projected dataframe
    # Calculate projection for the *entire* dataset X (not just train/test split locally)
    num_selected = np.sum(current_selected_matrix, axis=0)
    sum_all = np.matmul(X, current_selected_matrix)
    data_avg_all = np.divide(sum_all, num_selected, out=np.zeros_like(sum_all), where=num_selected != 0)
    
    projected_df = pd.DataFrame(data_avg_all, columns=[f'Dim{i}' for i in range(dims)])
    projected_df[target_col] = y
    
    return [projected_df]

def main():
    parser = argparse.ArgumentParser(description='Multi-stage Particle Swarm Optimization (MPSO) for Feature Selection')
    parser.add_argument('--dataset', type=str, default='sample_CA_post_variance.csv', help='Input CSV dataset')
    parser.add_argument('--target', type=str, default='class', help='Target column name')
    parser.add_argument('--dims', type=int, default=5, help='Total dimensions to optimize')
    parser.add_argument('--rescount', type=int, default=3, help='Constrain aim residue count per dim')
    parser.add_argument('--respair', type=int, default=1, help='Constrain aim residue pair per dim')
    parser.add_argument('--preselected', type=str, default='', help='Comma-separated preselected features')
    parser.add_argument('--start_iter', type=int, default=1, help='Starting iteration index')
    parser.add_argument('--num_iters', type=int, default=1, help='Number of iterations to run')
    parser.add_argument('--resume_file', type=str, help='Path to .npy selection matrix to resume from')
    parser.add_argument('--ratio', type=float, default=0.8, help='Feature excluding ratio (default: 0.8)')
    parser.add_argument('--alpha', type=float, default=0.99, help='Weight for accuracy vs constraints')
    parser.add_argument('--iters', type=int, default=200, help='Max PSO internal iterations')
    parser.add_argument('--pop', type=int, default=30, help='PSO population size')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--estimators', type=int, default=10, help='Number of bagging estimators')

    args = parser.parse_args()
    args.preselected = args.preselected.split(",")

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        return
        
    print(f"Loading dataset: {args.dataset}")
    df_iter = [pd.read_csv(args.dataset)]

    start_time = time.time()
    
    try:
        result_iter = run_multistage_mpso(
            df_iter, 
            target_col=args.target, 
            dims=args.dims,
            rescount=args.rescount,
            respair=args.respair,
            preselected=args.preselected,
            start_iter=args.start_iter,
            num_iters=args.num_iters,
            resume_file=args.resume_file,
            ratio=args.ratio,
            alpha=args.alpha,
            iters=args.iters,
            pop=args.pop,
            seed=args.seed,
            estimators=args.estimators
        )
        
        for res_df in result_iter:
            print(f"MPSO Workflow completed. Result shape: {res_df.shape}")
            # res_df.to_csv('mpso_final_projected.csv', index=False)
            
    except ValueError as e:
        print(f"Error: {e}")
        
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
