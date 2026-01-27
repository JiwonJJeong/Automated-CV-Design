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
                       args, iteration_idx):
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
        constrain_aim_rescount=args.rescount,
        constrain_aim_respair=args.respair,
        n_estimators=args.estimators,
        feature_excluding_ratio=args.ratio,
        alpha=args.alpha
    )
    
    task = Task(problem, max_iters=args.iters)
    algorithm = ParticleSwarmOptimization(population_size=args.pop, seed=args.seed)
    best_features, _ = algorithm.run(task)
    
    # Calculate final selection matrix for this iteration
    selected_matrix = (((best_features > args.ratio).reshape(Available_data_map.shape[0], Available_data_map.shape[1])) * Available_data_map + Preselected_data_map).astype(bool)
    
    return selected_matrix

def evaluate_selection(selected_matrix, X_train, X_test, y_train, y_test, feature_names, feature_index1, feature_index2, args):
    """
    Evaluates the final selected features on the test set.
    """
    num_selected = np.sum(selected_matrix, axis=0)
    print(f"Number of selected features: {num_selected}")
    
    if np.max(num_selected) < 20:
        for i in range(selected_matrix.shape[1]):
            print(f"Selected features in Dim{i}: {', '.join(feature_names[selected_matrix[:, i]])}")

    model_selected = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), n_jobs=1, n_estimators=args.estimators, max_samples=1.0/args.estimators), n_jobs=1)
    
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

def run_multistage_mpso(df, y, args):
    """
    Orchestrates multiple stages of MPSO.
    """
    feature_names = df.columns.values
    feature_index1, feature_index2, _ = get_feature_metadata(feature_names)
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=args.seed)
    
    Preselected_index = [np.where(feature_names == n)[0][0] for n in args.preselected if n != ""]
    Preselected_data_map = np.zeros((df.shape[1], args.dims)).astype(bool)
    for i, idx in enumerate(Preselected_index):
        if i < args.dims:
            Preselected_data_map[idx, i] = 1

    # Starting state
    current_selected_matrix = np.ones((df.shape[1], args.dims)).astype(bool) if args.start_iter == 1 else np.load(args.resume_file).astype(bool)

    for i in range(args.start_iter, args.start_iter + args.num_iters):
        Available_data_map = current_selected_matrix.copy()
        num_features = np.sum(Available_data_map, axis=0)
        n_residues = np.zeros(args.dims)
        for dim in range(args.dims):
            res_in_dim = np.concatenate((feature_index1[Available_data_map[:, dim]], feature_index2[Available_data_map[:, dim]]))
            n_residues[dim] = len(np.unique(res_in_dim))

        current_selected_matrix = run_mpso_iteration(
            X_train, y_train, Available_data_map, Preselected_data_map,
            feature_index1, feature_index2, num_features, n_residues,
            args, i
        )
        
        # Save intermediate state
        save_path = f"selected_feature_matrix_iter{i}.npy"
        np.save(save_path, current_selected_matrix)
        print(f"Iteration {i} complete. Selection matrix saved to {save_path}")

        evaluate_selection(current_selected_matrix, X_train, X_test, y_train, y_test, feature_names, feature_index1, feature_index2, args)

    return current_selected_matrix

def main():
    parser = argparse.ArgumentParser(description='Multi-stage Particle Swarm Optimization (MPSO) for Feature Selection')
    parser.add_argument('--dataset', type=str, default='sample_CA_post_variance.csv', help='Input CSV dataset')
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

    start_time = time.time()
    df = pd.read_csv(args.dataset)
    y = df['class'].to_numpy()
    df = df.drop(columns=['class'])
    
    run_multistage_mpso(df, y, args)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
