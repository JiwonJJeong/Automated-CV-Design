import numpy as np
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization

class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99, n_estimators=10):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.n_estimators = n_estimators

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        clf = OneVsRestClassifier(
            BaggingClassifier(
                LinearSVC(dual=False), 
                n_jobs=-1, 
                n_estimators=self.n_estimators, 
                max_samples=1.0/self.n_estimators
            ), 
            n_jobs=-1
        )

        accuracy = cross_val_score(clf, self.X_train[:, selected], self.y_train, cv=3, n_jobs=-1).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)

def run_bpso_workflow(df, target_col='class', alpha=0.99, max_iters=100, population_size=30, seed=1234, n_estimators=10):
    """
    Runs the BPSO feature selection workflow.
    """
    y = df[target_col].to_numpy()
    X = df.drop(target_col, axis=1).values
    feature_names = df.drop(target_col, axis=1).columns.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    print(f"Starting BPSO with {X.shape[1]} features...")
    problem = SVMFeatureSelection(X_train, y_train, alpha=alpha, n_estimators=n_estimators)
    task = Task(problem, max_iters=max_iters)
    algorithm = ParticleSwarmOptimization(population_size=population_size, seed=seed)
    
    best_features_raw, _ = algorithm.run(task)
    selected_mask = best_features_raw > 0.5
    
    selected_feature_names = feature_names[selected_mask]
    print(f"Number of selected features: {selected_mask.sum()}")
    print(f"Selected features: {', '.join(selected_feature_names)}")

    # Final evaluation
    model_selected = OneVsRestClassifier(
        BaggingClassifier(
            LinearSVC(dual=False), 
            n_jobs=1, 
            n_estimators=n_estimators, 
            max_samples=1.0/n_estimators
        ), 
        n_jobs=1
    )
    
    model_selected.fit(X_train[:, selected_mask], y_train)
    test_score = model_selected.score(X_test[:, selected_mask], y_test)
    cv_score = cross_val_score(model_selected, X_test[:, selected_mask], y_test, cv=5, n_jobs=1).mean()
    
    print(f"Subset accuracy on test: {test_score:.4f}")
    print(f"Subset CV accuracy on test: {cv_score:.4f}")

    # Save results
    df_red = df[list(selected_feature_names) + [target_col]]
    output_filename = 'bpso_selected_features.csv'
    df_red.to_csv(output_filename, index=False)
    print(f"Saved selected features to {output_filename}")
    
    return selected_feature_names

def main():
    parser = argparse.ArgumentParser(description='Binary Particle Swarm Optimization (BPSO) for Feature Selection')
    parser.add_argument('--dataset', type=str, default='sample_CA_post_variance.csv', help='Input CSV dataset')
    parser.add_argument('--target', type=str, default='class', help='Target column name')
    parser.add_argument('--alpha', type=float, default=0.99, help='Weight for accuracy vs feature count (default: 0.99)')
    parser.add_argument('--iters', type=int, default=100, help='Maximum PSO iterations (default: 100)')
    parser.add_argument('--pop', type=int, default=30, help='PSO population size (default: 30)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--estimators', type=int, default=10, help='Number of bagging estimators')

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        return

    df = pd.read_csv(args.dataset)
    run_bpso_workflow(
        df=df,
        target_col=args.target,
        alpha=args.alpha,
        max_iters=args.iters,
        population_size=args.pop,
        seed=args.seed,
        n_estimators=args.estimators
    )

if __name__ == '__main__':
    main()
