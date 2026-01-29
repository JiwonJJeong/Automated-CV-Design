import amino_fast_mod as amino
import numpy as np
import pandas as pd
import gc
import kneed
import fisher_score_mod as fsm
from kneed import KneeLocator
import argparse
import os
from collections.abc import Iterable

def calculate_fisher_scores(df, target_col='class'):
    """
    Calculates Fisher scores and indices using fisher_score_mod.
    """
    print("Calculating Fisher scores...")
    X = df.drop(target_col, axis=1)
    y = df[target_col].to_numpy()
    
    # Calculate Fisher scores
    score = fsm.fisher_score(X.to_numpy(), y, mode='score')
    # Removed CSV saving of scores
    
    # Calculate Fisher indices
    idx = fsm.fisher_score(X.to_numpy(), y, mode='index')
    # Removed CSV saving of indices
    
    return score, idx

def find_optimal_n_feat(scores, S=5, curve='convex', direction='decreasing'):
    """
    Uses KneeLocator to find the elbow point in sorted Fisher scores.
    """
    print("Finding optimal number of features using Kneedle algorithm...")
    df_scores = pd.DataFrame(scores, columns=['fs'])
    df_scores['ind'] = range(len(df_scores))
    
    # Sort scores in descending order
    df_scores_sorted = df_scores.sort_values(by='fs', ascending=False).reset_index(drop=True)
    # Removed CSV saving of sorted scores
    
    x = list(range(len(df_scores_sorted)))
    y = df_scores_sorted['fs']
    
    kneedle = KneeLocator(x, y, S=S, curve=curve, direction=direction)
    knee_point = kneedle.knee
    
    print(f"The knee point is at nFeat = {knee_point}, with a Fisher score of: {kneedle.knee_y}")
    return knee_point, df_scores_sorted

def prepare_amino_input(df_selected):
    """
    Converts DataFrame columns to amino.OrderParameter objects.
    """
    print(f"Preparing {len(df_selected.columns)} features for AMINO...")
    gc.collect()
    all_ops = []
    for col in df_selected.columns:
        all_ops.append(amino.OrderParameter(col, df_selected[col].tolist()))
    
    print(f"Created {len(all_ops)} OrderParameters.")
    return all_ops

def run_amino(all_ops, max_outputs=5, bins=10):
    """
    Runs the AMINO algorithm to find the final reduced set of features.
    """
    print(f"Running AMINO with max_outputs={max_outputs}, bins={bins}...")
    gc.collect()
    # Explicitly disable file output by passing None if supported
    final_ops = amino.find_ops(all_ops, max_outputs, bins, distortion_filename=None)
    
    print("\nAMINO selected features:")
    for op in final_ops:
        print(op)
        
    # Removed distortion summary saving
        
    return final_ops

def run_fisher_amino_workflow(df, target_col='class', n_feat_amino=5, bins_amino=10):
    """
    Orchestrates the Fisher-AMINO feature selection workflow.
    Accepts DataFrame or Iterator[DataFrame].
    Returns Iterator[DataFrame] with selected features.
    """
    
    # 0. Handle Iterator -> Full DataFrame
    if isinstance(df, Iterable) and not isinstance(df, pd.DataFrame):
        print("Consuming DataFrame iterator for Fisher-AMINO...")
        df = pd.concat(df, ignore_index=True)

    # Validation
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")

    # 1. Calculate Fisher Scores
    scores, indices = calculate_fisher_scores(df, target_col)
    
    # 2. Find optimal number of features (Kneedle)
    n_feat_optimal, df_scores_sorted = find_optimal_n_feat(scores)
    
    # 3. Reduce features for AMINO
    X = df.drop(target_col, axis=1)
    # The indices in 'indices' are the column indices of X in descending order of Fisher score
    reduced_X = X.iloc[:, indices[0:n_feat_optimal]]
    
    # 4. Prepare OPs
    all_ops = prepare_amino_input(reduced_X)
    
    # 5. Run AMINO
    final_ops = run_amino(all_ops, max_outputs=n_feat_amino, bins=bins_amino)
    
    # 6. Final preparation
    final_col_names = [str(op) for op in final_ops]
    df_amino = reduced_X[final_col_names].copy()
    df_amino[target_col] = df[target_col].values
    
    print(f"Fisher-AMINO completed. Selected {len(final_col_names)} features.")
    
    # Return as iterator
    return [df_amino]

def main():
    parser = argparse.ArgumentParser(description='Refactored Fisher-AMINO Feature Selection')
    parser.add_argument('--dataset', type=str, default='sample_CA_post_variance.csv', help='Input CSV dataset')
    parser.add_argument('--target', type=str, default='class', help='Target column name')
    parser.add_argument('--max_outputs', type=int, default=5, help='Max outputs for AMINO')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for AMINO')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        return

    print(f"Loading dataset: {args.dataset}")
    df_iter = [pd.read_csv(args.dataset)] # Simulate iterator
    
    try:
        result_iter = run_fisher_amino_workflow(
            df=df_iter,
            target_col=args.target,
            n_feat_amino=args.max_outputs,
            bins_amino=args.bins
        )
        for res_df in result_iter:
            print(f"Result shape: {res_df.shape}")
            # res_df.to_csv('fisher_amino_result.csv', index=False)
            
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
