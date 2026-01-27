import amino_fast_mod as amino
import numpy as np
import pandas as pd
import gc
import kneed
import fisher_score_mod as fsm
from kneed import KneeLocator
import argparse
import os

def calculate_fisher_scores(df, target_col='class'):
    """
    Calculates Fisher scores and indices using fisher_score_mod.
    """
    print("Calculating Fisher scores...")
    X = df.drop(target_col, axis=1)
    y = df[target_col].to_numpy()
    
    # Calculate Fisher scores
    score = fsm.fisher_score(X.to_numpy(), y, mode='score')
    df_scores = pd.DataFrame(score)
    df_scores.to_csv('fisher_scores_dfres.csv', encoding='utf-8', index=True, header=None)
    
    # Calculate Fisher indices
    idx = fsm.fisher_score(X.to_numpy(), y, mode='index')
    df_indices = pd.DataFrame(idx)
    df_indices.to_csv('fisher_indices_dfres.csv', encoding='utf-8', index=True, header=None)
    
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
    df_scores_sorted.to_csv('fisher_scores_dfres_descending.csv', index=False)
    
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

def run_amino(all_ops, max_outputs=5, bins=10, distortion_filename='distortion_array'):
    """
    Runs the AMINO algorithm to find the final reduced set of features.
    """
    print(f"Running AMINO with max_outputs={max_outputs}, bins={bins}...")
    gc.collect()
    final_ops = amino.find_ops(all_ops, max_outputs, bins, distortion_filename=distortion_filename)
    
    # Save final OPs to file
    with open('output2_final_OPs.txt', 'w') as f:
        print("\nAMINO order parameters:")
        for op in final_ops:
            f.write(f"{op}\n")
            print(op)
            
    # Save distortion summary
    if os.path.exists(f'{distortion_filename}.npy'):
        data_array = np.load(f'{distortion_filename}.npy')
        with open('output3_distortion_array.txt', 'w') as f:
            f.write(f"{data_array}\n")
        print("\nData summary:\n", data_array)
        
    return final_ops

def run_fisher_amino_workflow(df, target_col='class', n_feat_amino=5, bins_amino=10):
    """
    Orchestrates the Fisher-AMINO feature selection workflow.
    """
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
    
    # 6. Final save (merging with classes)
    final_col_names = [str(op) for op in final_ops]
    df_amino = reduced_X[final_col_names].copy()
    df_amino[target_col] = df[target_col].values
    df_amino.to_csv('fisher-amino.csv', index=False)
    print(f"Successfully saved final feature set to fisher-amino.csv with {len(final_col_names)} features.")
    
    return final_ops

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
    df = pd.read_csv(args.dataset)
    
    run_fisher_amino_workflow(
        df=df,
        target_col=args.target,
        n_feat_amino=args.max_outputs,
        bins_amino=args.bins
    )

if __name__ == "__main__":
    main()
