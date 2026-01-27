import amino_fast_mod as amino
import numpy as np
import pandas as pd
import gc
import argparse
import os
from sklearn.feature_selection import SelectKBest, chi2
from kneed import KneeLocator

def calculate_chi_scores(df, target_col='class', q=5):
    """
    Discretizes continuous variables and calculates Chi-Squared scores.
    """
    print(f"Discretizing features into {q} bins and calculating Chi-Squared scores...")
    
    # 1. Separate features and target
    X_cont = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Binning (Quantile-based)
    binned_columns = {col: pd.qcut(X_cont[col], q=q, labels=False, duplicates='drop') for col in X_cont}
    binned_df = pd.DataFrame(binned_columns)
    
    # 3. Calculate Chi-Squared scores
    test = SelectKBest(score_func=chi2, k='all')
    test.fit(binned_df, y)
    
    chi_scores = test.scores_
    # Sort indices by scores in descending order
    sorted_indices = np.argsort(chi_scores)[::-1]
    sorted_scores = chi_scores[sorted_indices]
    
    return chi_scores, sorted_indices, sorted_scores

def find_optimal_n_feat(sorted_scores, S=100, curve='convex', direction='decreasing'):
    """
    Uses KneeLocator to find the elbow point in sorted Chi-Squared scores.
    """
    print("Finding optimal number of features using Kneedle algorithm...")
    x = range(1, len(sorted_scores) + 1)
    
    knee_locator = KneeLocator(x, sorted_scores, curve=curve, direction=direction, S=S)
    knee_point = knee_locator.knee
    
    print(f"The knee point is at nFeat = {knee_point}, with a Chi score is: {knee_locator.knee_y}")
    return knee_point

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

def run_amino(all_ops, max_outputs=10, bins=30, distortion_filename='distortion_array'):
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
            
    return final_ops

def run_chi_amino_workflow(df, target_col='class', max_outputs_amino=10, bins_amino=30):
    """
    Orchestrates the Chi-Squared AMINO feature selection workflow.
    """
    # 1. Calculate Chi-Squared Scores
    chi_scores, sorted_indices, sorted_scores = calculate_chi_scores(df, target_col)
    
    # 2. Find optimal number of features (Kneedle)
    n_feat_optimal = find_optimal_n_feat(sorted_scores)
    
    # 3. Reduce features for AMINO (on original continuous data)
    X_orig = df.drop(columns=[target_col])
    # Select top N features based on sorted indices
    selected_feature_names = X_orig.columns[sorted_indices[0:n_feat_optimal]]
    reduced_X = X_orig[selected_feature_names]
    
    # 4. Prepare OPs
    all_ops = prepare_amino_input(reduced_X)
    
    # 5. Run AMINO
    final_ops = run_amino(all_ops, max_outputs=max_outputs_amino, bins=bins_amino)
    
    # 6. Final save (merging with classes)
    final_col_names = [str(op) for op in final_ops]
    df_amino = X_orig[final_col_names].copy()
    df_amino[target_col] = df[target_col].values
    df_amino.to_csv('chi_amino_df.csv', index=False)
    print(f"Successfully saved final feature set to chi_amino_df.csv with {len(final_col_names)} features.")
    
    return final_ops

def main():
    parser = argparse.ArgumentParser(description='Refactored Chi-Squared AMINO Feature Selection')
    parser.add_argument('--dataset', type=str, default='sample_CA_post_variance.csv', help='Input CSV dataset')
    parser.add_argument('--target', type=str, default='class', help='Target column name')
    parser.add_argument('--max_outputs', type=int, default=10, help='Max outputs for AMINO')
    parser.add_argument('--bins', type=int, default=30, help='Number of bins for AMINO')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} not found.")
        return

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    
    run_chi_amino_workflow(
        df=df,
        target_col=args.target,
        max_outputs_amino=args.max_outputs,
        bins_amino=args.bins
    )

if __name__ == "__main__":
    main()
