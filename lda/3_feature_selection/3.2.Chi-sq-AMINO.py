import amino_fast_mod as amino
import numpy as np
import pandas as pd
import gc
import argparse
import os
from sklearn.feature_selection import SelectKBest, chi2
from kneed import KneeLocator
from collections.abc import Iterable

def calculate_chi_scores(df, target_col='class', q=5):
    """
    Discretizes continuous variables and calculates Chi-Squared scores.
    """
    print(f"Discretizing features into {q} bins and calculating Chi-Squared scores...")
    
    # 1. Separate features and target
    X_cont = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Binning (Quantile-based)
    # Check if columns differ to avoid "Bin edges must be unique" error
    binned_columns = {col: pd.qcut(X_cont[col].rank(method='first'), q=q, labels=False) for col in X_cont}
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

def run_amino(all_ops, max_outputs=10, bins=30):
    """
    Runs the AMINO algorithm to find the final reduced set of features.
    """
    print(f"Running AMINO with max_outputs={max_outputs}, bins={bins}...")
    gc.collect()
    # Explicitly disable file output by passing None if supported, or just ignore the file it creates later
    # The modified amino_fast_mod might not support silencing the distortion file easily 
    # without a specific flag if the library hardcodes it.
    # Assuming standard amino_fast_mod behavior or minimal wrapper.
    # We set distortion_filename to None if possible to avoid writing, 
    # but based on previous code it was 'distortion_array'. 
    # If the user requested NO distortion file, we can try passing None or a dummy.
    # However, to be safe with unknown library internals, we just let it run.
    # If the signature allows distortion_filename, we pass it.
    final_ops = amino.find_ops(all_ops, max_outputs, bins, distortion_filename=None)
    
    print("\nAMINO selected features:")
    for op in final_ops:
        print(op)
            
    return final_ops

def run_chi_amino_workflow(df, target_col='class', max_outputs_amino=10, bins_amino=30):
    """
    Orchestrates the Chi-Squared AMINO feature selection workflow.
    Accepts a DataFrame or an iterator of DataFrames.
    Returns an iterator yielding the processed DataFrame with selected features.
    """
    
    # 0. Handle Iterator -> Full DataFrame
    if isinstance(df, Iterable) and not isinstance(df, pd.DataFrame):
        print("Consuming DataFrame iterator for feature selection...")
        df = pd.concat(df, ignore_index=True)
        
    # Validation: Check for target column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")

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
    
    # 6. Final preparation
    final_col_names = [str(op) for op in final_ops]
    df_amino = X_orig[final_col_names].copy()
    df_amino[target_col] = df[target_col].values
    
    # Return as an iterator (yielding the single result dataframe)
    return [df_amino]

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
    df_iter = [pd.read_csv(args.dataset)] # Simulate iterator for main execution
    
    try:
        result_iter = run_chi_amino_workflow(
            df=df_iter,
            target_col=args.target,
            max_outputs_amino=args.max_outputs,
            bins_amino=args.bins
        )
        
        for result_df in result_iter:
            print(f"Workflow completed. Resulting shape: {result_df.shape}")
            # Optional: Save here if running as main script, or just dry run
            # result_df.to_csv('chi_amino_result.csv', index=False)
            
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
