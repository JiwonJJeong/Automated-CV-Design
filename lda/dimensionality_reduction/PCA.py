import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from data_access import METADATA_COLS

def run_pca(
    data,
    num_eigenvector=2,
    target_col='class',
    save_csv=False,
    output_csv='PCA.csv',
    svd_solver='auto',
    whiten=False,
    tol=0.0,
    max_iter=None
):
    """
    Perform PCA dimensionality reduction on input data.
    
    Args:
        data: DataFrame or iterator of DataFrames containing the input data
        num_eigenvector: Number of principal components to extract
        target_col: Name of the target column (default: 'class')
        save_csv: If True, save results to CSV file (default: False)
        output_csv: Output CSV filename (only used if save_csv=True)
    
    Yields:
        DataFrame with PCA-transformed features and class labels
    """
    # Handle iterator or single DataFrame
    if hasattr(data, '__iter__') and not isinstance(data, pd.DataFrame):
        # If it's an iterator, consume it into a single DataFrame
        df = pd.concat(data, ignore_index=True)
    else:
        df = data.copy()
    
    # Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # Infer parameters from data
    descriptor_list = [col for col in df.columns if col != target_col]
    num_descriptor = len(descriptor_list)
    
    if num_descriptor == 0:
        raise ValueError("No feature columns found in DataFrame (all columns are target column)")
    
    # 1. Infer parameters: Filter for numeric features only & ignore metadata
    descriptor_list = [
        col for col in df.columns 
        if col != target_col 
        and col not in METADATA_COLS 
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    num_descriptor = len(descriptor_list)
    
    if num_descriptor == 0:
        raise ValueError("No numeric feature columns found (after filtering target and metadata).")
    
    # 2. Extract Data (Strictly Numeric)
    X = df[descriptor_list].values.astype(np.float64)
    y = df[target_col].values

    # Validate num_eigenvector against available features
    max_possible_pc = min(num_descriptor, num_eigenvector)
    if num_eigenvector > max_possible_pc:
        print(f"Warning: Requested {num_eigenvector} PCs, but max is {max_possible_pc}. Adjusting.")
        num_eigenvector = max_possible_pc
    
    if num_eigenvector < 1:
        raise ValueError("Cannot perform PCA: no features available.")

    ### STEP 3. Perform PCA
    # Standard PCA does not accept max_iter or tol in __init__ 
    # unless using specific solvers like 'arpack'
    pca_params = {
        'n_components': num_eigenvector,
        'svd_solver': svd_solver,
        'whiten': whiten
    }
    
    # Only add tol/max_iter if using the arpack solver
    if svd_solver == 'arpack':
        pca_params['tol'] = tols
        pca_params['max_iter'] = max_iter

    pca = PCA(**pca_params)
    pca_X = pca.fit_transform(X)
    pca_X = pca.fit_transform(X)
    print('Shape before PCA: ', X.shape)
    print('Shape after PCA: ', pca_X.shape)

    # Create result DataFrame with dynamic column names
    cols = [f'PC{i+1}' for i in range(num_eigenvector)]
    pca_df = pd.DataFrame(data=pca_X, columns=cols)
    pca_df['class'] = y
    
    if save_csv:
        pca_df.to_csv(output_csv, index=False)

    ### STEP 4. Calculate variances (eigenvalues) and CVs (eigenvectors)
    print('Variances:', pca.explained_variance_)
    print('Variance ratios:', pca.explained_variance_ratio_)
    # print('CVs:', pca.components_)S
    
    return pca_df

if __name__ == "__main__":
    # Example usage:
    # df = pd.read_csv('mpso.csv')  # Output from feature selection
    # result_iterator = run_pca(df, save_csv=True, output_csv='PCA.csv')
    # for result_df in result_iterator:
    #     print(result_df.head())
    pass