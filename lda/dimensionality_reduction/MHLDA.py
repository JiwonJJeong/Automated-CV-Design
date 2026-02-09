import pandas as pd
import numpy as np
from scipy import linalg
from data_access import METADATA_COLS

def run_mhlda(
    data,
    num_eigenvector=2,
    target_col='class',
    save_csv=False,
    output_csv='MHLDA.csv',
    regularization=1e-4, # Slightly higher default for better stability
    **kwargs
):
    # 1. Handle Input
    df = pd.concat(data, ignore_index=True) if hasattr(data, '__iter__') and not isinstance(data, pd.DataFrame) else data.copy()
    
    # 2. Extract Numeric Features
    descriptor_list = [
        col for col in df.columns 
        if col != target_col and col not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    X = df[descriptor_list].values.astype(np.float64)
    y = df[target_col].values
    unique_classes = np.unique(y)
    n_features = X.shape[1]

    # 3. Compute Heteroscedastic Scatter Matrices
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))
    overall_mean = np.mean(X, axis=0).reshape(-1, 1)
    total_samples = len(X)

    for cl in unique_classes:
        X_c = X[y == cl]
        n_c = len(X_c)
        if n_c < 2: continue
        
        mean_c = np.mean(X_c, axis=0).reshape(-1, 1)
        
        # Within-class (Heteroscedastic: weighting by inverse covariance)
        centered = X_c - mean_c.T
        cov_c = (centered.T @ centered) + np.eye(n_features) * regularization
        
        # MHLDA specific: Weighted inverse covariance sum
        try:
            S_W += (n_c / total_samples) * np.linalg.inv(cov_c)
        except np.linalg.LinAlgError:
            S_W += (n_c / total_samples) * np.linalg.pinv(cov_c)
            
        # Between-class
        diff = mean_c - overall_mean
        S_B += n_c * (diff @ diff.T)

    # 4. Solve Generalized Eigenvalue Problem: S_B * v = lambda * S_W * v
    # This is MUCH more stable than inv(S_W).dot(S_B)
    try:
        # Add a tiny bit of regularization to S_W to ensure it's positive definite
        S_W += np.eye(n_features) * 1e-9
        
        # We use eigh because both S_B and S_W are symmetric
        eig_vals, eig_vecs = linalg.eigh(S_B, S_W)
        
        # Sort descending
        idx = np.argsort(eig_vals)[::-1]
        W = eig_vecs[:, idx][:, :num_eigenvector].real
    except Exception as e:
        print(f"MHLDA Solver Error: {e}. Falling back to standard SVD.")
        # Fallback to simple projection if the generalized problem is ill-conditioned
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        W = Vh.T[:, :num_eigenvector]

    # 5. Project and Return
    X_lda = X @ W
    result_df = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(X_lda.shape[1])])
    result_df[target_col] = y
    
    if save_csv:
        result_df.to_csv(output_csv, index=False)
        
    return result_df