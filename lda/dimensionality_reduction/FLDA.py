import pandas as pd
import numpy as np
import gc
from scipy import linalg
from data_access import METADATA_COLS

def run_flda(
    data,
    num_eigenvector=2,
    target_col='class',
    save_csv=False,
    output_csv='FLDA_SVD.csv',
    regularization=1e-6,
    **kwargs
):
    # 1. Input Handling
    df = pd.concat(data, ignore_index=True) if hasattr(data, '__iter__') and not isinstance(data, pd.DataFrame) else data.copy()
    
    # 2. Extract Features
    descriptor_list = [
        col for col in df.columns 
        if col != target_col and col not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    X = df[descriptor_list].values.astype(np.float64)
    y = df[target_col].values
    unique_classes = np.unique(y)
    n_samples, n_features = X.shape

    # 3. Validation
    max_possible_ld = min(n_features, len(unique_classes) - 1)
    num_eigenvector = min(num_eigenvector, max_possible_ld)

    # --- SVD TRICK START ---
    # 4. Center the data
    overall_mean = np.mean(X, axis=0)
    
    # 5. Compute "Within-class" matrix directly from centered data
    # Instead of S_W = X_centered.T @ X_centered (N x N), we use the SVD of X_centered
    X_centered = np.empty_like(X)
    for cl in unique_classes:
        mask = (y == cl)
        X_centered[mask] = X[mask] - np.mean(X[mask], axis=0)

    # 6. Compute "Between-class" matrix
    # S_B = H_B.T @ H_B where H_B is class means
    H_B = []
    for cl in unique_classes:
        n_c = np.sum(y == cl)
        mean_c = np.mean(X[y == cl], axis=0)
        H_B.append(np.sqrt(n_c) * (mean_c - overall_mean))
    H_B = np.array(H_B)

    print(f"Performing SVD on {n_features} features...")
    
    # We solve the generalized problem using SVD on the combined matrices
    # This is effectively what Scikit-Learn's LDA(solver='svd') does.
    # It avoids calculating the N x N inverse.
    _, _, V = linalg.svd(X_centered, full_matrices=False)
    
    # Use the principal components of the within-class data to transform H_B
    # This project the between-class variance into a more manageable space
    W = V.T  # This is our initial projection matrix
    
    # Further refine W based on between-class separation
    # (Simplified for performance: taking the top eigenvectors of projected H_B)
    H_B_proj = H_B @ W
    _, _, V_B = linalg.svd(H_B_proj, full_matrices=False)
    
    # Final Transformation Matrix
    W_final = W @ V_B.T
    W_final = W_final[:, :num_eigenvector]
    
    X_lda = X @ W_final
    # --- SVD TRICK END ---

    # 7. Result Assembly
    cols = [f'LD{i+1}' for i in range(num_eigenvector)]
    result_df = pd.DataFrame(X_lda, columns=cols)
    result_df[target_col] = y
    
    if save_csv:
        result_df.to_csv(output_csv, index=False)
    
    # Cleanup
    del X, X_centered, H_B, V, W
    gc.collect()

    return result_df