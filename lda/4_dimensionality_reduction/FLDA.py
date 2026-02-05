import pandas as pd
import numpy as np
import gc

def run_flda(
    data,
    num_eigenvector=2,
    target_col='class',
    save_csv=False,
    output_csv='FLDA.csv',
    regularization=1e-6,
    solver='eig'
):
    # 1. Handle Input (DataFrame or Iterator)
    if hasattr(data, '__iter__') and not isinstance(data, pd.DataFrame):
        df = pd.concat(data, ignore_index=True)
    else:
        df = data.copy()
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    # 2. Extract Features and Targets (FIXED: Added missing variable assignment)
    descriptor_list = [col for col in df.columns if col != target_col]
    X = df[descriptor_list].values.astype(np.float64)
    y = df[target_col].values
    
    unique_classes = np.unique(y)
    num_class = len(unique_classes)
    num_descriptor = X.shape[1]

    # 3. Validation & Constraints
    # Theoretical limit for FLDA is min(features, classes - 1)
    max_possible_ld = min(num_descriptor, num_class - 1)
    if num_eigenvector > max_possible_ld:
        print(f"Warning: Requested {num_eigenvector} LDs, but max is {max_possible_ld}. Adjusting.")
        num_eigenvector = max_possible_ld
    
    if num_eigenvector < 1:
        raise ValueError("Cannot perform FLDA: not enough classes or features.")

    # 4. Compute Mean Vectors
    mean_vectors = []
    for cl in unique_classes:
        mask = (y == cl)
        mean_vectors.append(np.mean(X[mask], axis=0))

    # 5. Scatter Matrices (SW and SB)
    S_W = np.zeros((num_descriptor, num_descriptor))
    S_B = np.zeros((num_descriptor, num_descriptor))
    overall_mean = np.mean(X, axis=0).reshape(-1, 1)

    for cl, mv in zip(unique_classes, mean_vectors):
        # Within-class scatter
        class_sc_mat = np.zeros((num_descriptor, num_descriptor))
        X_c = X[y == cl]
        mv_res = mv.reshape(-1, 1)
        for row in X_c:
            row = row.reshape(-1, 1)
            class_sc_mat += (row - mv_res).dot((row - mv_res).T)
        S_W += class_sc_mat
        
        # Between-class scatter
        n = X_c.shape[0]
        S_B += n * (mv_res - overall_mean).dot((mv_res - overall_mean).T)

    # 6. Solve Generalized Eigenvalue Problem
    # Regularization is crucial for singular matrices (e.g., in property tests)
    S_W += np.eye(num_descriptor) * regularization
    
    if solver == 'eig':
        # Use eig for non-symmetric matrices (inv(SW).dot(SB))
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    elif solver == 'eigh':
        # Use eigh for symmetric matrices (more stable but requires symmetric input)
        try:
            eig_vals, eig_vecs = np.linalg.eigh(np.linalg.inv(S_W).dot(S_B))
        except np.linalg.LinAlgError:
            print("Warning: eigh failed, falling back to eig")
            eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    else:
        raise ValueError(f"Unknown solver: {solver}. Use 'eig' or 'eigh'.")

    # 7. Select and Sort Components
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
    # Project and ensure result is real
    W = np.column_stack([eig_pairs[i][1] for i in range(num_eigenvector)])
    X_lda = X.dot(W).real

    # 8. Result Assembly
    cols = [f'LD{i+1}' for i in range(num_eigenvector)]
    result_df = pd.DataFrame(X_lda, columns=cols)
    result_df[target_col] = y
    
    if save_csv:
        result_df.to_csv(output_csv, index=False)
    
    yield result_df