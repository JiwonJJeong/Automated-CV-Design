import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import normalize
from data_access import METADATA_COLS

def run_zhlda(
    data,
    num_eigenvector=2,
    learning_rate=0.0001,
    num_iteration=2000, 
    stop_crit=50,
    target_col='class',
    save_csv=False,
    output_csv='ZHLDA.csv',
    convergence_threshold=1e-6,
    normalize_features=True,
    **kwargs
):
    # 1. Standardize Input
    if hasattr(data, '__iter__') and not isinstance(data, pd.DataFrame):
        df = pd.concat(data, ignore_index=False)
    else:
        df = data.copy()

    # 2. Extract Features (Numeric Only)
    descriptor_list = [
        col for col in df.columns 
        if col != target_col and col not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    X = df[descriptor_list].values.astype(np.float64)
    y = df[target_col].values
    unique_classes = np.unique(y)
    num_descriptor = X.shape[1]

    if normalize_features:
        X = normalize(X, axis=0, norm='l2')

    # 3. Pre-calculate Statistics (Speed Up)
    # We use a dictionary to map class names to their specific matrices
    class_stats = {}
    S_W = np.zeros((num_descriptor, num_descriptor))
    
    for cl in unique_classes:
        X_c = X[y == cl]
        n_c = X_c.shape[0]
        mv_c = np.mean(X_c, axis=0).reshape(-1, 1)
        
        # Within-class scatter for this class
        centered = X_c - mv_c.T
        sw_c = centered.T.dot(centered)
        S_W += sw_c
        
        class_stats[cl] = {'n': n_c, 'mean': mv_c}

    # Pre-calculate pairwise Between-class (Bgh) components
    # This replaces the nested loops inside the 10,000 iteration loop
    pairs = []
    for i, g in enumerate(unique_classes[:-1]):
        for h in unique_classes[i+1:]:
            diff = class_stats[g]['mean'] - class_stats[h]['mean']
            Bgh = diff.dot(diff.T)
            weight = class_stats[g]['n'] * class_stats[h]['n']
            pairs.append({'Bgh': Bgh, 'weight': weight})

    # 4. Initialize W (Eigenvalue Warm Start)
    # Instead of random, we start near the FLDA solution for instant convergence
    S_W_reg = S_W + np.eye(num_descriptor) * 1e-6
    S_B_sum = sum(p['Bgh'] * p['weight'] for p in pairs)
    
    # Use eigh for symmetric matrices - much more stable
    # Note: inv(S_W_reg).dot(S_B_sum) is not necessarily symmetric, but its eigenvalues are real/positive
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W_reg).dot(S_B_sum))
    idx = np.argsort(np.abs(eig_vals))[::-1]
    eig_vals_sorted = np.abs(eig_vals[idx])
    
    if num_eigenvector is None:
        cumulative_variance = np.cumsum(eig_vals_sorted) / np.sum(eig_vals_sorted)
        num_eigenvector = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"ZHLDA Dynamic selection: {num_eigenvector} components capture 95% generalized scatter")
    else:
        num_eigenvector = min(num_eigenvector, num_descriptor)

    W = eig_vecs[:, idx][:, :num_eigenvector].real

    # 5. Optimized Gradient Descent Loop
    prev_obj = float('inf')
    strike_zone = 0

    print(f"ZHLDA: Optimizing {num_descriptor} features...")

    for niter in range(num_iteration):
        obj_func = 0
        dJ1 = np.zeros((num_descriptor, num_eigenvector))
        dJ2 = np.zeros((num_descriptor, num_eigenvector))
        
        # Pre-calculate SW @ W once per iteration
        SW_W = S_W.dot(W)
        tr_SWW = np.trace(W.T.dot(SW_W))

        # inside the niter loop...
        for p in pairs:
            Bgh = p['Bgh']
            weight = p['weight']
            
            # Project Bgh into subspace
            tr_B = np.trace(W.T.dot(Bgh).dot(W))
            
            # 1. Add epsilon to denominator for numerical stability
            eps = 1e-8
            denom = tr_B + eps
            
            if tr_B > 1e-12:
                obj_func += weight * tr_SWW / denom
                
                # 2. Compute gradients with stabilized denominators
                grad1 = (2 * weight / denom) * SW_W
                grad2 = (2 * weight * tr_SWW / (denom**2)) * Bgh.dot(W)
                
                dJ1 += grad1
                dJ2 += grad2

        # 3. Gradient Clipping: Prevent the update from being too large
        dJ = dJ1 - dJ2
        
        # Calculate the norm of the gradient
        grad_norm = np.linalg.norm(dJ)
        if grad_norm > 1.0:
            dJ = dJ / grad_norm # Normalize the gradient magnitude
            
        # Project gradient onto Stiefel tangent space
        grad = dJ - W.dot(dJ.T).dot(W)
        
        # 4. Learning Rate Safety
        W -= learning_rate * grad

        # Convergence Check
        if abs(prev_obj - obj_func) < convergence_threshold:
            strike_zone += 1
            if strike_zone >= stop_crit:
                break
        else:
            strike_zone = 0
        
        prev_obj = obj_func

        # Keep W orthogonal using QR decomposition (Faster/more stable than SVD)
        if niter % 20 == 0:
            W, _ = np.linalg.qr(W)

    # 6. Final Assembly
    result_df = pd.DataFrame(X.dot(W), columns=[f'LD{i+1}' for i in range(num_eigenvector)], index=df.index)
    result_df[target_col] = df[target_col].values
    
    # Preserve metadata attributes (selected_features) for the leaderboard
    if hasattr(df, 'attrs'):
        result_df.attrs.update(df.attrs)

    if save_csv:
        result_df.to_csv(output_csv, index=False)

    # Cleanup
    del X, S_W, S_B_sum, pairs
    gc.collect()

    return result_df