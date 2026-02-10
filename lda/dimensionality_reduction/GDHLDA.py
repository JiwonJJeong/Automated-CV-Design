import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import normalize
from data_access import METADATA_COLS

def run_gdhlda(
    data,
    num_eigenvector=2,
    learning_rate=0.0001,
    num_iteration=2000, # 10k is usually overkill with a good start
    stop_crit=50,
    target_col='class',
    save_csv=False,
    output_csv='GDHLDA.csv',
    convergence_threshold=1e-6,
    normalize_features=True,
    **kwargs
):
    # 1. Input Handling
    df = pd.concat(data, ignore_index=False) if hasattr(data, '__iter__') and not isinstance(data, pd.DataFrame) else data.copy()
    
    descriptor_list = [col for col in df.columns if col != target_col and col not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[col])]
    X = df[descriptor_list].values.astype(np.float64)
    y = df[target_col].values
    unique_classes = np.unique(y)
    num_descriptor = X.shape[1]

    if normalize_features:
        X = normalize(X, axis=0, norm='l2')

    # 2. Pre-calculate Statistics (Heteroscedastic S_W)
    class_stats = {}
    S_W = np.zeros((num_descriptor, num_descriptor))
    total_n = len(X)
    
    for cl in unique_classes:
        X_c = X[y == cl]
        n_c = len(X_c)
        mv_c = np.mean(X_c, axis=0).reshape(-1, 1)
        
        if n_c > 1:
            centered = X_c - mv_c.T
            # Heteroscedastic weighting logic
            cov_c = centered.T.dot(centered) + np.eye(num_descriptor) * 1e-7
            S_W += (n_c / total_n) * np.linalg.pinv(cov_c)
        
        class_stats[cl] = {'n': n_c, 'mean': mv_c}

    # Pre-calculate pairwise Between-class (Bgh) components
    pairs = []
    for i, g in enumerate(unique_classes[:-1]):
        for h in unique_classes[i+1:]:
            diff = class_stats[g]['mean'] - class_stats[h]['mean']
            pairs.append({
                'Bgh': diff.dot(diff.T), 
                'weight': class_stats[g]['n'] * class_stats[h]['n']
            })

    # 3. Warm Start W (Standard Eigen Problem)
    S_B_sum = sum(p['Bgh'] * p['weight'] for p in pairs)
    eig_vals, eig_vecs = np.linalg.eigh(S_B_sum) # Symmetric solver
    idx = np.argsort(np.abs(eig_vals))[::-1]
    sorted_eig_vals = np.abs(eig_vals[idx])
    
    if num_eigenvector is None:
        cumulative_variance = np.cumsum(sorted_eig_vals) / np.sum(sorted_eig_vals)
        num_eigenvector = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"GDHLDA Dynamic selection: {num_eigenvector} components capture 95% between-class scatter")
    else:
        num_eigenvector = min(num_eigenvector, num_descriptor)

    W = eig_vecs[:, idx][:, :num_eigenvector].real

    # 4. Stabilized Gradient Descent
    prev_obj = float('inf')
    strike_zone = 0
    eps = 1e-10 # Stability constant

    for niter in range(num_iteration):
        obj_func = 0
        dJ1 = np.zeros((num_descriptor, num_eigenvector))
        dJ2 = np.zeros((num_descriptor, num_eigenvector))
        
        SW_W = S_W.dot(W)
        tr_SWW = np.trace(W.T.dot(SW_W))

        for p in pairs:
            Bgh = p['Bgh']
            tr_B = np.trace(W.T.dot(Bgh).dot(W)) + eps
            
            # Objective
            obj_func += p['weight'] * tr_SWW / tr_B
            
            # Gradients
            dJ1 += (2 * p['weight'] / tr_B) * SW_W
            dJ2 += (2 * p['weight'] * tr_SWW / (tr_B**2)) * Bgh.dot(W)

        # 5. Gradient Clipping & Manifold Update
        dJ = dJ1 - dJ2
        norm = np.linalg.norm(dJ)
        if norm > 1.0: dJ /= norm # Cap the gradient
        
        grad = dJ - W.dot(dJ.T).dot(W)
        W -= learning_rate * grad

        # Convergence
        if abs(prev_obj - obj_func) < convergence_threshold:
            strike_zone += 1
            if strike_zone >= stop_crit: break
        else: strike_zone = 0
        prev_obj = obj_func

        if niter % 25 == 0:
            W, _ = np.linalg.qr(W) # Fast re-orthogonalization

    # 6. Result Assembly
    result_df = pd.DataFrame(X.dot(W), columns=[f'LD{i+1}' for i in range(num_eigenvector)], index=df.index)
    result_df[target_col] = df[target_col].values
    
    # Preserve metadata attributes (selected_features) for the leaderboard
    if hasattr(df, 'attrs'):
        result_df.attrs.update(df.attrs)
    
    if save_csv: result_df.to_csv(output_csv, index=False)
    return result_df