import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def run_zhlda(
    data,
    num_eigenvector=2,
    learning_rate=0.0001,
    num_iteration=10000,
    stop_crit=500,
    target_col='class',
    save_csv=False,
    output_csv='ZHLDA.csv'
):
    """
    Perform Zero-mean Heteroscedastic Linear Discriminant Analysis on input data.
    
    Args:
        data: DataFrame or iterator of DataFrames containing the input data
        num_eigenvector: Number of eigenvectors to use
        learning_rate: Learning rate for gradient descent
        num_iteration: Maximum number of iterations
        stop_crit: Stopping criterion for convergence
        target_col: Name of the target column (default: 'class')
        save_csv: If True, save results to CSV file (default: False)
        output_csv: Output CSV filename (only used if save_csv=True)
    
    Yields:
        DataFrame with ZHLDA-transformed features and class labels
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
    
    # Infer class information
    num_class = df[target_col].nunique()
    class_counts = df[target_col].value_counts()
    nDataPoints = class_counts.iloc[0]  # Assumes balanced classes
    
    print(f"Inferred parameters:")
    print(f"  - Number of features: {num_descriptor}")
    print(f"  - Number of classes: {num_class}")
    print(f"  - Data points per class: {nDataPoints}")
    print(f"  - Feature columns: {descriptor_list}")
    ### STEP 3. Separate data and generate labels
    X = df[descriptor_list].values
    X = X.astype(np.float64)
    y = df[target_col].values
    print(X)
    print(y)

    ### STEP 4. Compute the d-dimensional mean vectors
    ### Here, we calculate #num_class column vectors, each of which contains #num_descriptor elements (means)
    np.set_printoptions(precision=4)
    # Get unique classes and handle arbitrary labels
    unique_classes = np.unique(y)
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    mean_vectors = []
    for cl in unique_classes:
        mean_vectors.append(np.mean(X[y == cl], axis=0))
        print(f'Mean Vector class {cl}: {mean_vectors[class_to_idx[cl]]}')

    ### STEP 5. Compute the scatter matrices
    ### 5-1. Within-class scatter matrix SW
    S_W = np.zeros((num_descriptor,num_descriptor))
    for cl, mv in zip(unique_classes, mean_vectors):
        class_sc_mat = np.zeros((num_descriptor,num_descriptor))
        class_data = X[y == cl]
        for row in class_data:
            row_vec = row.reshape(num_descriptor,1)
            mv_vec = mv.reshape(num_descriptor,1)
            class_sc_mat += (row_vec - mv_vec).dot((row_vec - mv_vec).T)
        S_W += class_sc_mat  

    print('within-class Scatter Matrix:\n', S_W)

    ### 5-2. Between-class scatter matrix SB
    overall_mean = np.mean(X, axis=0) 
    S_B = np.zeros((num_descriptor,num_descriptor))
    for i, cl in enumerate(unique_classes):
        class_data = X[y == cl]
        n = len(class_data)
        mean_vec = mean_vectors[i].reshape(num_descriptor,1) 
        overall_mean_vec = overall_mean.reshape(num_descriptor,1) 
        S_B += n * (mean_vec - overall_mean_vec).dot((mean_vec - overall_mean_vec).T)

    print('between-class Scatter Matrix:\n', S_B)

    # Validate num_eigenvector against theoretical limit
    max_possible_ld = min(num_descriptor, len(unique_classes) - 1)
    if num_eigenvector > max_possible_ld:
        print(f"Warning: Requested {num_eigenvector} LDs, but max is {max_possible_ld}. Adjusting.")
        num_eigenvector = max_possible_ld
    
    if num_eigenvector < 1:
        raise ValueError("Cannot perform ZHLDA: not enough classes or features.")
    
    # Solve generalized eigenvalue problem with regularization
    try:
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    except np.linalg.LinAlgError:
        # If S_W is singular, use pseudo-inverse
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
        
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(num_descriptor,1)
        print(f'\nEigenvector {i+1}: \n{eigvec_sc.real}')
        print(f'Eigenvalue {i+1}: {eig_vals[i].real:.2e}')

    for i in range(len(eig_vals)):
        eigv = eig_vecs[:,i].reshape(num_descriptor,1)
        np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv), eig_vals[i] * eigv, decimal=3, err_msg='', verbose=True)
    print('ok')

    ### STEP 7. Select linear discriminants for the new feature subspace
    ### 7-1. Sort the eigenvectors by decreasing eigenvalues
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])

    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        print(f'eigenvalue {i+1}: {(j[0]/eigv_sum).real:.2%}')

    W = np.concatenate([eig_pairs[i][1].reshape(num_descriptor,1) for i in range(num_eigenvector)], axis=1)
    print('Matrix W:\n', W.real)
    W = W.real

    ##### Step 8. Perform the Stiefel gradient decent algorithm for HLDA
    print("From ZHLDA")
    print("W", W.real)
    print("WTW", (W.T).dot(W))

    objFuncList = []
    objFuncDiffSum = 0
    for niter in range(num_iteration):
        if niter != 0:
            prevObjFunc = objFunc
            
        # Gradient descent objective function
        objFunc = 0
        for g_idx, g in enumerate(unique_classes[:-1]):  # Exclude last class
            ng = X[y == g].shape[0]
            mvg = mean_vectors[g_idx].reshape(num_descriptor, 1)
            
            for h_idx, h in enumerate(unique_classes[g_idx + 1:], start=g_idx + 1):
                nh = X[y == h].shape[0]
                mvh = mean_vectors[h_idx].reshape(num_descriptor, 1)
                
                WTSWW = ((W.T).dot(S_W)).dot(W)
                Bgh = (mvg - mvh).dot((mvg - mvh).T)
                WTBghW = ((W.T).dot(Bgh)).dot(W)
                
                # Avoid division by zero
                trace_bgh = WTBghW.trace()
                if trace_bgh > 1e-10:
                    objFunc += ng * nh * (WTSWW.trace()) / trace_bgh
        
        objFuncList.append(objFunc)
        
        if niter != 0:
            if abs(objFunc-prevObjFunc) == 0:
                objFuncDiffSum += 1
                if objFuncDiffSum == stop_crit:
                    break
        
        # Stiefel gradient descent algorithm
        dJ1 = np.zeros((num_descriptor, num_eigenvector))
        
        for p_idx, p in enumerate(unique_classes[:-1]):  # Exclude last class
            npp = X[y == p].shape[0]
            mvp = mean_vectors[p_idx].reshape(num_descriptor, 1)
            
            for q_idx, q in enumerate(unique_classes[p_idx + 1:], start=p_idx + 1):
                nqq = X[y == q].shape[0]
                mvq = mean_vectors[q_idx].reshape(num_descriptor, 1)
                
                Bpq = (mvp - mvq).dot((mvp - mvq).T)
                WTBpqW = ((W.T).dot(Bpq)).dot(W)
                
                trace_bpq = WTBpqW.trace()
                if trace_bpq > 1e-10:
                    gradD1 = ((2 * npp * nqq) / trace_bpq) * (S_W.dot(W))
                    gradD1 = gradD1.astype('float64')
                    dJ1 += gradD1

        dJ2 = np.zeros((num_descriptor, num_eigenvector))
        
        for r_idx, r in enumerate(unique_classes[:-1]):  # Exclude last class
            nrr = X[y == r].shape[0]
            mvr = mean_vectors[r_idx].reshape(num_descriptor, 1)
            
            for s_idx, s in enumerate(unique_classes[r_idx + 1:], start=r_idx + 1):
                nss = X[y == s].shape[0]
                mvs = mean_vectors[s_idx].reshape(num_descriptor, 1)
                
                WTSWW = ((W.T).dot(S_W)).dot(W)
                Brs = (mvr - mvs).dot((mvr - mvs).T)
                WTBrsW = ((W.T).dot(Brs)).dot(W)
                
                trace_brs = WTBrsW.trace()
                if trace_brs > 1e-10:
                    gradD2 = 2 * nrr * nss * (WTSWW.trace()) / (trace_brs * trace_brs) * (Brs.dot(W))
                    gradD2 = gradD2.astype('float64')
                    dJ2 += gradD2

        dJ = dJ1-dJ2
        dJ = normalize(dJ.astype('float64'), axis=0, norm='l2')
        W -= learning_rate*(dJ-(W.dot(dJ.T)).dot(W))
        W = W.astype('float64')
        if niter % 20 == 0 and niter != 0:
            # Re-orthogonalize using SVD
            U, D, VT = np.linalg.svd(W)
            # Keep only the top num_eigenvector singular values
            D_truncated = np.eye(W.shape[0], W.shape[1])
            D_truncated[:num_eigenvector, :num_eigenvector] = np.diag(np.ones(num_eigenvector))
            W = U.dot(D_truncated.dot(VT))
            W = W.astype('float64')
            
            WTW = (W.T).dot(W)
            print("----------Iteration", niter, "----------")
            print("W", W)
            print("dJ", dJ)
            print("WTW", WTW)
            
            # Check orthogonality
            if not np.allclose(WTW, np.eye(WTW.shape[0]), atol=1e-6):
                print("Warning: W is not perfectly orthogonal")

    ### STEP 9. Transform the samples onto the new subspace
    X_ldaz = X.dot(W.real) 
    y_output = df[target_col].values

    # Create result DataFrame with dynamic column names
    cols = [f'LD{i+1}' for i in range(num_eigenvector)]
    result_df = pd.DataFrame(X_ldaz, columns=cols)
    result_df[target_col] = y_output
    
    if save_csv:
        result_df.to_csv(output_csv, encoding='utf-8', index=False)
    
    yield result_df

if __name__ == "__main__":
    # Example usage:
    # df = pd.read_csv('mpso.csv')
    # result_iterator = run_zhlda(df, save_csv=True, output_csv='ZHLDA.csv')
    # for result_df in result_iterator:
    #     print(result_df.head())
    pass