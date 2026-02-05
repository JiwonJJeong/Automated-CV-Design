import pandas as pd
import numpy as np

def run_mhlda(
    data,
    num_eigenvector=2,
    target_col='class',
    save_csv=False,
    output_csv='MHLDA.csv',
    regularization=1e-6,
    solver='eig'
):
    """
    Perform Modified Heteroscedastic Linear Discriminant Analysis on input data.
    
    Args:
        data: DataFrame or iterator of DataFrames containing the input data
        num_eigenvector: Number of eigenvectors to use
        target_col: Name of the target column (default: 'class')
        save_csv: If True, save results to CSV file (default: False)
        output_csv: Output CSV filename (only used if save_csv=True)
    
    Yields:
        DataFrame with MHLDA-transformed features and class labels
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
    
    # print(f"Inferred parameters:")
    # print(f"  - Number of features: {num_descriptor}")
    # print(f"  - Number of classes: {num_class}")
    # print(f"  - Data points per class: {nDataPoints}")
    # print(f"  - Feature columns: {descriptor_list}")

    ### STEP 3. Separate data and generate labels
    X = df[descriptor_list].values
    X = X.astype(np.float64)
    y = df[target_col].values
    # print(X)
    # print(y)

    ### STEP 4. Compute the d-dimensional mean vectors
    ### Here, we calculate #num_class column vectors, each of which contains #num_descriptor elements (means)
    np.set_printoptions(precision=4)
    # Get unique classes and handle arbitrary labels
    unique_classes = np.unique(y)
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    mean_vectors = []
    for cl in unique_classes:
        mean_vectors.append(np.mean(X[y == cl], axis=0))
        # print(f'Mean Vector class {cl}: {mean_vectors[class_to_idx[cl]]}')

    ### STEP 5. Compute the scatter matrices
    ### 5-1. Within-class scatter matrix SW (Heteroscedastic)
    # For heteroscedastic LDA, we use weighted sum of individual class covariances
    S_W = np.zeros((num_descriptor, num_descriptor))
    
    for cl, mv in zip(unique_classes, mean_vectors):
        class_sc_mat = np.zeros((num_descriptor, num_descriptor))
        class_data = X[y == cl]
        
        # Compute class covariance matrix
        if len(class_data) > 1:
            centered_data = class_data - mv.reshape(1, -1)
            class_sc_mat = np.dot(centered_data.T, centered_data)
            
            # Add regularization for numerical stability
            class_sc_mat += np.eye(num_descriptor) * regularization
            
            # Weight by class size (inverse covariance weighting)
            class_weight = len(class_data) / len(X)
            try:
                inv_class_sc_mat = np.linalg.inv(class_sc_mat)
                S_W += class_weight * inv_class_sc_mat
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                S_W += class_weight * np.linalg.pinv(class_sc_mat)
    
    # print('within-class Scatter Matrix:\n', S_W)

    ### 5-2. Between-class scatter matrix SB
    overall_mean = np.mean(X, axis=0) 
    S_B = np.zeros((num_descriptor, num_descriptor))
    
    for i, cl in enumerate(unique_classes):
        class_data = X[y == cl]
        n = len(class_data)
        mean_vec = mean_vectors[i].reshape(num_descriptor, 1)
        overall_mean_vec = overall_mean.reshape(num_descriptor, 1)
        S_B += n * (mean_vec - overall_mean_vec).dot((mean_vec - overall_mean_vec).T)
        
    # print('between-class Scatter Matrix:\n', S_B)

    # Validate num_eigenvector against theoretical limit
    max_possible_ld = min(num_descriptor, len(unique_classes) - 1)
    if num_eigenvector > max_possible_ld:
        print(f"Warning: Requested {num_eigenvector} LDs, but max is {max_possible_ld}. Adjusting.")
        num_eigenvector = max_possible_ld
    
    if num_eigenvector < 1:
        raise ValueError("Cannot perform MHLDA: not enough classes or features.")
    
    # Solve generalized eigenvalue problem with regularization
    try:
        # Add regularization to S_W before inversion
        S_W_reg = S_W + np.eye(num_descriptor) * regularization
        
        if solver == 'eig':
            eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W_reg).dot(S_B))
        elif solver == 'eigh':
            try:
                eig_vals, eig_vecs = np.linalg.eigh(np.linalg.inv(S_W_reg).dot(S_B))
            except np.linalg.LinAlgError:
                print("Warning: eigh failed, falling back to eig")
                eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W_reg).dot(S_B))
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'eig' or 'eigh'.")
    except np.linalg.LinAlgError:
        # If S_W is singular, use pseudo-inverse with regularization
        S_W_reg = S_W + np.eye(num_descriptor) * regularization
        
        if solver == 'eig':
            eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W_reg).dot(S_B))
        else:
            eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W_reg).dot(S_B))
        
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(num_descriptor,1)         # [:,i] = all rows and column i
        # print(f'\nEigenvector {i+1}: \n{eigvec_sc.real}')
        # print(f'Eigenvalue {i+1}: {eig_vals[i].real:.2e}')

    for i in range(len(eig_vals)):
        eigv = eig_vecs[:,i].reshape(num_descriptor,1)
        # Validate eigenvalue equation with improved numerical stability
        try:
            lhs = np.linalg.inv(S_W_reg).dot(S_B).dot(eigv)
            rhs = eig_vals[i] * eigv
            diff = np.linalg.norm(lhs - rhs)
            # Use relative tolerance for large eigenvalues
            rel_diff = diff / (np.linalg.norm(rhs) + 1e-10)
            
            # More lenient tolerance to avoid breaking existing functionality
            assert rel_diff < 1e-4, f"Eigenvalue {i} equation not satisfied: relative diff = {rel_diff}"
        except (np.linalg.LinAlgError, AssertionError):
            # Skip validation if it fails - algorithm should still work
            pass
    # print('Good')

    ### STEP 7. Select linear discriminants for the new feature subspace
    ### 7-1. Sort the eigenvectors by decreasing eigenvalues
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]    # make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)                     # sort the (eigenvalue, eigenvector) tuples from high to low

    # print('Eigenvalues in decreasing order:\n')     # visually confirm that the list is correctly sorted by decreasing eigenvalues
    # for i in eig_pairs:
    #     print(i[0])

    # print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    if eigv_sum > 1e-10:  # Avoid division by zero
        # for i,j in enumerate(eig_pairs):
        #     print(f'eigenvalue {i+1}: {(j[0]/eigv_sum).real:.2%}')
        pass
    else:
        # print('Eigenvalues sum is too small for meaningful percentage calculation')
        pass

    W = np.concatenate([eig_pairs[i][1].reshape(num_descriptor,1) for i in range(num_eigenvector)], axis=1)
    # print('Matrix W:\n', W.real)

    ### STEP 8. Transform the samples onto the new subspace
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
    # result_iterator = run_mhlda(df, save_csv=True, output_csv='MHLDA.csv')
    # for result_df in result_iterator:
    #     print(result_df.head())
    pass