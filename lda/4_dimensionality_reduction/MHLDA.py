import pandas as pd
import numpy as np

def run_mhlda(
    input_csv='mpso.csv',
    output_csv='MHLDA.csv',
    nDataPoints=754,
    num_class=3,
    num_descriptor=7,
    num_eigenvector=2,
    descriptor_list=['res159.439', 'res245.369', 'res64.137', 'res199.471', 'res78.450', 'res242.340', 'res77.293']
):
    ### STEP 1. Load input data
    df = pd.read_csv(input_csv)

    ### STEP 2. Zero-mean the data
    np.set_printoptions(precision=8)
    for elem in descriptor_list:
        print('Mean for ', elem, ': ', df[elem].mean())
        df[elem] = df[elem] - df[elem].mean()

    ### STEP 3. Separate data and generate labels
    X = df.iloc[:,:num_descriptor].values
    X = X.astype(np.float64)
    y = np.concatenate([np.zeros(nDataPoints)+1,np.ones(nDataPoints)+1,np.ones(nDataPoints)+2])
    print(X)
    print(y)

    ### STEP 4. Compute the d-dimensional mean vectors
    ### Here, we calculate #num_class column vectors, each of which contains #num_descriptor elements (means)
    np.set_printoptions(precision=4)
    mean_vectors = []
    for cl in range(1,num_class+1):
        mean_vectors.append(np.mean(X[y==cl], axis=0))                
        print(f'Mean Vector class {cl}: {mean_vectors[cl-1]}')

    ### STEP 5. Compute the scatter matrices
    ### 5-1. Within-class scatter matrix SW
    S_W = np.zeros((num_descriptor,num_descriptor))
    S_W_int = np.zeros((num_descriptor,num_descriptor))
    for cl,mv in zip(range(1,num_class+1), mean_vectors):
        class_sc_mat = np.zeros((num_descriptor,num_descriptor))
        for row in X[y==cl]:
            row, mv = row.reshape(num_descriptor,1), mv.reshape(num_descriptor,1)   # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W_int += np.linalg.inv(class_sc_mat)                                      # sum class scatter matrices
    S_W = np.linalg.inv(S_W_int)

    print('within-class Scatter Matrix:\n', S_W)

    ### 5-2. Between-class scatter matrix SB
    overall_mean = np.mean(X, axis=0)                               
    S_B = np.zeros((num_descriptor,num_descriptor))
    for i,mean_vec in enumerate(mean_vectors):                      
        n = X[y==i+1,:].shape[0]                                    
        mean_vec = mean_vec.reshape(num_descriptor,1)               
        overall_mean = overall_mean.reshape(num_descriptor,1)       
        S_B += n*(mean_vec-overall_mean).dot((mean_vec-overall_mean).T)
        
    print('between-class Scatter Matrix:\n', S_B)

    ### STEP 6. Solve the generalized eigenvalue problem for the matrix SW^-1.SB
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(num_descriptor,1)         # [:,i] = all rows and column i
        print(f'\nEigenvector {i+1}: \n{eigvec_sc.real}')
        print(f'Eigenvalue {i+1}: {eig_vals[i].real:.2e}')

    for i in range(len(eig_vals)):
        eigv = eig_vecs[:,i].reshape(num_descriptor,1)
        np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv), eig_vals[i] * eigv, decimal=6, err_msg='', verbose=True)
    print('Good')

    ### STEP 7. Select linear discriminants for the new feature subspace
    ### 7-1. Sort the eigenvectors by decreasing eigenvalues
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]    # make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)                     # sort the (eigenvalue, eigenvector) tuples from high to low

    print('Eigenvalues in decreasing order:\n')     # visually confirm that the list is correctly sorted by decreasing eigenvalues
    for i in eig_pairs:
        print(i[0])

    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        print(f'eigenvalue {i+1}: {(j[0]/eigv_sum).real:.2%}')

    W = np.concatenate([eig_pairs[i][1].reshape(num_descriptor,1) for i in range(num_eigenvector)], axis=1)
    print('Matrix W:\n', W.real)

    ### STEP 8. Transform the samples onto the new subspace
    X_ldaz = X.dot(W.real) 
    y = np.concatenate([np.zeros(nDataPoints),np.ones(nDataPoints),np.ones(nDataPoints)+1])

    np.savetxt(output_csv, X_ldaz, delimiter=",", fmt="%.5f", header="LD1, LD2", comments="")
    df2 = pd.read_csv(output_csv)
    df2['class'] = y
    df2.to_csv(output_csv, encoding='utf-8', index=False)

    return X_ldaz, W

if __name__ == "__main__":
    run_mhlda()