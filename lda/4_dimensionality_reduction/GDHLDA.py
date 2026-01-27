### STEP 0. Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def run_gdhlda(
    input_csv='mpso.csv',
    output_csv='GDHLDA.csv',
    nDataPoints=754,
    num_class=3,
    num_descriptor=7,
    num_eigenvector=2,
    descriptor_list=['res159.439', 'res245.369', 'res64.137', 'res199.471', 'res78.450', 'res242.340', 'res77.293'],
    learning_rate=0.0001,
    num_iteration=10000,
    stop_crit=500
):
    ### STEP 1. Load input data
    df = pd.read_csv(input_csv)
    df

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
    for cl in range(1, num_class+1):
        mean_vectors.append(np.mean(X[y==cl], axis=0)) # X[y==cl] : Boolean indexing/slicing - select lists (rows) in X that satisfy the condition y==cl 
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
    for i, mean_vec in enumerate(mean_vectors): 
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(num_descriptor,1) # make column vector
        overall_mean = overall_mean.reshape(num_descriptor,1) # make column vector
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
        np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv), eig_vals[i] * eigv, decimal=3, err_msg='', verbose=True)
    print('ok')

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
    W = W.real

    ##### Step 8. Perform the Stiefel gradient decent algorithm for HLDA
    print("From HLDA")
    print("W", W.real)
    print("WTW", (W.T).dot(W))

    objFuncList = []
    objFuncDiffSum = 0
    for niter in range(num_iteration):
        if niter != 0:
            prevObjFunc = objFunc
        # compute objective function
        objFunc = 0
        for g in range(1,num_class):
            ng = X[y==g,:].shape[0]
            mvg = mean_vectors[g-1].reshape(num_descriptor,1)
            for h in range(g+1,num_class+1):
                nh = X[y==h,:].shape[0]
                mvh = mean_vectors[h-1].reshape(num_descriptor,1)
                WTSWW = ((W.T).dot(S_W)).dot(W)
                Bgh = (mvg-mvh).dot((mvg-mvh).T)
                WTBghW = ((W.T).dot(Bgh)).dot(W)
                objFunc += ng*nh*(WTSWW.trace())/(WTBghW.trace())
        objFuncList.append(objFunc)
        
        if niter != 0:
            if abs(objFunc-prevObjFunc) == 0:
                objFuncDiffSum += 1
                if objFuncDiffSum == stop_crit:
                    break
        
        # Stiefel gradient decent algorithm
        dJ1 = np.zeros((num_descriptor, num_eigenvector))
        for p in range(1,num_class): # p here is k in the literature, num_class here is K in the literature
            npp = X[y==p,:].shape[0]
            mvp = mean_vectors[p-1].reshape(num_descriptor,1)
            for q in range(p+1,num_class+1): # q here is l in the literature
                nqq = X[y==q,:].shape[0]
                mvq = mean_vectors[q-1].reshape(num_descriptor,1)
                Bpq = (mvp-mvq).dot((mvp-mvq).T)
                WTBpqW = ((W.T).dot(Bpq)).dot(W)
                gradD1 = ((2*npp*nqq)/(WTBpqW.trace()))*(S_W.dot(W))
                gradD1 = gradD1.astype('float64')
                dJ1 += gradD1

        dJ2 = np.zeros((num_descriptor, num_eigenvector))
        for r in range(1,num_class): # r here is k in the literature, num_class here is K in the literature
            nrr = X[y==r,:].shape[0]
            mvr = mean_vectors[r-1].reshape(num_descriptor,1)
            for s in range(r+1,num_class+1): # s here is l in the literature
                nss = X[y==s,:].shape[0]
                mvs = mean_vectors[s-1].reshape(num_descriptor,1)
                WTSWW = ((W.T).dot(S_W)).dot(W)
                Brs = (mvr-mvs).dot((mvr-mvs).T)
                WTBrsW = ((W.T).dot(Brs)).dot(W)
                gradD2 = 2*nrr*nss*(WTSWW.trace())/((WTBrsW.trace())*(WTBrsW.trace()))*(Brs.dot(W))
                gradD2 = gradD2.astype('float64')
                dJ2 += gradD2

        dJ = dJ1-dJ2
        dJ = normalize(dJ.astype('float64'), axis=0, norm='l2')
        W -= learning_rate*(dJ-(W.dot(dJ.T)).dot(W))
        W = W.astype('float64')
        if niter % 20 == 0 and niter != 0:
            U, D, VT = np.linalg.svd(W)
            D = np.ones((D.shape))
            S = np.zeros((W.shape[0], W.shape[1]))
            S[:W.shape[1], :W.shape[1]] = np.diag(D)
            W = U.dot(S.dot(VT))
            W = W.astype('float64')
            WTW = (W.T).dot(W)
            print("----------Iteration", niter, "----------")
            print("W", W)
            print("dJ", dJ)
            print("WTW", WTW)
            assert (WTW.shape[0] == WTW.shape[1]) and np.allclose(WTW, np.eye(WTW.shape[0]))

    ### STEP 9. Transform the samples onto the new subspace
    X_ldaz = X.dot(W.real) 
    y = np.concatenate([np.zeros(nDataPoints),np.ones(nDataPoints),np.ones(nDataPoints)+1])

    np.savetxt(output_csv, X_ldaz, delimiter=",", fmt="%.5f", header="LD1, LD2", comments="")
    df2 = pd.read_csv(output_csv)
    df2['class'] = y
    df2.to_csv(output_csv, encoding='utf-8', index=False)
    
    return X_ldaz, W

if __name__ == "__main__":
    run_gdhlda()