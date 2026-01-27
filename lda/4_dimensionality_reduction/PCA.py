import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def run_pca(
    input_csv='mpso.csv',
    output_csv='PCA.csv',
    nDataPoints=754,
    num_eigenvector=2,
    descriptor_list=['res159.439', 'res245.369', 'res64.137', 'res199.471', 'res78.450', 'res242.340', 'res77.293']
):
    ### STEP 1. Load input data
    df = pd.read_csv(input_csv)

    ### STEP 1. Zero-mean the data
    np.set_printoptions(precision=8)
    for elem in descriptor_list:
        print('Mean for ', elem, ': ', df[elem].mean())
        df[elem] = df[elem] - df[elem].mean()

    ### STEP 2. Separate data and generate labels
    X = df.iloc[:,:len(descriptor_list)].values
    X = X.astype(np.float64)
    y = np.concatenate([np.zeros(nDataPoints),np.ones(nDataPoints),np.ones(nDataPoints)+1])
    print(X)

    ### STEP 3. Perform PCA
    pca = PCA(n_components=num_eigenvector)
    pca_X = pca.fit_transform(X)
    print('Shape before PCA: ', X.shape)
    print('Shape after PCA: ', pca_X.shape)

    pca_df = pd.DataFrame(data=pca_X, columns=['PC1', 'PC2'])
    pca_df['class'] = y
    pca_df.to_csv(output_csv, index=False)

    ### STEP 4. Calculate variances (eigenvalues) and CVs (eigenvectors)
    print('Variances:', pca.explained_variance_)
    print('Variance ratios:', pca.explained_variance_ratio_)
    print('CVs:', pca.components_)
    
    return pca_X, pca.components_

if __name__ == "__main__":
    run_pca()