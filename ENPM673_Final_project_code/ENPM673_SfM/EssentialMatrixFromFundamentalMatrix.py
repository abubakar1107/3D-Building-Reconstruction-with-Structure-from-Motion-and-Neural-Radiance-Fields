import numpy as np

def get_essential_matrix(K, F):
   
    # Compute the essential matrix using the relationship E = K'FK
    E = K.T @ F @ K

    # Perform Singular Value Decomposition (SVD)
    U, S, VT = np.linalg.svd(E)

    # Correct the singular values to be [1, 1, 0]
    S_corrected = np.array([1, 1, 0])

    # Reconstruct the essential matrix with the corrected singular values
    E_corrected = U @ np.diag(S_corrected) @ VT

    return E_corrected

