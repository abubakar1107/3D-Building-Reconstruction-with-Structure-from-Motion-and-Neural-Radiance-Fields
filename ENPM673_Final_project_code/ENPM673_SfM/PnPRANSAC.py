import numpy as np
from LinearPnP import PnP

def ProjectionMatrix(R, C, K):
    C = np.reshape(C, (3, 1))        
    P = K @ np.hstack((R, -R @ C))
    return P

def PnPError(feature, X, R, C, K):
    u, v = feature
    X_homogeneous = np.hstack((X, 1))
    X_homogeneous = X_homogeneous.reshape(4, 1)
    C = C.reshape(-1, 1)
    P = ProjectionMatrix(R, C, K)

    u_proj = P[0] @ X_homogeneous / (P[2] @ X_homogeneous)
    v_proj = P[1] @ X_homogeneous / (P[2] @ X_homogeneous)

    x_proj = np.array([u_proj, v_proj]).reshape(1, -1)
    x = np.array([u, v]).reshape(1, -1)
    err = np.linalg.norm(x - x_proj)

    return err

def PnPRANSAC(K, features, x3D, iter=1000, thresh=5):
    inliers_thresh = 0
    R_best, t_best = None, None
    n_rows = x3D.shape[0]

    for _ in range(iter):
        rand_indices = np.random.choice(n_rows, size=6, replace=False)
        X_set, x_set = x3D[rand_indices], features[rand_indices]

        R, C = PnP(X_set, x_set, K)

        if R is not None:
            indices = [j for j in range(n_rows) if PnPError(features[j], x3D[j], R, C, K) < thresh]

            if len(indices) > inliers_thresh:
                inliers_thresh = len(indices)
                R_best, t_best = R, C

    return R_best, t_best
