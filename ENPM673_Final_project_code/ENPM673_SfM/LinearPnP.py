import numpy as np

def ProjectionMatrix(R, C, K):
    C = np.reshape(C, (3, 1))
    I = np.identity(3)
    P = K @ R @ np.concatenate((I, -C), axis=1)
    return P

def homo(pts):
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)

def reprojectionErrorPnP(x3D, pts, K, R, C):
    P = ProjectionMatrix(R, C, K)
    X_homogeneous = homo(x3D)
    pts_homogeneous = homo(pts)
    X_homogeneous = X_homogeneous.reshape(-1, 1, 4)
    pts_homogeneous = pts_homogeneous.reshape(-1, 1, 3)
    P = P.reshape(1, 3, 4)
    X_proj = np.divide(P @ X_homogeneous.transpose(0, 2, 1), P[:, 2].reshape(1, 1, 4))
    u_proj = X_proj[:, :, 0] / X_proj[:, :, 2]
    v_proj = X_proj[:, :, 1] / X_proj[:, :, 2]
    error = np.square(pts_homogeneous[:, :, 0] - u_proj) + np.square(pts_homogeneous[:, :, 1] - v_proj)
    mean_error = np.mean(error.squeeze())
    return mean_error

def PnP(X_set, x_set, K):
    N = X_set.shape[0]
    X_4 = homo(X_set)
    x_3 = homo(x_set)
    K_inv = np.linalg.inv(K)
    x_n = x_3 @ K_inv.T
    A = np.zeros((0, 9))
    for i in range(N):
        X = X_4[i].reshape((1, 4))
        u, v, _ = x_n[i]
        u_cross = np.array([[0, -1, v], [1, 0, -u], [-v, u, 0]])
        X_tilde = np.concatenate((np.concatenate((X, np.zeros((1, 4)), np.zeros((1, 4))), axis=1),
                                  np.concatenate((np.zeros((1, 4)), X, np.zeros((1, 4))), axis=1),
                                  np.concatenate((np.zeros((1, 4)), np.zeros((1, 4)), X), axis=1)), axis=0)
        a = u_cross @ X_tilde
        A = np.vstack((A, a)) if i > 0 else a
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    R = P[:, :3]
    U_r, D, V_rT = np.linalg.svd(R)
    R = U_r @ V_rT
    C = P[:, 3]
    C = - np.linalg.inv(R) @ C
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return R, C
