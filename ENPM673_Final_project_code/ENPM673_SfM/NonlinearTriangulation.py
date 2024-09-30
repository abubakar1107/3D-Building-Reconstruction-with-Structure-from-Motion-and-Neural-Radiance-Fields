import numpy as np
import scipy.optimize as optimize

def projection_matrix(R, C, K):
    """
    Compute the projection matrix given rotation matrix, translation vector, and camera intrinsic matrix.
    """
    T = -R @ C.reshape((3, 1))
    P = K @ np.hstack((R, T))
    return P

def nonlinear_triangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    """
    Refine the 3D point estimates by minimizing the reprojection error using non-linear least squares.
    """
    P1 = projection_matrix(R1, C1, K)
    P2 = projection_matrix(R2, C2, K)

    assert pts1.shape[0] == pts2.shape[0] == x3D.shape[0], "Mismatch in number of points across inputs."

    refined_x3D = []
    for i in range(len(x3D)):
        result = optimize.least_squares(
            fun=reprojection_loss,
            x0=x3D[i],
            method='trf',
            args=(pts1[i], pts2[i], P1, P2)
        )
        refined_x3D.append(result.x)

    return np.array(refined_x3D)

def reprojection_loss(X, pt1, pt2, P1, P2):
    """
    Calculate the reprojection loss for a given 3D point and two sets of camera parameters.
    """
    X_hom = np.hstack((X, [1])) if X.shape[0] == 3 else X
    proj1 = P1 @ X_hom
    proj2 = P2 @ X_hom
    u1_proj = proj1[0] / proj1[2]
    v1_proj = proj1[1] / proj1[2]
    u2_proj = proj2[0] / proj2[2]
    v2_proj = proj2[1] / proj2[2]
    error1 = (u1_proj - pt1[0])**2 + (v1_proj - pt1[1])**2
    error2 = (u2_proj - pt2[0])**2 + (v2_proj - pt2[1])**2

    return np.array([error1, error2])
