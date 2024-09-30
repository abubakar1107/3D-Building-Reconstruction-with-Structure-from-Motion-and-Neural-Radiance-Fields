import numpy as np
from scipy.spatial.transform import Rotation 
import scipy.optimize as optimize

def get_rotation(Q, type_='q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def homogenize(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def get_quaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def projection_matrix(R, C, K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def pnp_loss(X0, x3D, pts, K):
    Q, C = X0[:4], X0[4:].reshape(-1, 1)
    R = get_rotation(Q)
    P = projection_matrix(R, C, K)
    
    Error = []
    for X, pt in zip(x3D, pts):
        p_1T, p_2T, p_3T = P
        p_1T, p_2T, p_3T = p_1T.reshape(1, -1), p_2T.reshape(1, -1), p_3T.reshape(1, -1)
        X = homogenize(X.reshape(1, -1)).reshape(-1, 1)
        
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X), p_3T.dot(X))
        v_proj = np.divide(p_2T.dot(X), p_3T.dot(X))
        
        E = np.square(v - v_proj) + np.square(u - u_proj)
        Error.append(E)

    sum_error = np.mean(np.array(Error).squeeze())
    return sum_error

def nonlinear_pnp(K, pts, x3D, R0, C0):
    Q = get_quaternion(R0)
    X0 = np.concatenate((Q, C0))

    optimized_params = optimize.least_squares(
        fun=pnp_loss,
        x0=X0,
        method="trf",
        args=[x3D, pts, K]
    )
    
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = get_rotation(Q)
    
    return R, C
