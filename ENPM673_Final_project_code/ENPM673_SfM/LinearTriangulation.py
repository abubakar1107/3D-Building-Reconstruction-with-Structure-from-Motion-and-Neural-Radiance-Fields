import numpy as np

def linear_triangulation(K, C1, R1, C2, R2, x1, x2):
   
    # Identity matrix
    I = np.identity(3)
    
    # Reshape translation vectors
    t1 = -R1 @ C1.reshape((3, 1))
    t2 = -R2 @ C2.reshape((3, 1))
    
    # Compute projection matrices for the two cameras
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    
    # Prepare to store the 3D points
    points_3d = []
    
    # Triangulate each point
    for i in range(x1.shape[0]):
        x, y = x1[i, 0], x1[i, 1]
        xp, yp = x2[i, 0], x2[i, 1]

        # Create the matrix A for the homogenous equation system Ax = 0
        A = np.vstack([
            x * P1[2, :] - P1[0, :],
            y * P1[2, :] - P1[1, :],
            xp * P2[2, :] - P2[0, :],
            yp * P2[2, :] - P2[1, :]
        ])
        
        # Solve the equation using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[-1]  # Normalize to make the last element 1
        
        points_3d.append(X)
    
    return np.array(points_3d)

