import numpy as np

import numpy as np

def extract_camera_pose(E):
    
    # Singular Value Decomposition of the Essential Matrix
    U, _, VT = np.linalg.svd(E)
    
    # Define matrix W
    W = np.array([[0, -1, 0], 
                  [1,  0, 0], 
                  [0,  0, 1]])
    
    # Possible rotation matrices
    R1 = U @ W @ VT
    R2 = U @ W.T @ VT
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Possible translation vectors (up to scale)
    C1 = U[:, 2]
    C2 = -U[:, 2]

    # Four possible camera poses
    camera_poses = [(R1, C1), (R1, C2), (R2, C1), (R2, C2)]

    return [R1, R2], [C1, C2]


