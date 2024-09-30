import numpy as np
import time
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def bundle_adjustment(camera_params, points_3d, points_2d, visibility_matrix, K):
    """
    Perform bundle adjustment to refine camera poses and 3D point coordinates.
    """
    num_cameras = camera_params.shape[0]
    num_points = points_3d.shape[0]
    num_observations = int(np.sum(visibility_matrix))

    m = 2 * num_observations
    n = 6 * num_cameras + 3 * num_points
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(num_observations)
    camera_indices, point_indices = visibility_to_indices(visibility_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, num_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, num_cameras * 6 + point_indices * 3 + s] = 1

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    t0 = time.time()
    res = least_squares(
        fun=bundle_adjustment_func, 
        x0=x0, 
        jac_sparsity=A, 
        verbose=2, 
        x_scale='jac', 
        ftol=1e-4, 
        method='trf', 
        args=(camera_indices, point_indices, points_2d.ravel(), K, num_cameras, num_points)
    )
    t1 = time.time()

    optimized_params = res.x
    optimized_camera_params = optimized_params[:num_cameras * 6].reshape((num_cameras, 6))
    optimized_points_3d = optimized_params[num_cameras * 6:].reshape((num_points, 3))

    print("Time required for Bundle Adjustment: ", t1 - t0, " seconds")

    return optimized_camera_params, optimized_points_3d


def project(points, camera_params, K):
    """
    Project 3D points into 2D using camera parameters.
    """
    projections = []
    for point, param in zip(points, camera_params):
        R = Rot.from_euler('xyz', param[:3]).as_matrix()
        t = param[3:]
        P = K @ np.hstack((R, t[:, np.newaxis]))
        point_homogeneous = np.hstack([point, 1])
        projection = P @ point_homogeneous
        projections.append(projection[:2] / projection[2])
    return np.vstack(projections)  # Stack into a single array for easy comparison

def bundle_adjustment_func(params, camera_indices, point_indices, points_2d, K, num_cameras, num_points):
    """
    Compute reprojection errors for bundle adjustment.
    """
    camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
    points_3d = params[num_cameras * 6:].reshape((num_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)

    # Ensure points_2d is reshaped properly for subtraction
    points_2d = points_2d.reshape(-1, 2)

    return (points_proj - points_2d).ravel()


def visibility_to_indices(visibility_matrix):
    """
    Convert visibility matrix to camera and point indices for bundle adjustment.
    """
    camera_indices = []
    point_indices = []
    for i in range(visibility_matrix.shape[0]):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i, j] == 1:
                camera_indices.append(j)
                point_indices.append(i)
    return np.array(camera_indices), np.array(point_indices)

# Example usage and initialization
camera_params = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])  # Initial camera parameters
points_3d = np.array([[1, 2, 3], [4, 5, 6]])   # Initial 3D point coordinates
points_2d = np.array([[10, 20], [30, 40]])  # Corresponding 2D image points
visibility_matrix = np.array([[1, 0], [0, 1]])  # Matrix indicating which points are visible in which camera
K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])  # Camera intrinsic matrix

optimized_camera_params, optimized_points_3d = bundle_adjustment(camera_params, points_3d, points_2d, visibility_matrix, K)

print("Optimized camera parameters:")
print(optimized_camera_params)
print("Optimized 3D points:")
print(optimized_points_3d)
