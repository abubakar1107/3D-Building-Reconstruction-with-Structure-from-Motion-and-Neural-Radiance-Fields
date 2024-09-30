import numpy as np

def disambiguate_camera_pose(rotation_sets, translation_sets, points_3d_sets):
  
    best_index = 0
    max_in_front = 0

    for i in range(len(rotation_sets)):
        R = rotation_sets[i]
        C = translation_sets[i].reshape(-1, 1)
        points_3d = points_3d_sets[i]
        
        # Normalize points to ensure homogeneity
        points_3d = points_3d / points_3d[:, 3].reshape(-1, 1)
        points_3d = points_3d[:, :3]

        num_in_front = count_points_in_front_of_camera(points_3d, R, C)
        if num_in_front > max_in_front:
            best_index = i
            max_in_front = num_in_front

    return rotation_sets[best_index], translation_sets[best_index], points_3d_sets[best_index]

def count_points_in_front_of_camera(points_3d, R, C):

    r3 = R[2, :]  # The third row of the rotation matrix
    count = 0
    for X in points_3d:
        X = X.reshape(-1, 1)
        z = r3 @ (X - C)
        if z > 0 and X[2, 0] > 0:  # Check both the cheirality condition and positive depth in camera coords
            count += 1
    return count
