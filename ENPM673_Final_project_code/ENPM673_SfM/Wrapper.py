import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from EssentialMatrixFromFundamentalMatrix import get_essential_matrix
from ExtractCameraPose import extract_camera_pose
from DisambiguateCameraPose import disambiguate_camera_pose
from LinearTriangulation import linear_triangulation
from NonlinearTriangulation import nonlinear_triangulation

import matplotlib.pyplot as plt


def match_features(image1, image2):
    """ Match features between two images using SIFT and FLANN. """
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # FLANN parameters and matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    # Convert matches to the format needed for findFundamentalMat
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2, good_matches

def plot_comparison(linear_points, nonlinear_points, camera_position):
    """ Plot a comparison between linear and nonlinear triangulation results. """
    fig, ax = plt.subplots()

    # Plot linear triangulation points
    ax.scatter(linear_points[:, 0], linear_points[:, 2], c='red', label='Linear Triangulation', alpha=0.5)

    # Plot nonlinear triangulation points
    ax.scatter(nonlinear_points[:, 0], nonlinear_points[:, 2], c='blue', label='Nonlinear Triangulation', alpha=0.5)

    # Plot camera position
    ax.scatter(camera_position[0], camera_position[2], c='cyan', marker='^', s=100, label='Camera Position')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    plt.title('Comparison between Non-linear vs Linear Triangulation')
    plt.grid(True)
    plt.show()

def plot_3d_points_and_cameras(points_3D, camera_poses):
    """ Plot 3D points and camera positions in a 3D plot. """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each set of 3D points
    for points in points_3D:
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        ax.scatter(xs, ys, zs)

    # Plot camera positions
    for rotation, translation in camera_poses:
        t = -rotation.T @ translation  # Camera position in world coordinates
        ax.scatter(t[0], t[1], t[2], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Scene Reconstruction')
    plt.show()

def plot_2d_xz(points_3D, camera_poses):
    """ Plot 2D points in the X-Z plane with camera positions. """
    plt.figure(figsize=(10, 5))  
    for points, (rotation, translation) in zip(points_3D, camera_poses):
        # Plotting the points focusing on X and Z axes
        xs = points[:, 0]
        zs = points[:, 1]
        plt.scatter(xs, zs, color='black', alpha=0.6, edgecolors='none')  
        # Plot camera position
        camera_position = -np.dot(rotation.T, translation.reshape(-1, 1))
        plt.scatter(camera_position[0], camera_position[2], color='red', marker='^', s=100) 

    plt.title('2D X-Z Plane Reconstruction (Simplified)')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.grid(True)
    plt.axis('equal')  
    plt.show()

def plot_initial_triangulation_with_disambiguity(points_3d_sets, camera_poses):
    """ Plot initial triangulation with disambiguity. """
    fig, ax = plt.subplots()

    colors = ['red', 'blue', 'green', 'cyan']  
    markers = ['o', '^', 's', 'p']  

    for i, points_3d in enumerate(points_3d_sets):
        xs = points_3d[:, 0] 
        zs = points_3d[:, 1]  
        color = colors[i % len(colors)] 
        marker = markers[i % len(markers)] 
        ax.scatter(xs, zs, c=color, marker=marker, label=f'Pose {i+1}')

        # plot the camera position
        R, C = camera_poses[i]
        camera_position = -np.dot(R.T, C.reshape(-1, 1))
        ax.scatter(camera_position[0], camera_position[2], c=color, marker='x', s=100) 

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    plt.title('2D Plot of Initial Triangulation with Disambiguity')
    plt.grid(True)
    plt.show()


def structure_from_motion(image_paths):
    """ Main SfM pipeline integrating all components. """

    images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_paths]
    # Camera matrix from calibration.txt
    K = np.array([[568.996140852, 0, 643.21055941],[0, 568.988362396, 477.982801038],[0, 0, 1]]) 
    
    all_points_3D_linear = []  # List to store linear triangulation results
    all_points_3D_nonlinear = []  # List to store nonlinear triangulation results
    all_poses = []  # List to store camera poses

    for i in range(len(images) - 1):
        # Match features between consecutive images
        pts1, pts2, matches = match_features(images[i], images[i+1])
        
        # Estimate fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        
        # Get essential matrix from fundamental matrix
        E = get_essential_matrix(K, F)
        
        # Extract camera poses
        rotations, translations = extract_camera_pose(E)
        
        # Perform linear triangulation
        linear_points = [linear_triangulation(K, translations[0], rotations[0], translations[1], rotations[1], pts1, pts2) for j in range(4)]
        
        # Disambiguate camera pose
        correct_rotation, correct_translation, points_3D = disambiguate_camera_pose(rotations, translations, linear_points)
    
        # Perform nonlinear triangulation
        nonlinear_points = nonlinear_triangulation(K, pts1, pts2, points_3D, correct_rotation, correct_translation, correct_rotation, correct_translation)
        
        # Append results to respective lists
        all_points_3D_linear.append(points_3D)
        all_points_3D_nonlinear.append(nonlinear_points)
        all_poses.append((correct_rotation, correct_translation))

    # Calculate camera position from the last camera pose
    last_rotation, last_translation = all_poses[-1]
    camera_position = -np.dot(last_rotation.T, last_translation)

    # Plot comparison between linear and nonlinear triangulation results
    plot_comparison(np.vstack(all_points_3D_linear), np.vstack(all_points_3D_nonlinear), camera_position)

    return all_points_3D_linear, all_points_3D_nonlinear, all_poses, camera_position

if __name__ == "__main__":
    image_paths = [r"Data\1.jpg", r"Data\2.jpg", r"Data\3.jpg", r"Data\4.jpg", r"Data\5.jpg", r"Data\6.jpg"]
    all_points_3D_linear, all_points_3D_nonlinear, all_poses, camera_position = structure_from_motion(image_paths)
    plot_initial_triangulation_with_disambiguity(all_points_3D_linear, all_poses)
    plot_3d_points_and_cameras(all_points_3D_nonlinear, all_poses)
    # plot_2d_xz(all_points_3D_linear, all_poses)
