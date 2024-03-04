import os
import numpy as np
import random
import open3d as o3d
from pyntcloud import PyntCloud
from sklearn.neighbors import KDTree

def read_modelnet40_normal(filepath):
    point_cloud_pynt = PyntCloud.from_file(filepath, sep=",", names=["x", "y", "z", "nx", "ny", "nz"])
    point_cloud_pynt.points = point_cloud_pynt.points[['x', 'y', 'z']]
    points = np.asarray(point_cloud_pynt.points)
    return points


def computeCovarianceEigval(nearest_point_cloud, nearest_distance):
    nearest_distance = np.expand_dims(nearest_distance, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        nearest_point_cloud = nearest_point_cloud / (nearest_distance)
        nearest_point_cloud[~np.isfinite(nearest_point_cloud)] = 0
    nearest_point_cloud_cov = np.cov(nearest_point_cloud.transpose())
    cov = nearest_point_cloud_cov * sum(nearest_distance)
    eigenvectors, eigenvalues, eigenvectorsT = np.linalg.svd(cov)
    eigval_sort_index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[eigval_sort_index]
    return eigenvalues


def iss(points, gamma21=0.05, gamma32=0.7, nms_radius=0.3):
    feature_values = []
    keypoints = []
    keypoints_index_after_nms = []

    leaf_size = 4
    radius = 0.1
    tree = KDTree(points, leaf_size=leaf_size)
    nearest_index, nearest_distance = tree.query_radius(points, radius, return_distance=True)

    eigvals = []
    for i in range(len(nearest_index)):
        eigval = computeCovarianceEigval(points[nearest_index[i]], nearest_distance[i])
        eigvals.append(eigval)
    eigvals = np.asarray(eigvals)

    gamma21 = np.median(eigvals[:, 1] / eigvals[:, 0], axis=0)
    gamma32 = np.median(eigvals[:, 2] / eigvals[:, 1], axis=0)
    lamda = np.median(eigvals[:, 2], axis=0)
    for i in range(eigvals.shape[0]):
        if eigvals[i, 1] / eigvals[i, 0] < gamma21 and eigvals[i, 2] / eigvals[i, 1] < gamma32 and eigvals[i, 2] < lamda:
            feature_values.append(eigvals[i, 2])
            keypoints.append(points[i])
    feature_values = np.asarray(feature_values)
    keypoints = np.asarray(keypoints)

    leaf_size = 8
    keypoints_tree = KDTree(keypoints, leaf_size=leaf_size)
    
    while keypoints[~np.isnan(keypoints)].shape[0]:
        feature_index = np.argmax(feature_values)
        feature_point = keypoints[feature_index]
        feature_point = np.expand_dims(feature_point, axis=0)

        nearest_index = keypoints_tree.query_radius(feature_point, nms_radius)
        keypoints_index_after_nms.append(feature_index)
        keypoints[feature_index] = np.nan
        keypoints[nearest_index[0]] = np.nan
        feature_values[feature_index] = 0
        feature_values[nearest_index[0]] = 0

    keypoints_after_nms = points[keypoints_index_after_nms]
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(points)
    points_o3d.paint_uniform_color([0.5, 0.5, 0.5])
    keypoints_after_nms_o3d = o3d.geometry.PointCloud()
    keypoints_after_nms_o3d.points = o3d.utility.Vector3dVector(keypoints_after_nms)
    o3d.visualization.draw_geometries([points_o3d, keypoints_after_nms_o3d])


if __name__ == '__main__':
    ROOT_PATH = "/Users/fengzetao/Workspace/Github/PointCloud-Learning/datasets/modelnet40_normal_resampled/"
    file_path_airplane = os.path.join(ROOT_PATH, "airplane/airplane_0001.txt")
    file_path_bench = os.path.join(ROOT_PATH, "bench/bench_0001.txt")
    file_path_car = os.path.join(ROOT_PATH, "car/car_0001.txt")

    point_cloud_airplane = read_modelnet40_normal(file_path_airplane)
    point_cloud_bench = read_modelnet40_normal(file_path_bench)
    point_cloud_car = read_modelnet40_normal(file_path_car)

    iss(point_cloud_airplane)
    iss(point_cloud_bench)
    iss(point_cloud_car)