# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    
    # step1 normalize the data
    X_mean = data - np.sum(data, axis=0) / np.size(data, axis=0)

    # step2 compute covariance matrix
    if correlation:
        H = np.corrcoef(X_mean.T)
    else:
        H = np.cov(X_mean.T)

    # step3 compute eigenvalues and eigenvectors via SVD
    eigenvectors, eigenvalues, eigenvectorsT = np.linalg.svd(H)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 加载原始点云
    ROOT_PATH = "/Users/fengzetao/Workspace/Github/PointCloud-Learning/datasets/modelnet40_normal_resampled/"
    paths = os.listdir(ROOT_PATH)
    for path in paths[:3]:
        filename = os.path.join(ROOT_PATH, path, path + '_0001.txt')
        if not os.path.exists(filename):
            continue
        print(filename)
        point_cloud_pynt = PyntCloud.from_file(filename, sep=",", names=["x", "y", "z", "nx", "ny", "nz"])
        point_cloud_pynt.points = point_cloud_pynt.points[['x', 'y', 'z']]
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

        # 从点云中获取点，只对点进行处理
        point_cloud_original = point_cloud_pynt.points
        print('total points number is:', point_cloud_original.shape[0])

        # 用PCA分析点云主方向
        w, v = PCA(point_cloud_original)
        point_cloud_vector = v[:, :2]
        print('the main orientation of this pointcloud is: ', point_cloud_vector)
        new_points = np.zeros_like(point_cloud_original)
        new_points[:, :2] = np.matmul(point_cloud_original, point_cloud_vector)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)
        o3d.visualization.draw_geometries([new_pcd])

        # 循环计算每个点的法向量
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        # 作业2
        normals = []
        # 屏蔽开始

        for i in range(point_cloud_original.shape[0]):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 8)  # pick 8 nearest points to compute normal
                k_nearest_point = np.asarray(point_cloud_original)[idx, :]
                w, v = PCA(k_nearest_point)
                normals.append(v[:, 2])

        # 屏蔽结束
        normals = np.array(normals, dtype=np.float64)
        point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
