# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import random
import math
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pyntcloud import PyntCloud
import open3d as o3d

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始

    # 初始化数据
    idx_segmented = []
    segmented_cloud = []
    iters = 100
    sigma = 0.4

    # 最好的模型参数估计和内点数量，aX + bY + cZ + d = 0
    best_a, best_b, best_c, best_d = 0, 0, 0, 0
    pretotal = 0 # 上一次 inlier 的数量

    # 希望得到正确模型的概率
    P = 0.99
    
    n = len(data) # 点的数量
    outlier_ratio = 0.6

    for i in range(iters):
        ground_cloud = []
        idx_ground = []

        # step1 选择可以估计出模型的最小数据集，对于平面拟合来说，就是三个点
        sample_index = random.sample(range(n), 3)
        point1 = data[sample_index[0]]
        point2 = data[sample_index[1]]
        point3 = data[sample_index[2]]

        # step2 求解模型

        ## step2.1 求解法向量
        point1_2 = point2 - point1
        point1_3 = point3 - point1
        plane_normal = np.cross(point1_2, point1_3)

        ## step2.2 求解模型的 a,b,c,d
        a = plane_normal[0]
        b = plane_normal[1]
        c = plane_normal[2]
        d = -plane_normal.dot(point1)

        # step3 将所有数据带入模型，计算出内点的数量
        
        ## step3.1 求 sample（三点）外的点与 sample 内的三点其中一点的距离
        print(point1)

        total_inlier = 0
        pointn_1 = (data - point1)
        distance = abs(pointn_1.dot(plane_normal)) / np.linalg.norm(plane_normal)

        ## step3.2 使用距离判断 inlier
        idx_ground = (distance <= sigma)
        total_inlier = np.sum(idx_ground == True)

        ## step3.3 更新当前模型的参数
        if total_inlier > pretotal:
            iters = math.log(1 - P) / math.log(1 - pow(total_inlier / n, 3))
            pretotal = total_inlier
            best_a, best_b, best_c, best_d = a, b, c, d
        
        ## step3.4 判断当前模型是否符合超过 inlier_ratio
        if total_inlier > n * (1 - outlier_ratio):
            break
    
    print("iters = %f" % iters)

    # step4 提取分割后的点
    idx_segmented = np.logical_not(idx_ground)
    ground_cloud = data[idx_ground]
    segmented_cloud = data[idx_segmented]

    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmented_cloud.shape[0])
    return ground_cloud, segmented_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始

    clusters_index = []
    dbscan = cluster.DBSCAN(eps=1.2)
    dbscan.fit(data)
    clusters_index = dbscan.labels_.astype(np.int64)

    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    filename = '/Users/fengzetao/Workspace/Github/PointCloud-Learning/datasets/000000.bin' # 数据集路径
    print('clustering pointcloud file:', filename)

    origin_points = read_velodyne_bin(filename)
    origin_points_df = pd.DataFrame(origin_points, columns=['x', 'y', 'z'])
    point_cloud_pynt = PyntCloud(origin_points_df)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d])

    # 地面分割
    ground_points, segmented_points = ground_segmentation(data=origin_points)
    ground_points_df = pd.DataFrame(ground_points, columns=['x', 'y', 'z'])
    point_cloud_pynt_ground = PyntCloud(ground_points_df)
    point_cloud_o3d_ground = point_cloud_pynt_ground.to_instance("open3d", mesh=False)
    point_cloud_o3d_ground.paint_uniform_color([0, 0, 255])

    # 显示 segmented_points 地面点云
    segmented_points_df = pd.DataFrame(segmented_points, columns=['x', 'y', 'z'])
    point_cloud_pynt_segmented = PyntCloud(segmented_points_df)
    point_cloud_o3d_segmented = point_cloud_pynt_segmented.to_instance("open3d", mesh=False)
    point_cloud_o3d_segmented.paint_uniform_color([255, 0, 0])

    o3d.visualization.draw_geometries([point_cloud_o3d_ground, point_cloud_o3d_segmented])

    cluster_index = clustering(segmented_points)
    plot_clusters(segmented_points, cluster_index)


if __name__ == '__main__':
    main()
