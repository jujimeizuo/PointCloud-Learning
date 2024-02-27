# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, mode='centroid'):
    points = np.array(point_cloud)
    filtered_points = []
    # 作业3
    # 屏蔽开始

    # 1. 计算点云的范围
    # x_max, y_max, z_max = np.max(points, axis=0)
    # x_min, y_min, z_min = np.min(points, axis=0)

    x_max, y_max, z_max = max(points[:, 0]), max(points[:, 1]), max(points[:, 2])
    x_min, y_min, z_min = min(points[:, 0]), min(points[:, 1]), min(points[:, 2])
    print(x_max, y_max, z_max)
    print(x_min, y_min, z_min)

    # 2. 确定纬度
    x_dim = (x_max - x_min) // leaf_size
    y_dim = (y_max - y_min) // leaf_size
    z_dim = (z_max - z_min) // leaf_size

    # 3. 计算每个点索引
    ids = []
    for p in points:
        x_index = (p[0] - x_min) // leaf_size
        y_index = (p[1] - y_min) // leaf_size
        z_index = (p[2] - z_min) // leaf_size
        # print(p[0], x_min, p[1], y_min, p[2], z_min)
        # print(x_index + y_index * x_dim + z_index * x_dim * y_dim)
        ids.append(x_index + y_index * x_dim + z_index * x_dim * y_dim)

    # 4. 索引排序
    sorted_ids = np.sort(ids)
    sorted_ids_index = np.argsort(ids)

    # 5. 遍历排序后的点，进行滤波
    local_points = []
    previous_id = sorted_ids[0]
    for i in range(len(sorted_ids)):
        if sorted_ids[i] == previous_id:
            local_points.append(points[sorted_ids_index[i]])
        else:
            if mode == 'centroid':
                new_point = np.mean(local_points, axis=0)
                filtered_points.append(new_point)
            elif mode == 'random':
                np.random.shuffle(local_points)
                filtered_points.append(local_points[0])
            previous_id = sorted_ids[i]
            local_points = [points[sorted_ids_index[i]]]
    
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    print(filtered_points)

    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

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

        # 调用voxel滤波函数，实现滤波
        filtered_cloud = voxel_filter(point_cloud_pynt.points, 0.05, mode='random')
        point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
        # 显示滤波后的点云
        o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
