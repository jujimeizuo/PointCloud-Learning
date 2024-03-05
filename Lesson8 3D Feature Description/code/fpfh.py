import open3d as o3d
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_modelnet40_normal(filepath):
    # load data:
    df_point_cloud_with_normal = pd.read_csv(
        filepath, header=None
    )
    # add colunm names:
    df_point_cloud_with_normal.columns = [
        'x', 'y', 'z',
        'nx', 'ny', 'nz'
    ]
    
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(
        df_point_cloud_with_normal[['x', 'y', 'z']].values
    )
    point_cloud.normals = o3d.utility.Vector3dVector(
        df_point_cloud_with_normal[['nx', 'ny', 'nz']].values
    )

    return point_cloud


def detect(point_cloud, search_tree, radius):
    # points handler:
    points = np.asarray(point_cloud.points)

    # keypoints container:
    keypoints = {'id': [],'x': [],'y': [],'z': [],'lambda_0': [],'lambda_1': [],'lambda_2': []}

    # cache for number of radius nearest neighbors:
    num_rnn_cache = {}
    # heapq for non-maximum suppression:
    pq = []
    for idx_center, center in enumerate(points):
        # find radius nearest neighbors:
        [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(center, radius)

        # for each point get its nearest neighbors count:
        w = []
        deviation = []
        for idx_neighbor in np.asarray(idx_neighbors[1:]):
            # check cache:
            if not idx_neighbor in num_rnn_cache:
                [k_, _, _] = search_tree.search_radius_vector_3d(points[idx_neighbor], radius)
                num_rnn_cache[idx_neighbor] = k_
            # update:
            w.append(num_rnn_cache[idx_neighbor])
            deviation.append(points[idx_neighbor] - center)
        
        # calculate covariance matrix:
        w = np.asarray(w)
        deviation = np.asarray(deviation)

        cov = (1.0 / w.sum()) * np.dot(deviation.T,np.dot(np.diag(w), deviation))

        # get eigenvalues:
        w, _ = np.linalg.eig(cov)
        w = w[w.argsort()[::-1]]

        # add to pq:
        heapq.heappush(pq, (-w[2], idx_center))

        # add to dataframe:
        keypoints['id'].append(idx_center)
        keypoints['x'].append(center[0])
        keypoints['y'].append(center[1])
        keypoints['z'].append(center[2])
        keypoints['lambda_0'].append(w[0])
        keypoints['lambda_1'].append(w[1])
        keypoints['lambda_2'].append(w[2])
    
    # non-maximum suppression:
    suppressed = set()
    while pq:
        _, idx_center = heapq.heappop(pq)
        if not idx_center in suppressed:
            # suppress its neighbors:
            [_, idx_neighbors, _] = search_tree.search_radius_vector_3d(points[idx_center], radius)
            for idx_neighbor in np.asarray(idx_neighbors[1:]):
                suppressed.add(idx_neighbor)
        else:
            continue

    # format:        
    keypoints = pd.DataFrame.from_dict(keypoints)

    # first apply non-maximum suppression:
    keypoints = keypoints.loc[keypoints['id'].apply(lambda id: not id in suppressed),keypoints.columns]

    # then apply decreasing ratio test:
    keypoints = keypoints.loc[(keypoints['lambda_0'] > keypoints['lambda_1']) &(keypoints['lambda_1'] > keypoints['lambda_2']),keypoints.columns]

    return keypoints

## 入口函数
#  输入：point_cloud: 点云数据
#        search_tree: 点云搜索树
#        keypoint_id: 关键点的索引
#        radius:      r领域半径
#        B:           每个描述子特殊的维度，fpfh中为[α,ф,θ]，最终描述子的维度为3B
#  输出：3B描述子
def get_spfh(point_cloud, search_tree, keypoint_id, radius, B):
    # points handler:
    points = np.asarray(point_cloud.points)

    # get keypoint:
    keypoint = np.asarray(point_cloud.points)[keypoint_id]

    # find radius nearest neighbors:
    [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(keypoint, radius)
    # remove query point:
    idx_neighbors = idx_neighbors[1:]
    # get normalized diff:
    diff = points[idx_neighbors] - keypoint 
    diff /= np.linalg.norm(diff, ord=2, axis=1)[:,None]   #diff 维数(k*3)

    # get n1:
    n1 = np.asarray(point_cloud.normals)[keypoint_id]
    # get u:
    u = n1                 # u维数(3,)
    # get v:
    v = np.cross(u, diff)  # v维数(k*3)
    # get w:
    w = np.cross(u, v)     # 维数(k*3)

    # get n2:
    n2 = np.asarray(point_cloud.normals)[idx_neighbors]   #n2维数(k*3)
    # get alpha:
    alpha = (v * n2).sum(axis=1)  #alpha维数(k,)
    alpha_min,alpha_max=min(alpha),max(alpha)

    # get phi:
    phi = (u*diff).sum(axis=1)    #phi维数(k,)
    phi_min, phi_max = min(phi), max(phi)

    # get theta:
    theta = np.arctan2((w*n2).sum(axis=1), (u*n2).sum(axis=1))  #theta维数(k,)
    theta_min, theta_max = min(theta), max(theta)

    # 因为histogram返回值有两个参数，[0]选择第一个参数
    alpha_histogram = np.histogram(alpha, bins=B, range=(-1.0, +1.0))[0]
    alpha_histogram = alpha_histogram / alpha_histogram.sum()
    # get phi histogram:
    phi_histogram = np.histogram(phi, bins=B, range=(-1.0, +1.0))[0]
    phi_histogram = phi_histogram / phi_histogram.sum()
    # get theta histogram:
    theta_histogram = np.histogram(theta, bins=B, range=(-np.pi, +np.pi))[0]
    theta_histogram = theta_histogram / theta_histogram.sum()

    # build signature:
    #hstack进行水平扩展，vstack进行竖直扩展
    signature = np.hstack((alpha_histogram,phi_histogram,theta_histogram))  #维数(3*B,)

    return signature

## 入口函数
#  输入：point_cloud: 点云数据
#        search_tree: 点云搜索树
#        keypoint_id: 关键点的索引
#        radius:      r领域半径
#        B:           每个描述子特殊的维度，fpfh中为[α,ф,θ]，最终描述子的维度为3B
#  输出：fpfh的结果
def describe(point_cloud, search_tree, keypoint_id, radius, B):
    # points handler:
    points = np.asarray(point_cloud.points)

    # get keypoint:
    keypoint = np.asarray(point_cloud.points)[keypoint_id]

    # find radius nearest neighbors:
    [k, idx_neighbors, _] = search_tree.search_radius_vector_3d(keypoint, radius)

    if k <= 1:
        return None

    # remove query point:
    idx_neighbors = idx_neighbors[1:]

    # weights:
    w = 1.0 / np.linalg.norm(points[idx_neighbors] - keypoint, ord=2, axis=1)

    # SPFH from neighbors:
    X = np.asarray([get_spfh(point_cloud, search_tree, i, radius, B) for i in idx_neighbors])

    # neighborhood contribution:
    spfh_neighborhood = 1.0 / (k - 1) * np.dot(w, X)

    # query point spfh:
    spfh_query = get_spfh(point_cloud, search_tree, keypoint_id, radius, B)

    # finally:
    fpfh = spfh_query + spfh_neighborhood

    # normalize again:
    fpfh = fpfh / np.linalg.norm(fpfh)

    return fpfh


if __name__ == '__main__':
    # parse arguments:
    input = "/Users/fengzetao/Workspace/Github/PointCloud-Learning/datasets/modelnet40_normal_resampled/chair/chair_0001.txt"
    radius = 0.05
    # 加载点云
    point_cloud = read_modelnet40_normal(input)

    # 建立搜索树
    search_tree = o3d.geometry.KDTreeFlann(point_cloud)

    # detect keypoints:
    keypoints = detect(point_cloud, search_tree, radius)

    # 可视化
    point_cloud.paint_uniform_color([0.50, 0.50, 0.50])
    # show roi:
    max_bound = point_cloud.get_max_bound()
    min_bound = point_cloud.get_min_bound()
    center = (min_bound + max_bound) / 2.0
    print("min_bound:",min_bound[0],min_bound[1],min_bound[2])
    print("max_bound:", max_bound[0], max_bound[1], max_bound[2])
    print("center:",center[0],center[1],center[2])

    #椅子的平面[-0.1,0.1]，椅子角为[-0.8,-0.6]
    min_bound[1]=-0.1
    max_bound[1]=0.1
    # min_bound[1] = max_bound[1] - 0.1
    # max_bound[1] = max_bound[1]
    # min_bound[2] = center[2]
    # max_bound[2] = max_bound[2]

    #可视化提取ROI区域内的特征点
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
    roi = point_cloud.crop(bounding_box)
    roi.paint_uniform_color([1.00, 0.00, 0.00])
    keypoints_in_roi = keypoints.loc[
                       (
                           ((keypoints['x'] >= min_bound[0]) & (keypoints['x'] <= max_bound[0])) &
                           ((keypoints['y'] >= min_bound[1]) & (keypoints['y'] <= max_bound[1])) &
                           ((keypoints['z'] >= min_bound[2]) & (keypoints['z'] <= max_bound[2]))
                       ),:]
    print("筛选后的特征点的个数:",len(keypoints_in_roi['id']))
    np.asarray(point_cloud.colors)[ keypoints_in_roi['id'].values, :] = [1.0, 0.0, 0.0]
    # 添加坐标系
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([FOR1,point_cloud])

    # 计算特征点的描述子:
    df_signature_visualization = []
    for keypoint_id in keypoints_in_roi['id'].values:
        signature = describe(point_cloud, search_tree, keypoint_id, radius,6)  #B=20
        df_ = pd.DataFrame.from_dict({'index': np.arange(len(signature)),'feature': signature})
        df_['keypoint_id'] = keypoint_id
        df_signature_visualization.append(df_)

    df_signature_visualization = pd.concat(df_signature_visualization)
    #保存特征点的描述子，以data,csv的格式进行保存，可用EXCEL打开查看
    df_signature_visualization.to_csv("data.csv",encoding='utf-8')

    #在求椅子的平面时,df_signature_visualization为215*3B，其中215为特征点的个数，故head里面不要超过3B
    df_signature_visualization = df_signature_visualization.head(60)

    # 画线
    plt.figure(num=None, figsize=(16, 9))
    sns.lineplot(
        x="index", y="feature",
        hue="keypoint_id", style="keypoint_id",
        markers=True, dashes=False, data=df_signature_visualization
    )
    plt.title('Description Visualization for Keypoints')
    plt.show()
