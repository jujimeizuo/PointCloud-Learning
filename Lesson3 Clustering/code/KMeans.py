# 文件功能： 实现 K-Means 算法
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import spatial

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始

        # 1. 随机选择k个点作为初始的聚类中心
        self.centers_ = data[random.sample(range(data.shape[0]), self.k_)]
        old_centers = np.copy(self.centers_)

        # 2. E-step（expectation）
        leaf_size = 1
        k = 1
        for _ in range(self.max_iter_):
            labels = [[] for i in range(self.k_)]
            root = spatial.KDTree(self.centers_, leafsize=leaf_size)
            for i in range(data.shape[0]):
                _, query_index = root.query(data[i], k)
                labels[query_index].append(data[i])
            for i in range(self.k_):
                points = np.array(labels[i])
                self.centers_[i] = points.mean(axis=0)
            if np.sum(np.abs(self.centers_ - old_centers)) < self.tolerance_ * self.k_:
                break
            old_centers = np.copy(self.centers_)


        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始

        for point in p_datas:
            diff = np.linalg.norm(self.centers_ - point, axis=1)
            result.append(np.argmin(diff))

        # 屏蔽结束
        return result

if __name__ == '__main__':
    # x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    db_size = 10
    dim = 2
    x = np.genfromtxt(r"/Users/fengzetao/Workspace/Github/PointCloud-Learning/datasets/point.txt", delimiter="").reshape((-1, 2))
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    cat = k_means.predict(x)
    color = np.array(['r', 'g'])
    cats = np.array(cat)
    # plt.figure(figsize=(10, 10))
    plt.title('K-Means')
    plt.scatter(x[:, 0], x[:, 1], c=color[cats])
    plt.show()
    print(cat)