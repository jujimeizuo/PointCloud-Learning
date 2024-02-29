# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import KMeans

plt.style.use('seaborn-v0_8')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50, tolerance = 0.00001):
        self.k_ = n_clusters
        self.max_iter_ = max_iter
        self.tolerance_ = tolerance

    
    # 屏蔽开始
    # 更新W
        self.posteriori = None
    
    # 更新pi
        self.prior = None
 
    # 更新Mu
        self.mu = None

    # 更新Var
        self.cov = None

    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        
        # 1. init Mu pi cov
        k_means = KMeans.K_Means(n_clusters=self.k_)
        k_means.fit(data)
        self.mu = np.asarray(k_means.centers_)
        self.cov = np.asarray([eye(2, 2)] * self.k_)
        self.prior = np.asarray([1 / self.k_] * self.k_).reshape(self.k_, 1)
        self.posteriori = np.zeros((self.k_, len(data)))
        # print(self.mu)
        # print(self.cov)
        # print(self.prior)
        
        MLE = -inf
        for _ in range(self.max_iter_):
            
            # 2. E-Step
            for k in range(self.k_):
                self.posteriori[k] = multivariate_normal.pdf(x=data, mean=self.mu[k], cov=self.cov[k]) 
            self.posteriori = np.dot(diag(self.prior.ravel()), self.posteriori)
            self.posteriori /= np.sum(self.posteriori, axis=0)
            
            # 3. M-Step
            self.Nk = np.sum(self.posteriori, axis=1)
            self.mu = np.asarray([np.dot(self.posteriori[k], data) / self.Nk[k] for k in range(self.k_)])
            self.cov = np.asarray([np.dot((data-self.mu[k]).T, np.dot(np.diag(self.posteriori[k].ravel( )), data-self.mu[k])) / self.Nk[k] for k in range(self.k_)])
            self.prior = np.asarray( [self.Nk / self.k_]).reshape(self.k_, 1)

            nMLE = np.sum(np.log(self.posteriori))
            if np.abs(nMLE - MLE) < self.tolerance_:
                break
            MLE = np.copy(nMLE)

        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        
        result = np.argmax(self.posteriori, axis=0)
        return result

        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


def points_show(point,color):
    x = []
    y = []
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    plt.scatter(x, y,color=color)


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)  # 确定中心点的位置
    cat = gmm.predict(X)  # 提取点的索引
    K = 3

    # visualize:
    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels = [f'Cluster{k:02d}' for k in range(K)]

    cluster = [[] for i in range(K)]  # 用于分类所有数据点
    for i in range(len(X)):
        if cat[i] == 0:
            cluster[0].append(X[i])
        elif cat[i] == 1:
            cluster[1].append(X[i])
        elif cat[i] == 2:
            cluster[2].append(X[i])

    points_show(cluster[0], color="red")
    points_show(cluster[1], color="blue")
    points_show(cluster[2], color="yellow")
    plt.show()
