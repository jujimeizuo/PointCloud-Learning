import numpy as np
import scipy
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.neighbors import kneighbors_graph
import KMeans

plt.style.use('seaborn-v0_8')

class Spectral(object):
    def __init__(self, n_clusters, n_neighbors=10):
        self.n_clusters_ = n_clusters
        self.n_neighbors_ = n_neighbors
        self.weight_ = None
        self.degree_ = None
        self.laplacians_ = None
        self.eigenvector_ = None

    def fit(self, data):
        # 1. construct weight matrix
        weight = kneighbors_graph(data, n_neighbors=self.n_neighbors_, mode='connectivity', include_self=False)
        weight = 0.5 * (weight + weight.T)
        self.weight_ = weight.toarray()
        self.degree_ = np.diag(np.sum(self.weight_, axis=0).ravel())

        # 2. construct laplacian matrix and normalize
        self.laplacians_ = self.degree_ - self.weight_
        degree_nor = np.sqrt(np.linalg.inv(self.degree_))
        self.laplacians_ = np.dot(degree_nor, self.laplacians_)
        self.laplacians_ = np.dot(self.laplacians_, degree_nor)

        # 3. compute minimum k eigenvalues
        eigenvalues, eigenvector = np.linalg.eigh(self.laplacians_)
        sorted_index = eigenvalues.argsort()
        eigenvector = eigenvector[:, sorted_index]
        self.eigenvector_ = np.asarray([eigenvector[:, i] for i in range(self.n_clusters_)]).T
        self.eigenvector_ /= np.linalg.norm(self.eigenvector_, axis=1).reshape(data.shape[0], 1)

        # 4. kmeans with eigenvectors
        spectral_kmeans = KMeans.K_Means(n_clusters=self.n_clusters_)
        spectral_kmeans.fit(self.eigenvector_)
        spectral_label = spectral_kmeans.predict(self.eigenvector_)
        self.label_ = spectral_label
        self.fitted = True
    
    def predict(self, data):
        result = []
        if not self.fitted:
            raise ValueError("Spectral not fitted yet")
        return np.copy(self.label_)

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


# 二维点云显示函数
def Point_Show(point,color):
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
    #X = np.array([[1, 2], [2, 3], [5, 8], [8, 8], [1, 6], [9, 11]])

    spectral = Spectral(n_clusters=3)
    K = 3
    spectral.fit(X)
    cat = spectral.predict(X)
    print(cat)
    cluster =[[] for i in range(K)]
    for i in range(len(X)):
        if cat[i] == 0:
            cluster[0].append(X[i])
        elif cat[i] == 1:
            cluster[1].append(X[i])
        elif cat[i] == 2:
            cluster[2].append(X[i])
    Point_Show(cluster[0],color="red")
    Point_Show(cluster[1], color="orange")
    Point_Show(cluster[2],color="blue")
    plt.show()