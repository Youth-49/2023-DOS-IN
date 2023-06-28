import math

import numpy as np
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import time

class DPCKNN:
    """
    IDPC clustering.

    Parameters
    ----------

    p: int, default=0.02
        p in 0.001, 0.002, 0.005, 0.01, 0.02, 0.06
    """
    __density = None
    __density_index = None
    __delta = None
    __centers = None

    def __init__(self, p=0.02):
        self.p = p

    def set_param(self, p):
        self.p = p

    def line_select_callback(self, eclick, erelease):  # 框选的回调
        """--
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        centers = []
        # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        # print(f" The buttons you used were: {eclick.button} {erelease.button}")
        for index in reversed(self.__density_index):
            if self.__density[index] >= x1:
                if self.__delta[index] >= y1:
                    centers.append(index)
                    # print(density[index])
                    # print(delta[index])
            else:
                break
        self.__centers = centers
        print(centers)
        return

    def plot_decision(self, density, delta):
        fig1, ax1 = plt.subplots()
        density = density - np.amin(density)
        self.__density = density
        delta = delta - np.amin(delta)
        self.__delta = delta
        ax1.scatter(density, delta)
        ax1.set_title(
            "(DPC-KNN)Click and drag to rectangle-select cluster centers.\n")
        # ax1.scatter(density[clustering_center], delta[clustering_center], s=10, c='red')
        RS = RectangleSelector(ax1, self.line_select_callback,
                               drawtype='box', useblit=True,
                               button=[1, 3],  # disable middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        plt.show()

    def fit_predict(self, X):
        row_num, col_num = X.shape
        dist_matrix, dist_index = self.get_distance(X)
        density, density_index = self.get_density(dist_matrix, dist_index, row_num)
        delta = self.get_delta(density, density_index, dist_matrix, dist_index, row_num)
        # centers = self.get_centers(density, delta, row_num)
        t1 = time.perf_counter()
        self.plot_decision(density, delta)
        t2 = time.perf_counter()
        labels = self.get_labels(dist_matrix, row_num)
        return labels, t2-t1

    def get_distance(self, X):
        # 求距离（常规）
        distance = dis.pdist(X)
        distance_matrix = dis.squareform(distance)
        # distance_sort = np.sort(distance_matrix, axis=1)
        distance_index = np.argsort(distance_matrix, axis=1)
        return distance_matrix, distance_index

    def get_density(self, dist, dist_sort, row_num):
        k = math.ceil(row_num*self.p)
        # k = math.floor(row_num*self.p)
        density = np.zeros(row_num)
        # avg_max = 0
        # for i in range(row_num):
        #     avg_max += dist[i][dist_sort[i][row_num-1]]
        # avg_max /= row_num
        for i in range(row_num):
            s = float(0)
            for j in range(k):
                if j > row_num:
                    break
                # s += dist[i][dist_sort[i][j + 1]] * dist[i][dist_sort[i][j + 1]]
                s = s + dist[i][dist_sort[i][j + 1]] * dist[i][dist_sort[i][j + 1]]
            density[i] = math.exp(- s / k)
        density_index = np.argsort(density)  # 对密度排序，
        self.__density_index = density_index.copy()
        # print(density)
        return density, density_index

    # def get_density(self, dist, dist_sort, row_num):
    #     density = np.zeros(row_num)
    #     area = np.mean(dist_sort[:, math.ceil(row_num*self.p)])
    #     # 求密度(高斯密度)
    #     for i in range(row_num - 1):
    #         for j in range(i + 1, row_num):
    #             density[i] = density[i] + math.exp(- (dist[i][j] * dist[i][j]) / (area * area))
    #             density[j] = density[j] + math.exp(- (dist[i][j] * dist[i][j]) / (area * area))
    #     # density_sort = np.sort(density)
    #     density_index = np.argsort(density)  # 对密度排序，得到排序序号
    #     self.density = density.copy()
    #     return density, density_index

    def get_delta(self, density, density_index, dist_matrix, dist_index, row_num):
        # max_dis = np.amax(pdist_matrix)
        max_delta = 0
        delta = np.zeros(row_num)
        density_index = density_index[::-1]
        # print(density_index)
        for i in range(row_num):
            delta[density_index[i]] = dist_matrix[i][dist_index[i][row_num - 1]]
            if i == 0:
                continue
            for j in range(i):
                dij = dist_matrix[density_index[i]][density_index[j]]
                if dij < delta[density_index[i]]:
                    delta[density_index[i]] = dij
            if delta[density_index[i]] > max_delta:
                max_delta = delta[density_index[i]]
        delta[density_index[0]] = max_delta*1.1
        self.delta = delta.copy()
        return delta

    def get_labels(self, dist_matrix, row_num):
        # dim = len(parent)
        label = np.zeros(row_num, dtype=np.int8)

        i = 1
        for c in self.__centers:
            label[c] = i
            i = i + 1

        for p in range(row_num):
            if label[p] == 0:
                nearest_center = 0
                nearest_center_dis = float('inf')
                for c in self.__centers:
                    if dist_matrix[p][c] < nearest_center_dis:
                        nearest_center_dis = dist_matrix[p][c]
                        nearest_center = c
                label[p] = label[nearest_center]

        return label


# test
# knn = DPCKNN(p=0.005)
# data = np.loadtxt('data/flame.txt', dtype=np.float32)
# data = data[:, :2]
# result = knn.fit_predict(data)
