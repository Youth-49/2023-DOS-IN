import math

import numpy as np
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt


class IDPC:
    """
    IDPC clustering.

    Parameters
    ----------
    n_clusters: int, default=3
        The number of clusters to form as well as the number of
        centroids to generate.

    p: int, default=0.02
        .0~0.48
    """
    centers = None
    step12_label = None
    step3_label = None
    density = None
    delta = None
    __k = 0

    def __init__(self, n_clusters=3, p=0.02):
        self.n_clusters = n_clusters
        self.p = p

    def set_param(self, nc, p):
        self.n_clusters = nc
        self.p = p

    def fit_predict(self, X):
        row_num, col_num = X.shape
        dist_matrix, dist_index = self.get_distance(X)
        density, density_index = self.get_density(dist_matrix, dist_index, row_num)
        delta = self.get_delta(density_index, dist_matrix, row_num)
        centers = self.get_centers(density, delta, row_num)
        labels = self.get_labels(dist_matrix, dist_index, centers, density, density_index, row_num)
        return labels

    def get_distance(self, X):
        # 求距离（常规）
        distance = dis.pdist(X)
        distance_matrix = dis.squareform(distance)
        # distance_sort = np.sort(distance_matrix, axis=1)
        distance_index = np.argsort(distance_matrix, axis=1)
        return distance_matrix, distance_index

    def get_density(self, dist, dist_sort, row_num):
        density = np.zeros(row_num, dtype=np.float64)
        # avg_max = 0
        # for i in range(row_num):
        #     avg_max += dist[i][dist_sort[i][row_num-1]]
        # avg_max /= row_num
        k = int(self.p * row_num)
        self.__k = k
        for i in range(row_num):
            s = 0
            for j in range(k):
                if j > row_num:
                    break
                s = s + dist[i][dist_sort[i][j + 1]]*dist[i][dist_sort[i][j + 1]]
            density[i] = math.exp(- s / k)
        density_index = np.argsort(density)  # 对密度排序，得到排序序号
        # print(density)
        self.density = density.copy()
        return density, density_index

    # def get_density(self, dist, dist_sort, row_num):
    #     density = np.zeros(row_num)
    #     area = np.mean(dist_sort[:, self.k])
    #     # 求密度(高斯密度)
    #     for i in range(row_num - 1):
    #         for j in range(i + 1, row_num):
    #             density[i] = density[i] + math.exp(- (dist[i][j] * dist[i][j]) / (area * area))
    #             density[j] = density[j] + math.exp(- (dist[i][j] * dist[i][j]) / (area * area))
    #     # density_sort = np.sort(density)
    #     density_index = np.argsort(density)  # 对密度排序，得到排序序号
    #     return density, density_index

    def get_delta(self, density_index, dist_matrix, row_num):
        # max_dis = np.amax(pdist_matrix)
        max_delta = 0
        delta = np.zeros(row_num, dtype=np.float64)
        density_index = density_index[::-1]
        for i in range(row_num):
            if i == 0:
                continue
            delta[density_index[i]] = float('inf')
            for j in range(i):
                dij = dist_matrix[density_index[i]][density_index[j]]
                if dij < delta[density_index[i]]:
                    delta[density_index[i]] = dij
            if delta[density_index[i]] > max_delta:
                max_delta = delta[density_index[i]]
        delta[density_index[0]] = max_delta * 1.1
        self.delta = delta.copy()
        return delta

    def get_centers(self, density, delta, row_num):
        mu_den = np.mean(density)
        mu_del = np.mean(delta)
        sigma_den = np.std(density)
        sigma_del = np.std(delta)
        # density = (density - mu_den) / sigma_den
        # delta = (delta - mu_del) / sigma_del
        gamma = np.zeros(row_num)
        for i in range(row_num):
            gamma[i] = density[i] * delta[i]
        clustering_center = np.argsort(gamma)[row_num - self.n_clusters:]
        self.centers = clustering_center
        return clustering_center

    def get_labels(self, dist_matrix, dist_index, centers, density, density_index, row_num):
        label = np.zeros(row_num, dtype=np.int8)
        density_index = density_index[::-1]

        # Step 1: A distinct label is assigned to each cluster center.
        # Step 2: Each cluster center propagates its label to its k nearest
        # neighbors.
        i = 1
        for c in centers:
            label[c] = i
            for j in range(self.__k):
                if j+1 >= row_num:
                    break
                label[dist_index[c][j + 1]] = i
            i = i + 1
        self.step12_label = label.copy()

        # Step 3: For each data point 𝑥𝑖 which doesn’t have any label, if its
        # local density value 𝜌𝑖 is lower than 𝜌𝑗 (i.e. 𝑥𝑗 a neighbor of
        # 𝑥𝑖) then 𝑥𝑖 takes the label of 𝑥𝑗. If the local density of more
        # than one neighbor is higher than 𝜌𝑖, then the label of 𝑥𝑖 is
        # identified using the voting method.
        while True:
            update_num = 0
            update = []
            temp_label = [0 for i in range(row_num)]
            for i in range(row_num):
                if label[i] != 0:
                    continue
                votes = [0 for i in range(self.n_clusters + 1)]
                max_votes = 0
                max_votes_center = 0
                for j in range(self.__k):
                    if j+1 >= row_num:
                        break
                    if density[dist_index[i][j+1]] <= density[i]:
                        continue
                    lb = label[dist_index[i][j + 1]]
                    if lb == 0:
                        continue
                    votes[lb] += 1
                    if votes[lb] > max_votes:
                        max_votes = votes[lb]
                        max_votes_center = lb
                temp_label[i] = max_votes_center
                if max_votes_center != 0:
                    update_num += 1
                    update.append(i)
            if update_num != 0:
                for index in update:
                    label[index] = temp_label[index]
            elif update_num == 0:
                break
        self.step3_label = label.copy()

        # Step 4: Finally if a data point 𝑥𝑖 doesn’t have any label, the label
        # of nearest cluster center is assigned to 𝑥𝑖.
        for p in density_index:
            if label[p] == 0:
                nearest_center = 0
                nearest_center_dis = float('inf')
                for c in centers:
                    if dist_matrix[p][c] < nearest_center_dis:
                        nearest_center_dis = dist_matrix[p][c]
                        nearest_center = c
                label[p] = label[nearest_center]

        return label
