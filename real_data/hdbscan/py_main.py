import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.distance import pdist, squareform
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score

# Here is an implementation of HDBSCAN algorithm in Python without importing the 'hdbscan' package:
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import linkage


class HDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=None, alpha=1.0):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha

    def fit_predict(self, X):
        # Compute the mutual reachability distance matrix
        M = self._mutual_reachability(X)

        # Compute the core distances
        core_distances = self._core_distances(M)

        # Compute the minimum spanning tree
        T = self._minimum_spanning_tree(M)

        # Compute the cluster hierarchy
        hierarchy = self._cluster_hierarchy(T, core_distances)

        # Extract the flat clusters
        labels = self._extract_flat_clusters(hierarchy, X)

        return labels

    def _mutual_reachability(self, X):
        # Compute the distance matrix
        D = pairwise_distances(X)

        # Compute the mutual reachability distance matrix
        M = np.zeros_like(D)
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                mutual_reach_dist = max(D[i,j], self.alpha * np.min(D[i,:]), self.alpha * np.min(D[j,:]))
                M[i,j] = mutual_reach_dist
                M[j,i] = mutual_reach_dist

        return M

    def _core_distances(self, M):
        # Compute the core distances
        k = self.min_samples or self.min_cluster_size
        core_distances = np.zeros(len(M))
        for i in range(len(M)):
            sorted_reach_dists = np.sort(M[i,:])
            core_distances[i] = sorted_reach_dists[k]

        return core_distances

    def _minimum_spanning_tree(self, M):
        # Compute the minimum spanning tree
        T = minimum_spanning_tree(M)
        T = T.toarray()
        T = (T+T.T)
        return T

    def _cluster_hierarchy(self, T, core_distances):
        print(T.shape)
        # Compute the cluster hierarchy
        hierarchy = linkage(squareform(T), method='single')
        print(hierarchy)
        print(hierarchy.shape)
        hierarchy[:,2] = core_distances[hierarchy[:,0].astype(int)] + core_distances[hierarchy[:,1].astype(int)]

        return hierarchy

    def _extract_flat_clusters(self, hierarchy, X):
        # Extract the flat clusters
        labels = np.zeros(X.shape[0], dtype=int)
        cluster_label = 0
        for i in range(hierarchy.shape[0]):
            if hierarchy[i, 2] > self.core_distances_threshold:
                break
            if labels[hierarchy[i, 0]] == 0 and labels[hierarchy[i, 1]] == 0:
                cluster_label += 1
                labels[hierarchy[i, 0]] = cluster_label
                labels[hierarchy[i, 1]] = cluster_label
            elif labels[hierarchy[i, 0]] != 0 and labels[hierarchy[i, 1]] == 0:
                labels[hierarchy[i, 1]] = labels[hierarchy[i, 0]]
            elif labels[hierarchy[i, 0]] == 0 and labels[hierarchy[i, 1]] != 0:
                labels[hierarchy[i, 0]] = labels[hierarchy[i, 1]]
            elif labels[hierarchy[i, 0]] != labels[hierarchy[i, 1]]:
                labels[labels == labels[hierarchy[i, 1]]] = labels[hierarchy[i, 0]]

        return labels



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--algorithm', type=str, default='DBSCAN')
args = parser.parse_args()
DatasetName = args.dataset
print(f'Dealing with {DatasetName}')
data_ori = np.loadtxt('data/'+DatasetName)
data = data_ori[:, :-1]
label = data_ori[:, -1]
# cluster_num = int(max(label))
datacount = data.shape[0]

distance = squareform(pdist(data, "euclidean"))
distanceAsc = np.sort(distance)
indexDistanceAsc = np.argsort(distance)
base_eps = np.sum(distanceAsc, axis=0)[1]/datacount

NMI_opt = 0
ARI_opt = 0
RI_opt = 0
t = 0

for i in range(1, 11, 1):
    t1 = time.perf_counter()
    if i != 10:
        hdbscan = HDBSCAN(min_cluster_size=int(i*datacount/10))
    else:
        hdbscan = HDBSCAN()
    result = hdbscan.fit_predict(data)
    t2 = time.perf_counter()

    NMI = normalized_mutual_info_score(
            labels_pred=result, labels_true=label)
    AMI = adjusted_mutual_info_score(labels_pred=result, labels_true=label)
    ARI = adjusted_rand_score(labels_pred=result, labels_true=label)
    RI = rand_score(labels_pred=result, labels_true=label)
    print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}")
    if NMI > NMI_opt:
        NMI_opt = NMI
        t = t2 - t1
    if ARI > ARI_opt:
        ARI_opt = ARI
    if RI > RI_opt:
        RI_opt = RI


fig, ax1 = plt.subplots(1)

fig.suptitle(f'optimal: NMI={NMI_opt:.4f},ARI={ARI_opt:.4f},RI={RI_opt:.4f},time(NMI)={t:.6f}')

fig.savefig(
    fr'./fig-pure-py/{DatasetName[:-4]}_RI.png')
