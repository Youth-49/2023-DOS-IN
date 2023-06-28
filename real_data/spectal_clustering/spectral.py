from calendar import c
import numpy as np
from sklearn import metrics
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
DatasetName = args.dataset
print(f'Dealing with {DatasetName}')
data_ori = np.loadtxt('data/'+DatasetName)
data = data_ori[:, :-1]
label = data_ori[:, -1]
cluster_num = int(max(label))

t1 = time.perf_counter()
# spectral = SpectralClustering(n_clusters=cluster_num,eigen_solver='arpack',affinity="nearest_neighbors")
spectral = SpectralClustering(n_clusters=cluster_num)

spectral.fit(data)
result = spectral.labels_
t2 = time.perf_counter()
t = t2-t1
NMI = metrics.normalized_mutual_info_score(label, result)
ARI = metrics.adjusted_rand_score(label, result)
RI = metrics.rand_score(label, result)

#改改改
k_list = [cluster_num]
NMI_list = [NMI]
dpi = 600
fig, ax1 = plt.subplots(1)
ax1.scatter(k_list, NMI_list, c=NMI_list, cmap='rainbow', s=4)

fig.suptitle(f'optimal: cluster_num={cluster_num:.2f},NMI={NMI:.4f},RI={RI:.4f},ARI={ARI:.4f},t={t:.6f}',size=10)

fig.savefig(
    fr'./fig/by1_{DatasetName[:-4]}.png', dpi=dpi)
