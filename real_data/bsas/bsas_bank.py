import numpy as np
from pyclustering.cluster.bsas import bsas, bsas_visualizer
from sklearn import metrics
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt
import argparse
import time

# 改改改改改改改
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
DatasetName = args.dataset
print(f'Dealing with {DatasetName}')
data_ori = np.loadtxt('data/'+DatasetName)
data = data_ori[:, :-1]
label = data_ori[:, -1]
max_clusters = int(max(label))

print(data.shape, max_clusters)

distance = dis.pdist(data)
distance_matrix = dis.squareform(distance)
distance_sort = np.sort(distance_matrix, axis=1)
eps_original = np.mean(distance_sort[:, 1])
k_list = []
NMI_list = []
k_opt = 0
NMI_opt = 0

RI_k_list = []
RI_list = []
RI_k_opt = 0
RI_opt = 0

ARI_k_list = []
ARI_list = []
ARI_k_opt = 0
ARI_opt = 0

for k in np.arange(1, 11, 1):
    t1 = time.perf_counter()
    threshold = eps_original*k
    print(f'k = {k}, threshold = {threshold}')
    # Create instance of BSAS algorithm.
    bsas_instance = bsas(data, max_clusters, threshold)
    bsas_instance.process()
    # Get clustering results.
    clusters = bsas_instance.get_clusters()
    # representatives = bsas_instance.get_representatives()
    # Display results.
    # bsas_visualizer.show_clusters(sample, clusters, representatives)
    t2 = time.perf_counter()
    clusters = np.array(clusters)
    result = np.zeros(data.shape[0])
    for j in range(clusters.shape[0]):
        for z in clusters[j]:
            result[z] = j
    NMI = metrics.normalized_mutual_info_score(label, result)
    RI = metrics.rand_score(label, result)
    ARI = metrics.adjusted_rand_score(label, result)
    
    k_list.append(k)
    NMI_list.append(NMI)
    if NMI > NMI_opt:
        k_opt = k
        NMI_opt = NMI
        t = t2-t1

    RI_k_list.append(k)
    RI_list.append(RI)
    if RI > RI_opt:
        RI_k_opt = k
        RI_opt = RI

    ARI_k_list.append(k)
    ARI_list.append(ARI)
    if ARI > ARI_opt:
        ARI_k_opt = k
        ARI_opt = RI

dpi = 200

fig, ax1 = plt.subplots(1)
ax1.scatter(RI_k_list, RI_list, c=RI_list, cmap='rainbow', s=4)

fig.suptitle(f'optimal: k={RI_k_opt:.2f},NMI={NMI_opt:.4f},RI={RI_opt:.4f},ARI={ARI_opt:.4f},t={t:.6f}')

fig.savefig(
    fr'./fig/by1_{DatasetName[:-4]}.png', dpi=dpi)
