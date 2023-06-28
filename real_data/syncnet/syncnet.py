import numpy as np
from pyclustering.cluster.syncnet import syncnet, solve_type
from sklearn import metrics
import scipy.spatial.distance as dis
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

print(data.shape)

distance = dis.pdist(data)
distance_matrix = dis.squareform(distance)
distance_sort = np.sort(distance_matrix, axis=1)
eps_original = np.mean(distance_sort[:,1])
k_list = []
NMI_list = []
k_opt = 0
NMI_opt = -0.1
ARI_opt = 0
RI_opt = 0

for i in range(1,11,1):
    t1 = time.perf_counter()
    radiu = eps_original*i
    print(f'i = {i}, radius = {radiu}')
    # Create oscillatory network with connectivity radius 1.0.
    network = syncnet(data, radiu)
    # Run cluster analysis and collect output dynamic of the oscillatory network.
    # Network simulation is performed by Runge Kutta 4.
    # analyser = network.process(0.998, solve_type.RK4)
    analyser = network.process()
    # Show oscillatory network.
    # network.show_network()
    # Obtain clustering results.
    clusters = analyser.allocate_clusters()
    t2 = time.perf_counter()
    clusters = np.array(clusters)
    result = np.zeros(data.shape[0])
    # 改改改改改
    for j in range(clusters.shape[0]):
        for z in clusters[j]:
            result[z] = j
    NMI = metrics.normalized_mutual_info_score(label, result)
    ARI = metrics.adjusted_rand_score(label, result)
    RI = metrics.rand_score(label, result)
    k_list.append(i)
    NMI_list.append(NMI)
    if NMI > NMI_opt:
        k_opt = i
        NMI_opt = NMI
        t = t2-t1
    if ARI > ARI_opt:
        ARI_opt = ARI
    if RI > RI_opt:
        RI_opt = RI


dpi = 600
fig, ax1 = plt.subplots(1)
ax1.scatter(k_list, NMI_list, c=NMI_list, cmap='rainbow', s=4)

fig.suptitle(f'optimal: k={k_opt:.2f},NMI={NMI_opt:.4f},ARI={ARI_opt:.4f},RI={RI_opt:.4f},t={t:.6f}')

fig.savefig(
    fr'./fig/by1_{DatasetName[:-4]}.png', dpi=dpi)