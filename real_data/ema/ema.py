import numpy as np
from sklearn import metrics
from pyclustering.cluster.ema import ema, ema_visualizer
import matplotlib.pyplot as plt
import argparse
import time

#改改改改改改改

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
t1 = time.perf_counter()
ema_instance = ema(data, max_clusters)
# Run clustering process.
ema_instance.process()
# Get clustering results.
clusters = ema_instance.get_clusters()
t2 = time.perf_counter()
# covariances = ema_instance.get_covariances()
# means = ema_instance.get_centers()
# Visualize obtained clustering results.
# ema_visualizer.show_clusters(clusters, data, covariances, means)
result = np.zeros(data.shape[0])
clusters=np.array(clusters)

#改改改改改
for i in range(clusters.shape[0]):
    for j in clusters[i]:
        result[j]=i

NMI = metrics.normalized_mutual_info_score(label, result)
RI = metrics.rand_score(label, result)
ARI = metrics.adjusted_rand_score(label, result)
t = t2-t1

k_list = [max_clusters]
NMI_list = [NMI]
dpi = 600
fig, ax1 = plt.subplots(1)
ax1.scatter(k_list, NMI_list, c=NMI_list, cmap='rainbow', s=4)

fig.suptitle(f'optimal: num_cluster={max_clusters:.2f},NMI={NMI:.4f},RI={RI:.4f},ARI={ARI:.4f},t={t:.6f}')

fig.savefig(
    fr'./fig/fix_{DatasetName[:-4]}.png', dpi=dpi)
