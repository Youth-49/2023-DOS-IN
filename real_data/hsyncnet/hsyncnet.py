import numpy as np
from sklearn import metrics
from pyclustering.cluster.hsyncnet import hsyncnet
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
cluster_num = int(max(label))

print(data.shape, cluster_num)


# Get allocated clusters.
k_list = []
NMI_list = []
k_opt = 0
NMI_opt = -0.1

ARI_opt = 0
RI_opt = 0

for z in range(1,11,1):
    if DatasetName in {'digits.txt', 'skewed.txt'} and z >= 2:
        break
    t1 = time.perf_counter()
    network = hsyncnet(data, cluster_num)
    # Run cluster analysis and output dynamic of the network.
    # analyser = network.process(0.995, collect_dynamic=True)
    analyser = network.process() # default parameter setting
    clusters = analyser.allocate_clusters(eps=0.1*z)
    t2 = time.perf_counter()
    print(f'z = {z}')
    # # Show output dynamic of the network.
    # sync_visualizer.show_output_dynamic(analyser)
    # # Show allocated clusters.
    # draw_clusters(data, clusters)
    result = np.zeros(data.shape[0])
    clusters = np.array(clusters)
    for i in range(clusters.shape[0]):
        for j in clusters[i]:
            result[j] = i
    NMI = metrics.normalized_mutual_info_score(label, result)
    ARI = metrics.adjusted_rand_score(label, result)
    RI = metrics.rand_score(label, result)
    k_list.append(z)
    NMI_list.append(NMI)
    if NMI > NMI_opt:
        k_opt = z
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


