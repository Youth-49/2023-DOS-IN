import hdbscan
import numpy as np
from sklearn import metrics
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
# cluster_num = int(max(label))
datacount = data.shape[0]
print(datacount)

NMI_opt = 0
ARI_opt = 0
RI_opt = 0
t = 0

for i in range(1, 11, 1):
    t1 = time.perf_counter()
    if i != 10:
        # print(int(i*datacount/10))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(i*datacount/10))
    else:
        clusterer = hdbscan.HDBSCAN()
    result = clusterer.fit_predict(data)
    t2 = time.perf_counter()
    NMI = metrics.normalized_mutual_info_score(label, result)
    ARI = metrics.adjusted_rand_score(label, result)
    RI = metrics.rand_score(label, result)

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
    fr'./fig/by1_{DatasetName[:-4]}.png',)
