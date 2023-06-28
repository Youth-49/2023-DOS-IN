import numpy as np
from  Extreme_Clustering import Extreme_Clustreing
from  Visualization import Visualization
from scipy.spatial.distance import pdist,squareform
from numpy import sort, argsort, arange, sum
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
DatasetName = args.dataset
data = np.loadtxt("data/"+DatasetName)
label = data[:,-1]
data = data[:,:-1]
dist= squareform(pdist(data,metric='euclidean'))
distanceAsc = sort(dist)
F = sum(distanceAsc, axis=0)/data.shape[0]
base = F[1]
k_list = []
NMI_list = []
k_opt = 0
NMI_opt = 0

ARI_opt = 0
RI_opt = 0
for k in arange(1, 11, 1):
    t1 = time.perf_counter()
    delta = k*base
    print(f'k = {k}, delta = {delta}')
    clusteringResult = Extreme_Clustreing(data, delta, False)
    t2 = time.perf_counter()
    NMI, ARI, RI = Visualization(DatasetName, data, label, clusteringResult, False)
    k_list.append(k)
    NMI_list.append(NMI)
    if NMI > NMI_opt:
        NMI_opt = NMI
        k_opt = k
        t = t2-t1
    if ARI > ARI_opt:
        ARI_opt = ARI
    if RI > RI_opt:
        RI_opt = RI

    print(f'NMI = {NMI}\n')

dpi = 600
fig, ax1 = plt.subplots(1)
# ax1.set_aspect('equal')
fig.suptitle(
    f'NMI_opt={NMI_opt:.4f}, k_opt={k_opt:.4f}, ARI={ARI_opt:.4f}, RI={RI_opt:.4f}, t={t:.6f}')
ax1.scatter(k_list, NMI_list, s=3, c=NMI_list, cmap='rainbow')
for i in range(len(k_list)):
    ax1.text(k_list[i], NMI_list[i], f'{NMI_list[i]:.2f}', size=4)
# plt.show()
fig.savefig(
    fr'./fig/by1_{DatasetName[:-4]}.png', dpi=dpi)