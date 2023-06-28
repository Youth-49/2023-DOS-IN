import numpy as np
import random
import matplotlib.pyplot as plt
import time
import argparse
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score

# https://medium.com/ntust-aivc/dbscan-a-common-clustering-algorithm-including-python-code-implementation-6948d6452a83

def findNeighbor(j, datas, eps):
    N = []
    for p in range(datas.shape[0]):
        distance = np.sqrt(np.sum(np.square(datas[j] - datas[p])))#find Euclidean distance
        if distance <= eps:
            N.append(p)#put the point in the "Neighbor" list
    return N


def dbscan(X, eps, min_pts):
    k = -1
    NeighborPts = []
    Ner_NeighborPts = []
    fil = []
    gama = [x for x in range(len(X))]
    cluster = [-1 for y in range(len(X))]
    while len(gama) > 0:#repeat until all the points are removed from gama
        j = random.choice(gama)#start with a random point
        gama.remove(j)
        fil.append(j)
        NeighborPts = findNeighbor(j, X, eps)
        if len(NeighborPts) < min_pts:
            cluster[j] = -1#mark this point as a noise point

        else:
            k = k + 1
            cluster[j] = k
            for i in NeighborPts:
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)
                    Ner_NeighborPts = findNeighbor(i, X, eps)
                    if len(Ner_NeighborPts) >= min_pts:
                        for a in Ner_NeighborPts:
                            if a not in NeighborPts:
                                NeighborPts.append(a)
                    if cluster[i] == -1:
                        cluster[i] = k
    return cluster


    
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

NMI_list = []
AMI_list = []
ARI_list = []
RI_list = []
t_list = []
eps_list = []
minpt_list = []

for eps in np.arange(base_eps, 11*base_eps, base_eps):
    for minpt in range(1, 11, 1):
        t1 = time.perf_counter()
        result = dbscan(data, eps=eps, min_pts=minpt)
        t2 = time.perf_counter()
        print(f'eps={eps}, minpt={minpt}, time={t2-t1}')

        NMI = normalized_mutual_info_score(
                labels_pred=result, labels_true=label)
        AMI = adjusted_mutual_info_score(labels_pred=result, labels_true=label)
        ARI = adjusted_rand_score(labels_pred=result, labels_true=label)
        RI = rand_score(labels_pred=result, labels_true=label)
        print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}")
        eps_list.append(eps)
        minpt_list.append(minpt)
        NMI_list.append(NMI)
        AMI_list.append(AMI)
        ARI_list.append(ARI)
        RI_list.append(RI)
        t_list.append(t2-t1)

# plot figure
N = len(eps_list)
k_list = np.array(eps_list)
NMI_list = np.array(NMI_list)
NMI_optimal = 0
minpt_optimal = 0
k_optimal = 0

RI_list = np.array(RI_list)
RI_optimal = 0
ARI_list = np.array(ARI_list)
ARI_optimal = 0
# plot figure
plotStart = time.time()
dpi = 600
fig, ax1 = plt.subplots(1)
ax1.scatter(eps_list, minpt_list, c=NMI_list, cmap='rainbow', s=4)
for i in range(N):
    ax1.text(k_list[i], minpt_list[i], f'{NMI_list[i]:.2f}', size=4)
    if (NMI_list[i] > NMI_optimal):
        NMI_optimal = NMI_list[i]
        k_optimal = k_list[i]
        minpt_optimal = minpt_list[i]
        t = t_list[i]
    if (ARI_list[i] > ARI_optimal):
        ARI_optimal = ARI_list[i]
    if (RI_list[i] > RI_optimal):
        RI_optimal = RI_list[i]

fig.suptitle(f'optimal: k={k_optimal:.2f},minpt={minpt_optimal:.2f},NMI={NMI_optimal:.4f},ARI={ARI_optimal:.4f},RI={RI_optimal:.4f},t={t:.6f}', size=8)
plotEnd = time.time()
# plt.show()
print(f'ploting costs {plotEnd-plotStart}s')

fig.savefig(
    fr'./result-pure-py/{args.algorithm}_by1_{DatasetName[:-4]}.png', dpi=dpi)