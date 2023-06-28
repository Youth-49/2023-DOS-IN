from dis import dis
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
vars(args)['algorithm'] = 'DPC'

DatasetName = args.dataset
pathDataset = 'data/'+DatasetName
print(f'input file name: {DatasetName}')

input_file = []
N_node = 0
print('Reading input distance matrix')

data = np.loadtxt(pathDataset)
N_node = np.size(data, 0)
label = data[:, -1]
data = data[:, :-1]
dist = squareform(pdist(data, "euclidean"))

t1 = time.perf_counter()
percent = 2.0

print('average percentage of neighbours (hard coded):\n', percent)

M = N_node*(N_node-1)/2
position = round(M*percent/100)+N_node
print(sum(dist.flatten() == 0))
print(position)
# ascend
sorted_dist = sorted(dist.flatten())
dc = sorted_dist[position]

print('Computing Rho with gaussian kernel of radius:\n', dc)

# Gaussian kernel
rho = np.zeros(N_node)
for i in range(N_node-1):
    for j in range(i+1, N_node):
        rho[i] = rho[i]+np.exp(-(dist[i][j]/dc)*(dist[i][j]/dc))
        rho[j] = rho[j]+np.exp(-(dist[i][j]/dc)*(dist[i][j]/dc))

# descend
maxd = np.max(dist)
ordrho = sorted(range(rho.shape[0]), key=lambda x: rho[x], reverse=True)
rho_sorted = sorted(rho, reverse=True)
delta = np.zeros(N_node)
nneigh = np.zeros(N_node)
delta[ordrho[0]] = -1.0
nneigh[ordrho[0]] = 0

for i in range(1, N_node):
    delta[ordrho[i]] = maxd
    for j in range(i):
        if dist[ordrho[i]][ordrho[j]] < delta[ordrho[i]]:
            delta[ordrho[i]] = dist[ordrho[i]][ordrho[j]]
            nneigh[ordrho[i]] = ordrho[j]

delta[ordrho[0]] = np.max(delta)
nneigh = nneigh.astype(np.int16)

t2 = time.perf_counter()

print('Decision graph:')
plt.figure(dpi=100)
plt.scatter(rho, delta, s=5)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\delta$')
plt.savefig(fr'./fig/{args.algorithm}_{DatasetName[:-4]}_dg.png', dpi=600)
plt.show()

plt.clf()

ind = np.array(range(N_node))
gamma = np.zeros(N_node)
for i in range(N_node):
    gamma[i] = rho[i]*delta[i]

# plt.figure(dpi=100)
# plt.scatter(ind, gamma, s=5)
# plt.xlabel(r'$i$')
# plt.ylabel(r'$\gamma$')
# plt.show()


print('please input rhomin and deltamin: (separated by single space)')
rhomin, deltamin = input().split(' ')
t3 = time.perf_counter()
rhomin, deltamin = float(rhomin), float(deltamin)
N_cluster = 0
cluster = np.zeros(N_node)
inv_cluster = [0]

for i in range(N_node):
    cluster[i] = -1

for i in range(N_node):
    if (rho[i] > rhomin) and (delta[i] > deltamin):
        N_cluster = N_cluster+1
        cluster[i] = int(N_cluster)
        inv_cluster.append(i)

inv_cluster = np.array(inv_cluster).astype(np.int16)

print('number of clusters:\n', N_cluster)
for i in range(N_node):
    if cluster[ordrho[i]] == -1:
        cluster[ordrho[i]] = cluster[nneigh[ordrho[i]]]

cluster = cluster.astype(np.int16)

halo = np.zeros(N_node)
for i in range(N_node):
    halo[i] = cluster[i]

if N_cluster > 1:
    bord_rho = np.zeros(N_cluster+1)
    for i in range(N_node):
        for j in range(i+1, N_node):
            if (cluster[i] != cluster[j]) and (dist[i][j] <= dc):
                rho_ave = 0.5*(rho[i]+rho[j])
                bord_rho[cluster[i]] = max(bord_rho[cluster[i]], rho_ave)
                bord_rho[cluster[j]] = max(bord_rho[cluster[j]], rho_ave)

    for i in range(N_node):
        if rho[i] < bord_rho[cluster[i]]:
            halo[i] = 0


for i in range(1, N_cluster+1):
    nc, nh = 0, 0
    for j in range(N_node):
        if cluster[j] == i:
            nc = nc+1
        if halo[j] == i:
            nh = nh+1

    nc, nh = int(nc), int(nh)
    print(
        f'CLUSTER: {i} CENTER: {inv_cluster[i]} ELEMENTS: {nc} CORE: {nh} HALO: {nc-nh}')
t4 = time.perf_counter()
print('performing 2D MDS')
mds = manifold.MDS(max_iter=200, eps=1e-4, n_init=1,
                   dissimilarity='precomputed')
dp_mds = mds.fit_transform(dist)
x = dp_mds[:, 0]
y = dp_mds[:, 1]
NMI = normalized_mutual_info_score(
    labels_pred=halo, labels_true=label)
AMI = adjusted_mutual_info_score(labels_pred=halo, labels_true=label)
ARI = adjusted_rand_score(labels_pred=halo, labels_true=label)
RI = rand_score(labels_pred=halo, labels_true=label)
print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}, RI = {RI}, t = {t4-t3+t2-t1}")
plt.figure(dpi=100)
plt.scatter(x, y, s=3, c=halo, cmap='rainbow')
plt.title(fr'percent={percent}, rhomin={rhomin}, deltamin={deltamin}, cluster={N_cluster}, noise={sum(halo == 0)}'+'\n'+f'NMI={NMI:.4f}, AMI={AMI:.4f}, ARI={ARI:.4f}, RI={RI:.4f}, t={t4-t3+t2-t1:.6f}')
plt.legend()
plt.savefig(fr'./fig-time/{args.algorithm}_{DatasetName[:-4]}.png', dpi=600)
plt.show()
