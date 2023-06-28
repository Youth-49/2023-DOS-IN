from sklearn.datasets import load_iris, load_wine

import numpy as np
from sklearn import metrics
from DPCKNN import DPCKNN
import matplotlib.pyplot as plt
import argparse
import time

p_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.06]

knn = DPCKNN()

# wine =load_wine()
# data=wine.data
# labels=wine.target
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
name = args.dataset
print(f'Dealing with {name}')
data = np.loadtxt('data/'+name)
labels = data[:, -1]
# print(labels)
data = data[:, :-1]
# print(data)
# y = knn.fit_predict(data)

k_list = []
NMI_list = []
k_opt = 0
NMI_opt = 0

ARI_list = []
ARI_opt = 0
RI_list = []
RI_opt = 0

for p in p_range:
    t1 = time.perf_counter()
    knn = DPCKNN()
    knn.set_param(p)
    y, t3 = knn.fit_predict(data)
    t2 = time.perf_counter()
    NMI = metrics.normalized_mutual_info_score(y, labels)
    ARI = metrics.adjusted_rand_score(y, labels)
    RI = metrics.rand_score(y, labels)
    print("p: {} \n NMI: {} \n ------------------".format(p, NMI))
    k_list.append(p)
    NMI_list.append(NMI)
    if NMI > NMI_opt:
        NMI_opt = NMI
        k_opt = p
        t = t2-t1-t3
    if RI > RI_opt:
        RI_opt = RI
    if ARI > ARI_opt:
        ARI_opt = ARI

dpi = 600
# fig, ax1 = plt.subplots(1)
# # ax1.set_aspect('equal')
# fig.suptitle(
#     f'NMI_opt={NMI_opt:.4f}, p_opt={k_opt:.4f}')
# ax1.scatter(k_list, NMI_list, s=3, c=NMI_list, cmap='rainbow')
# for i in range(len(k_list)):
#     ax1.text(k_list[i], NMI_list[i], f'{NMI_list[i]:.2f}', size=4)
# # plt.show()
# fig.savefig(
#     fr'./fig/byp_{name[:-4]}.png', dpi=dpi)

fig, ax1 = plt.subplots(1)
# ax1.set_aspect('equal')
fig.suptitle(
    f'NMI_opt={NMI_opt:.4f}, ARI_opt={ARI_opt:.4f}, RI_opt={RI_opt:.4f}, t={t:.6f}')
ax1.scatter(k_list, NMI_list, s=3, c=NMI_list, cmap='rainbow')
fig.savefig(
    fr'./fig/byp_{name[:-4]}.png', dpi=dpi)
