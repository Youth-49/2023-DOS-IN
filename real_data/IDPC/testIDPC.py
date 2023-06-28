import numpy as np
from IDPC import IDPC
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, rand_score, adjusted_rand_score
from time import perf_counter

p_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.48]

idpc = IDPC()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
name = args.dataset
print(f'Dealing with {name}')
data = np.loadtxt('data/'+name)
labels = data[:, -1]
data = data[:, :-1]
nc = int(max(labels))
k_list = []
NMI_list = []
k_opt = 0
NMI_opt = -0.1
RI_list = []
RI_opt = 0
ARI_list = []
ARI_opt = 0
t = 0

for p in p_range:
    t1 = perf_counter()
    idpc.set_param(nc, p)
    y = idpc.fit_predict(data)
    t2 = perf_counter()
    NMI = normalized_mutual_info_score(y, labels)
    AMI = adjusted_mutual_info_score(labels_pred=y, labels_true=labels)
    ARI = adjusted_rand_score(labels_pred=y, labels_true=labels)
    RI = rand_score(labels_pred=y, labels_true=labels)
    print(f'p = {p}, NMI = {NMI}')
    k_list.append(p)
    NMI_list.append(NMI)
    if NMI > NMI_opt:
        NMI_opt = NMI
        k_opt = p
        t = t2-t1
    if RI > RI_opt:
        RI_opt = RI
    if ARI > ARI_opt:
        ARI_opt = ARI

dpi = 600
fig, ax1 = plt.subplots(1)
# ax1.set_aspect('equal')
fig.suptitle(f'NMI_opt={NMI_opt:.4f}, p_opt={k_opt:.4f}, RI={RI_opt:.4f}, ARI={ARI_opt:.4f}, t={t:.6f}')
ax1.scatter(k_list, NMI_list, s=3, c=NMI_list, cmap='rainbow')
for i in range(len(k_list)):
    ax1.text(k_list[i], NMI_list[i], f'{NMI_list[i]:.2f}', size=4)
# plt.show()
fig.savefig(
    fr'./fig/byp_{name[:-4]}.png', dpi=dpi)
