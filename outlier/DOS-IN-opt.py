from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, intersect1d, union1d, arange, argmax, bincount
from scipy.spatial.distance import pdist, squareform
from queue import Queue
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score
import argparse
import time
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DOS-IN'

    # for convinience, -1 means unassigned, 0 means noise
    DatasetName = args.dataset
    pathDataset = 'data/'+DatasetName
    if DatasetName[-1] == '1':
        k=3.5
    else:
        k=5
    figname = {'cluster_1': ('(A)', "(E)", "(I)"), 'cluster_2': ('(B)', '(F)', '(J)'), 'cluster_3': ('(C)', '(G)', '(K)'), 'cluster_4': ('(D)', '(H)', '(L)')}
    unassigned = -1
    noise_label = 0

    # load data
    print(f"Dealing with {DatasetName}...")
    data: ndarray = loadtxt(pathDataset)
    dataCount = size(data, 0)
    print(data.shape)

    # compute distance matrix
    distance = squareform(pdist(data, "euclidean"))
    distanceAsc = sort(distance)
    indexDistanceAsc = argsort(distance)
    F = sum(distanceAsc, axis=0)/dataCount
    # note that the first element = 0 because it refers to itself

    dpi = 200
    fig, ax1 = plt.subplots(1)
    fig.suptitle(f'{figname[DatasetName][0]}', fontstyle='italic',size=20)
    ax1.scatter(data[:, 0], data[:, 1], s=40, c='dimgrey', marker='.')
    plt.show()
    fig.savefig(
        fr'./fig/{args.algorithm}_{DatasetName}_init.png', dpi=dpi, bbox_inches='tight')

    baseDelta = F[1]
    k_list = []
    NMI_list = []
    k_opt = 0
    NMI_opt = 0

    RI_list = []
    RI_opt = 0
    ARI_list = []
    ARI_opt = 0

    delta = k*baseDelta
    noise_ratio = 0.01
    print(
        f"baseDelta={baseDelta}, k={k}, delta={delta}, noise_ratio={0.01}")
    
    numNeighbor = np.sum(distanceAsc < delta, axis=1)
    
    radius = full((dataCount, dataCount), -1.0)

    def cal_CN(array1, array2):
        set1 = set(array1.tolist())
        set2 = set(array2.tolist())
        return len(set1 & set2) / len(set1 | set2)
    
    for i in range(dataCount):
        for j in indexDistanceAsc[i][:numNeighbor[i]]:
            if j == i or radius[i][j] != -1.0:
                continue
            K = (numNeighbor[i] + numNeighbor[j]) //2
            CN = cal_CN(indexDistanceAsc[i][:K], indexDistanceAsc[j][:K])
            radius[i][j] = radius[j][i] = delta * CN

    # propagation and cluster assignment
    clusterStart = time.time()
    ii = 0
    next_ii = 0
    typeFlag = 0
    q = Queue()
    cluster: ndarray = full(dataCount, unassigned)
    tmp_set: list = []
    tmp_set_len = 0
    while(ii < dataCount):
        q.put(ii)
        next_ii = ii + 1
        typeFlag = typeFlag+1
        cluster[ii] = typeFlag

        while(not q.empty()):
            jj = q.get()
            tmp_set.append(jj)
            tmp_set_len = tmp_set_len + 1
            for kk in indexDistanceAsc[jj][:numNeighbor[jj]]:
                if(distance[kk][jj] < radius[kk][jj] and cluster[kk] == unassigned):
                    cluster[kk] = typeFlag
                    if kk == next_ii:
                        next_ii = next_ii + 1
                    q.put(kk)

        if next_ii < dataCount:
            while(cluster[next_ii] != unassigned):
                next_ii = next_ii + 1
                if next_ii == dataCount:
                    break

        if(tmp_set_len < dataCount*noise_ratio):
            typeFlag = typeFlag-1
            for node in tmp_set:
                cluster[node] = noise_label

        ii = next_ii

        q.queue.clear()
        tmp_set.clear()
        tmp_set_len = 0

    dis2cluster = []
    print(f"number of cluster = {typeFlag}, noise = {sum(cluster == noise_label)}")

    for i in range(dataCount):
        if cluster[i] == noise_label:
            for j in range(dataCount):
                if cluster[indexDistanceAsc[i][j]] != noise_label:
                    dis2cluster.append(np.linalg.norm(data[i, :] - data[indexDistanceAsc[i][j], :], ord=2))
                    break

    dis2cluster = sort(dis2cluster)
    dpi = 200
    fig, ax1 = plt.subplots(1)
    fig.suptitle(f'{figname[DatasetName][1]}', fontstyle='italic',size=20)
    ax1.scatter(list(range(len(dis2cluster))), dis2cluster, s=40, marker='.', c='dimgrey')
    ax1.plot(list(range(len(dis2cluster))), [1.2]*len(dis2cluster), linestyle='--', linewidth=2)
    ax1.set_ylabel(r'$\eta$', fontstyle='italic', size=20)
    ax1.set_xticks([])
    plt.show()
    fig.savefig(
        fr'./fig/{args.algorithm}_{DatasetName}_dg.png', dpi=dpi, bbox_inches='tight')
    
    threshold_dis = float(input())
    cluster_2 = full(dataCount, 0)
    for i in range(dataCount):
        if cluster[i] == noise_label:
            for j in range(dataCount):
                if cluster[indexDistanceAsc[i][j]] != noise_label:
                    if np.linalg.norm(data[i, :] - data[indexDistanceAsc[i][j], :], ord=2) <= threshold_dis:
                        cluster_2[i] = cluster[indexDistanceAsc[i][j]]
                        break
    cluster = cluster+cluster_2

    n_cluster = max(cluster)

    # plot figure
    color_list = [  '#696969',  # dimgrey
                    '#319DA0',
                    '#FFA500', 
                    '#D61C4E',
                    '#6b76ff', 
                ]
    dpi = 200
    fig, ax1 = plt.subplots(1)
    fig.suptitle(f'{figname[DatasetName][2]}', fontstyle='italic',size=20)
    for mx in range(n_cluster+1):
        idx = np.where(cluster == mx)
        if mx == 0:
            ax1.scatter(data[idx, 0], data[idx, 1], s=120, c=color_list[mx], marker='.')
        else:
            ax1.scatter(data[idx, 0], data[idx, 1], s=40, c=color_list[mx], marker='.')
    plt.show()
    fig.savefig(
        fr'./fig/{args.algorithm}_{DatasetName}.png', dpi=dpi, bbox_inches='tight')
