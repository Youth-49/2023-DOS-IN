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
    unassigned = -1
    noise_label = 0

    # load data
    print(f"Dealing with {DatasetName}...")
    data: ndarray = loadtxt(pathDataset)
    dataCount = size(data, 0)
    label = data[:, -1]
    data = data[:, :-1]

    # compute distance matrix
    distance = squareform(pdist(data, "euclidean"))
    distanceAsc = sort(distance)
    indexDistanceAsc = argsort(distance)
    F = sum(distanceAsc, axis=0)/dataCount
    # note that the first element = 0 because it refers to itself

    baseDelta = F[1]
    k_list = []
    NMI_list = []
    k_opt = 0
    NMI_opt = 0

    RI_list = []
    RI_opt = 0
    k_RI_opt = 0
    ARI_list = []
    ARI_opt = 0
    t = 0

    for k in arange(1, 11, 1):
        if DatasetName == 'digits.txt' and k > 2:
            break
        t1 = time.perf_counter()

        ## add time for cal distance
        distance = squareform(pdist(data, "euclidean"))
        distanceAsc = sort(distance)
        indexDistanceAsc = argsort(distance)
        ##

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

        print(time.perf_counter() - t1)
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

        cluster_2 = full(dataCount, 0)
        for i in range(dataCount):
            if cluster[i] == noise_label:
                for j in range(dataCount):
                    if cluster[indexDistanceAsc[i][j]] != noise_label:
                        cluster_2[i] = cluster[indexDistanceAsc[i][j]]
                        break
        cluster = cluster+cluster_2

        t2 = time.perf_counter()
        print(f"number of cluster = {typeFlag}, noise = {sum(cluster == noise_label)}")

        NMI = normalized_mutual_info_score(
            labels_pred=cluster, labels_true=label)
        AMI = adjusted_mutual_info_score(labels_pred=cluster, labels_true=label)
        ARI = adjusted_rand_score(labels_pred=cluster, labels_true=label)
        RI = rand_score(labels_pred=cluster, labels_true=label)
        print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}")
        k_list.append(k)
        NMI_list.append(NMI)
        if NMI > NMI_opt:
            NMI_opt = NMI
            k_opt = k
            t = t2-t1
        if RI > RI_opt:
            RI_opt = RI
            k_RI_opt = k
        if ARI > ARI_opt:
            ARI_opt = ARI

    # plot figure
    dpi = 100
    fig, ax1 = plt.subplots(1)
    fig.suptitle(
        f'NMI_opt={NMI_opt:.4f}, k_NMI_opt={k_opt:.4f}, RI={RI_opt:.4f}, k_RI_opt={k_RI_opt:.4f}, ARI={ARI_opt:.4f}, t={t:.6f}')
    ax1.scatter(k_list, NMI_list, s=3, c=NMI_list, cmap='rainbow')
    # plt.show()
    fig.savefig(
        fr'./fig-opt/{args.algorithm}_by1_{DatasetName[:-4]}.png', dpi=dpi)
