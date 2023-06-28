from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, intersect1d, union1d, arange, argmax, bincount
from scipy.spatial.distance import pdist, squareform
from queue import Queue
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score
import argparse
import time
from sklearn.neighbors import KDTree
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
    ARI_list = []
    ARI_opt = 0
    t = 0

    for k in arange(1, 11, 1):
        if DatasetName == 'digits.txt' and k > 2:
            break

        delta = k*baseDelta
        noise_ratio = 0.01
        print(
            f"baseDelta={baseDelta}, k={k}, delta={delta}, noise_ratio={0.01}")
        
        t1 = time.perf_counter()
        indexNeighbor = array([indexDistanceAsc[rowid][distanceAsc[rowid] < delta]
                            for rowid in range(dataCount)], dtype=object)
        numNeighbor = array([np.sum(distanceAsc[rowid] < delta)
                            for rowid in range(dataCount)])

        radius = full((dataCount, dataCount), 0.0)
        def cal_CN(array1, array2):
            set1 = set(array1.tolist())
            set2 = set(array2.tolist())
            return len(set1 & set2) / len(set1 | set2)
        
        for i in range(dataCount):
            for j in indexDistanceAsc[i][:numNeighbor[i]]:
                if j == i:
                    continue
                K = (numNeighbor[i] + numNeighbor[j]) //2
                # intersectSet: ndarray = intersect1d(
                #     indexDistanceAsc[i][:K], indexDistanceAsc[j][:K], assume_unique=True)
                # unionSet: ndarray = union1d(
                #     indexDistanceAsc[i][:K], indexDistanceAsc[j][:K])
                # CN = intersectSet.size/unionSet.size
                CN = cal_CN(indexDistanceAsc[i][:K], indexDistanceAsc[j][:K])
                radius[i][j] = radius[j][i] = delta * CN

        # propagation and cluster assignment
        clusterStart = time.time()
        ii = 0
        typeFlag = 0
        q = Queue()
        cluster: ndarray = full(dataCount, unassigned)
        tmp_set: list = []
        tmp_set_len = 0
        reachable: ndarray = full((dataCount, dataCount), 0)
        while(ii < dataCount):
            q.put(ii)
            typeFlag = typeFlag+1
            cluster[ii] = typeFlag

            while(not q.empty()):
                jj = q.get()
                tmp_set.append(jj)
                tmp_set_len = tmp_set_len + 1
                for kk in range(ii+1, dataCount):
                    if(distance[kk][jj] < radius[kk][jj] and cluster[kk] == unassigned):
                        reachable[kk][jj] = typeFlag
                        cluster[kk] = typeFlag
                        q.put(kk)

            if(tmp_set_len < dataCount*noise_ratio):
                typeFlag = typeFlag-1
                for node in tmp_set:
                    cluster[node] = noise_label

            for nn in range(ii, dataCount):
                if (cluster[nn] == unassigned):
                    ii = nn
                    break
                elif (nn == dataCount-1):
                    ii = dataCount

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
        if ARI > ARI_opt:
            ARI_opt = ARI
        
        # if typeFlag==1 and sum(cluster == 1) == dataCount:
        #     break

    # plot figure
    dpi = 100
    fig, ax1 = plt.subplots(1)
    fig.suptitle(
        f'NMI_opt={NMI_opt:.4f}, k_opt={k_opt:.4f}, RI={RI_opt:.4f}, ARI={ARI_opt:.4f}, t={t:.6f}')
    ax1.scatter(k_list, NMI_list, s=3, c=NMI_list, cmap='rainbow')
    # plt.show()
    fig.savefig(
        fr'./fig/{args.algorithm}_by1_{DatasetName[:-4]}_RI.png', dpi=dpi)
