from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, intersect1d, union1d, arange, argmax, bincount
from scipy.spatial.distance import pdist, squareform
from queue import Queue
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import argparse
import time

if __name__ == '__main__':

    # one loop for noise, in t48k
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DOS-IN-16'

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

    for k in arange(0.625, 10.1, 0.625):
        delta = k*baseDelta
        noise_ratio = 0.01
        print(
            f"baseDelta={baseDelta}, k={k}, delta={delta}, noise_ratio={0.01}")

        # calculate rho, neighborhood, KNN, CN and radius
        indexNeighbor = array([indexDistanceAsc[rowid][distanceAsc[rowid] < delta]
                            for rowid in range(dataCount)], dtype=object)

        radius = full((dataCount, dataCount), 0.0)
        for i in range(dataCount):
            for j in indexNeighbor[i]:
                if j == i:
                    continue
                K = (indexNeighbor[i].shape[0] + indexNeighbor[j].shape[0])//2
                intersectSet: ndarray = intersect1d(
                    indexDistanceAsc[i][:K], indexDistanceAsc[j][:K], assume_unique=True)
                unionSet: ndarray = union1d(
                    indexDistanceAsc[i][:K], indexDistanceAsc[j][:K])
                CN = intersectSet.size/unionSet.size
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

        # 2nd assign
        cluster_2 = full(dataCount, 0)
        for i in range(dataCount):
            if cluster[i] == noise_label:
                for j in range(dataCount):
                    if cluster[indexDistanceAsc[i][j]] != noise_label:
                        cluster_2[i] = cluster[indexDistanceAsc[i][j]]
                        break
        cluster = cluster+cluster_2

        print(f"number of cluster = {typeFlag}, noise = {sum(cluster == noise_label)}")

        # for i in range(dataCount):
        #     if cluster[i] == noise_label:
        #         cluster[i] = -i

        NMI = normalized_mutual_info_score(
            labels_pred=cluster, labels_true=label)
        AMI = adjusted_mutual_info_score(
            labels_pred=cluster, labels_true=label)
        ARI = adjusted_rand_score(
            labels_pred=cluster, labels_true=label)
        print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}")
        k_list.append(k)
        NMI_list.append(NMI)
        if NMI > NMI_opt:
            NMI_opt = NMI
            k_opt = k

    # write NMI
    file_w = open(f'res/{args.algorithm}_{DatasetName[:-4]}.txt', "w")
    for nmi in NMI_list:
        file_w.write(str(nmi)+'\n')

    # plot figure
    dpi = 600
    fig, ax1 = plt.subplots(1)
    # ax1.set_aspect('equal')
    fig.suptitle(
        f'NMI_opt={NMI_opt:.4f}, k_opt={k_opt:.4f}')
    ax1.scatter(k_list, NMI_list, s=3, c=NMI_list, cmap='rainbow')
    for i in range(len(k_list)):
        ax1.text(k_list[i], NMI_list[i], f'{NMI_list[i]:.2f}', size=4)
    # plt.show()
    fig.savefig(
        fr'./fig/{args.algorithm}_by16_{DatasetName[:-4]}.png', dpi=dpi)
