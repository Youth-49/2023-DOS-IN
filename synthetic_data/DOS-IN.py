from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, intersect1d, union1d, where
from scipy.spatial.distance import pdist, squareform
from queue import Queue
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DOS-IN'

    parameter = {"R15.tsv": 7,
                 "unbalance.tsv": 30, "Flame.tsv": 3.1,
                 "D31.txt": 3.65, "Spiral.tsv": 6,
                 "a1.tsv": 4.935, "a2.tsv": 4.55, "a3.tsv": 4.3, "s1.tsv": 9.95,
                 "s2.tsv": 4, "s3.tsv": 2.8, "s4.tsv":3.1, "unbalance2.txt": 30, 
                 "asymmeteric.txt": 7.5, "skewed.txt": 8.5}

    # for convinience, -1 means unassigned, 0 means noise
    DatasetName = 's4.tsv'
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

    k = parameter[DatasetName]
    baseDelta = F[1]
    delta = k*baseDelta
    noise_ratio = 0.01
    print(
        f"baseDelta={baseDelta}, k={k}, delta={delta}, noise_ratio={0.01}")

    numNeighbor = sum(distanceAsc < delta, axis=1)

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
    
    # 2nd assign
    cluster_2 = full(dataCount, 0)
    for i in range(dataCount):
        if cluster[i] == noise_label:
            for j in range(dataCount):
                if cluster[indexDistanceAsc[i][j]] != noise_label:
                    cluster_2[i] = cluster[indexDistanceAsc[i][j]]
                    break
    cluster = cluster+cluster_2
            
    cluster_num = typeFlag
    print(f"number of cluster = {cluster_num}, noise = {sum(cluster == noise_label)}")

    NMI = normalized_mutual_info_score(
        labels_pred=cluster, labels_true=label)
    AMI = adjusted_mutual_info_score(labels_pred=cluster, labels_true=label)
    ARI = adjusted_rand_score(labels_pred=cluster, labels_true=label)
    print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}")

    # plot figure
    cValue = ['#319DA0','#293462','#D61C4E',"#21E6A1",'#AC4425','#FEDB39','#1CD6CE','#781C68','#224B0C','#F94892',
                "#A20650", "#0b409c", "#A6fAd2", "#6b76ff"]
    color_num = len(cValue)
    plt.figure(1)
    idx_0 = where(cluster == 0)
    plt.scatter(data[idx_0, 0], data[idx_0, 1], s=20, marker='.', c="dimgrey")
    for mx in range(1, cluster_num+1):
        color = mx % color_num
        idx_0 = where(cluster== mx)
        if DatasetName in ["dim128.tsv", "dim256.tsv", "dim512.tsv", "dim1024.tsv"]:
            plt.scatter(data[idx_0, 0], data[idx_0, 1], s=120, marker='.', c=cValue[color])
        else:
            plt.scatter(data[idx_0, 0], data[idx_0, 1], s=20, marker='.', c=cValue[color])
    plt.show()