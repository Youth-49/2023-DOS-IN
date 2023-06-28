from numpy import array, empty, ndarray, loadtxt, size, argsort, sort, sum, full, array, exp, intersect1d, union1d, arange, ceil
from scipy.spatial.distance import pdist, squareform
from queue import Queue
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score
from sklearn.cluster import KMeans
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    vars(args)['algorithm'] = 'KMeans'
    
    # for convinience, -1 means unassigned, 0 means noise
    DatasetName = args.dataset
    pathDataset = 'data/'+DatasetName

    # load data
    print(f"Dealing with {DatasetName}...")
    loadDatasetStart = time.time()
    data: ndarray = loadtxt(pathDataset)
    loadDatasetEnd = time.time()
    print(f'loading dataset costs {loadDatasetEnd-loadDatasetStart}s')

    dataCount = size(data, 0)
    label = data[:, -1]
    data = data[:, :-1]

    distance = squareform(pdist(data, "euclidean"))
    distanceAsc = sort(distance)
    indexDistanceAsc = argsort(distance)
    base_eps = sum(distanceAsc, axis=0)[1]/dataCount
    NMI_list = []
    AMI_list = []
    ARI_list = []
    RI_list = []
    K_list = []
    K = int(max(label))
    print(K)
    t1 = time.perf_counter()
    res = KMeans(n_clusters=K, random_state=42).fit(data)
    cluster = res.labels_
    t2 = time.perf_counter()
    t = t2-t1
    # calculate some index
    calIndexStart = time.time()
    NMI = normalized_mutual_info_score(
        labels_pred=cluster, labels_true=label)
    AMI = adjusted_mutual_info_score(labels_pred=cluster, labels_true=label)
    ARI = adjusted_rand_score(labels_pred=cluster, labels_true=label)
    RI = rand_score(labels_pred=cluster, labels_true=label)
    calIndexEnd = time.time()
    print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}")
    print(f'calculating Index costs {calIndexEnd-calIndexStart}s')
    K_list.append(K)
    NMI_list.append(NMI)
    AMI_list.append(AMI)
    ARI_list.append(ARI)
    RI_list.append(RI)

    # plot figure
    N = len(K_list)
    NMI_list = array(NMI_list)
    ARI_list = array(ARI_list)
    RI_list = array(RI_list)
    NMI_optimal = 0
    minpt_optimal = 0
    K_optimal = 0

    ARI_optimal = 0
    RI_optimal = 0
    # plot figure
    plotStart = time.time()
    dpi = 600
    fig, ax1 = plt.subplots(1)
    ax1.scatter(K_list, NMI_list, c=NMI_list, cmap='rainbow', s=4)
    for i in range(N):
        ax1.text(K_list[i], NMI_list[i], f'{NMI_list[i]:.2f}', size=4)
        if (NMI_list[i] > NMI_optimal):
            NMI_optimal = NMI_list[i]
            K_optimal = K_list[i]

        if ARI_list[i] > ARI_optimal:
            ARI_optimal = ARI

        if RI_list[i] > RI_optimal:
            RI_optimal = RI

    fig.suptitle(f'optimal: K={K_optimal:.2f},NMI={NMI_optimal:.4f},ARI={ARI_optimal:.4f},RI={RI_optimal:.4f},t={t:.6f}')
    plotEnd = time.time()
    # plt.show()
    print(f'ploting costs {plotEnd-plotStart}s')

    fig.savefig(
        fr'./result/{args.algorithm}_fixK_sd=42_{DatasetName[:-4]}_RI.png', dpi=dpi)
