from numpy import array, empty, ndarray, loadtxt, size, argsort, sort, sum, full, array, exp, intersect1d, union1d, arange, ceil
from scipy.spatial.distance import pdist, squareform
from queue import Queue
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score
from sklearn.cluster import DBSCAN
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DBSCAN'
    
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
    print(f'base_eps = {base_eps}')
    NMI_list = []
    AMI_list = []
    ARI_list = []
    RI_list = []
    t_list = []
    eps_list = []
    minpt_list = []
    for eps in arange(base_eps, 11*base_eps, base_eps):
        for minpt in range(1, 11, 1):
            t1 = time.perf_counter()
            db = DBSCAN(eps=eps, min_samples=minpt).fit(data)
            cluster = db.labels_
            t2 = time.perf_counter()
            # calculate some index
            NMI = normalized_mutual_info_score(
                labels_pred=cluster, labels_true=label)
            AMI = adjusted_mutual_info_score(labels_pred=cluster, labels_true=label)
            ARI = adjusted_rand_score(labels_pred=cluster, labels_true=label)
            RI = rand_score(labels_pred=cluster, labels_true=label)
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
    k_list = array(eps_list)
    NMI_list = array(NMI_list)
    NMI_optimal = -0.1
    minpt_optimal = 0
    k_optimal = 0

    RI_list = array(RI_list)
    RI_optimal = 0
    ARI_list = array(ARI_list)
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
        fr'./result/{args.algorithm}_by1_{DatasetName[:-4]}.png', dpi=dpi)
