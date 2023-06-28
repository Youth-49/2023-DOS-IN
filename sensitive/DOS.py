from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, exp, arange
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DOS-16'

    # for convinience, -1 means unassigned, 0 means noise
    DatasetName = args.dataset
    print(f"Dealing with {DatasetName}...")
    pathDataset = "data/"+DatasetName
    unassigned = -1
    noise_label = 0

    # load data
    data: ndarray = loadtxt(pathDataset)
    dataCount = size(data, 0)
    label = data[:, -1]
    data = data[:, :-1]
    # compute distance matrix
    distance = squareform(pdist(data, "euclidean"))
    indexDistanceAsc = argsort(distance)
    # compute F
    distanceAsc = sort(distance)
    # note that the first element = 0 because it refers to itself
    F = sum(distanceAsc, axis=0)/dataCount

    K_list = []
    l_list = []
    NMI_list = []
    K_opt = 0
    l_opt = 0
    NMI_opt = 0

    # set parameters
    baseDelta = F[1]
    for l in arange(2.5, 10.1, 2.5):
        for K in arange(2.5, 10.1, 2.5):
            delta = baseDelta*l
            noise_ratio = 0.01
            print(f"baseDelta={baseDelta}, delta={delta}, K={K}")
            # compute rho, mu and std
            rho = array([len(arr[arr < delta])-1 for arr in distance])
            mu = rho.mean()
            sigma = rho.std()  # 总体方差
            radius = array([delta if n_i >= mu-sigma
                            else K*delta*(1-exp((n_i-mu)/(sigma)))/(1+exp((n_i-mu)/(sigma))) for n_i in rho])

            # assign
            cluster = full(dataCount, unassigned)
            num_cluster = 0
            for i in range(dataCount):
                if (cluster[i] == unassigned):
                    num_cluster = num_cluster + 1
                    cluster[i] = num_cluster

                for j in range(1, dataCount):
                    if (distanceAsc[i][j] < radius[i]):
                        if (cluster[indexDistanceAsc[i][j]] == unassigned):
                            cluster[indexDistanceAsc[i][j]] = cluster[i]
                        elif (cluster[i] != cluster[indexDistanceAsc[i][j]]):
                            x = max(cluster[i], cluster[indexDistanceAsc[i][j]])
                            y = min(cluster[i], cluster[indexDistanceAsc[i][j]])
                            num_cluster = num_cluster-1
                            for kk in range(dataCount):
                                if (cluster[kk] == x):
                                    cluster[kk] = y
                                elif (cluster[kk] > x):
                                    cluster[kk] = cluster[kk]-1

                    else:
                        break

            N = num_cluster
            i = 1
            while (i <= num_cluster):
                if (sum(cluster == i) < noise_ratio*dataCount):
                    num_cluster = num_cluster - 1
                    for j in range(dataCount):
                        if (cluster[j] == i):
                            cluster[j] = noise_label
                        if (cluster[j] > i):
                            cluster[j] = cluster[j] - 1
                else:
                    i = i + 1

            for i in range(dataCount):
                if cluster[i] == noise_label:
                    cluster[i] = -i

            NMI = normalized_mutual_info_score(
                labels_pred=cluster, labels_true=label)
            AMI = adjusted_mutual_info_score(
                labels_pred=cluster, labels_true=label)
            ARI = adjusted_rand_score(
                labels_pred=cluster, labels_true=label)
            print(
                f"number of cluster = {num_cluster}, noise = {sum(cluster <= noise_label)}")
            print(f"NMI = {NMI}, AMI = {AMI}")
            K_list.append(K)
            l_list.append(l)
            NMI_list.append(NMI)
            if NMI > NMI_opt:
                NMI_opt = NMI
                K_opt = K
                l_opt = l

    # write NMI
    file_w = open(f'res/{args.algorithm}_{DatasetName[:-4]}.txt', "w")
    for nmi in NMI_list:
        file_w.write(str(nmi)+'\n')

    # plot figure
    dpi = 600
    fig, ax1 = plt.subplots(1)
    # ax1.set_aspect('equal')
    fig.suptitle(
        f'NMI_opt={NMI_opt:.4f}, l_opt={l_opt}, K_opt={K_opt:.4f}')
    ax1.scatter(l_list, K_list, s=3, c=NMI_list, cmap='rainbow')
    for i in range(len(l_list)):
        for j in range(len(K_list)):
            ax1.text(l_list[i], K_list[i], f'{NMI_list[i]:.2f}', size=4)

    # plt.show()
    fig.savefig(
        fr'./fig/{args.algorithm}_by16_{DatasetName[:-4]}.png', dpi=dpi)
