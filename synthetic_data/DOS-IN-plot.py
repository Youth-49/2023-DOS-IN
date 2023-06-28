from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, intersect1d, union1d, where
from scipy.spatial.distance import pdist, squareform
from queue import Queue
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA

if __name__ == '__main__':

    # not set_aspect==equal

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DOS-IN'

    parameter = {
                "Flame.tsv": 3.1, "smile.txt": 8, "t4.8k(nonoise).txt": 6, "Spiral.tsv": 6,
                 "a1.tsv": 4.935, "a2.tsv": 4.55, "a3.tsv": 4.3, "D31.txt": 3.65,
                 "s1.tsv": 9.95, "s2.tsv": 4, "s3.tsv": 2.8, "s4.tsv":3.1, 
                 "asymmetric.txt": 7.5, "skewed.txt": 8.5, 
                 "dim128.tsv": 3, "dim256.tsv": 3, "dim512.tsv": 3, "dim1024.tsv": 3,
                "cluster_num_smallest.txt": 4, "cluster_num_smaller.txt": 4,
                "cluster_num_higher.txt": 4, "cluster_num_highest.txt": 4,
                # "cluster_objects_less.txt": 6, "cluster_objects_few.txt": 6,
                # "cluster_objects_more.txt": 6, "cluster_objects_most.txt": 6,  
                "unbalance.tsv": 30, "unbalance2.txt": 30, 
                 }

    title = {
            "s1.tsv": "I", "s2.tsv": "J", "s3.tsv":"K", "s4.tsv": "L",
            "a1.tsv": "I", "a2.tsv": "J", "a3.tsv": "K", "D31.txt": "L",
            "asymmetric.txt": "A", "skewed.txt": "B", "unbalance.tsv": "C", "unbalance2.txt": "D",
            "Flame.tsv": "A", "smile.txt": "B", "Spiral.tsv": "C", "t4.8k(nonoise).txt": "D", 
            "dim128.tsv": "A", "dim256.tsv": "B", "dim512.tsv": "C", "dim1024.tsv": "D",
            "cluster_num_smallest.txt": "A", "cluster_num_smaller.txt": "B",
            "cluster_num_higher.txt": "C", "cluster_num_highest.txt": "D",
            "cluster_objects_less.txt": "A", "cluster_objects_few.txt": "B",
            "cluster_objects_more.txt": "C", "cluster_objects_most.txt": "D",  
            }


    # for convinience, -1 means unassigned, 0 means noise
    for DatasetName in parameter.keys():
        pathDataset = 'data/'+DatasetName
        unassigned = -1
        noise_label = 0

        # load data
        print(f"Dealing with {DatasetName}...")
        data: ndarray = loadtxt(pathDataset)
        dataCount = size(data, 0)

        if DatasetName not in ["smile.txt", "t4.8k(nonoise).txt", \
                    "cluster_num_smallest.txt", "cluster_num_smaller.txt", \
                    "cluster_num_higher.txt", "cluster_num_highest.txt", \
                    "cluster_objects_less.txt", "cluster_objects_few.txt",\
                    "cluster_objects_more.txt", "cluster_objects_most.txt"]:
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

        # calculate rho, neighborhood, KNN, CN and radius
        indexNeighbor = array([indexDistanceAsc[rowid][distanceAsc[rowid] < delta]
                            for rowid in range(dataCount)], dtype=object)

        radius = full((dataCount, dataCount), 0.0)
        for i in range(dataCount):
            for j in indexNeighbor[i]:
                if j == i:
                    continue
                K = (indexNeighbor[i].shape[0]+ indexNeighbor[j].shape[0])//2
                intersectSet: ndarray = intersect1d(
                    indexDistanceAsc[i][:K], indexDistanceAsc[j][:K], assume_unique=True)
                unionSet: ndarray = union1d(
                    indexDistanceAsc[i][:K], indexDistanceAsc[j][:K])
                CN = intersectSet.size/unionSet.size
                radius[i][j] = radius[j][i] = delta * CN

        # propagation and cluster assignment
        ii = 0
        cluster_num = 0
        q = Queue()
        cluster: ndarray = full(dataCount, unassigned)
        tmp_set: list = []
        tmp_set_len = 0
        while(ii < dataCount):
            q.put(ii)
            cluster_num = cluster_num+1
            cluster[ii] = cluster_num

            while(not q.empty()):
                jj = q.get()
                tmp_set.append(jj)
                tmp_set_len = tmp_set_len + 1
                for kk in range(ii+1, dataCount):
                    if(distance[kk][jj] < radius[kk][jj] and cluster[kk] == unassigned):
                        cluster[kk] = cluster_num
                        q.put(kk)

            if(tmp_set_len < dataCount*noise_ratio):
                cluster_num = cluster_num-1
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
                
        # print(f"number of cluster = {cluster_num}, noise = {sum(cluster == noise_label)}")

        # NMI = normalized_mutual_info_score(
        #     labels_pred=cluster, labels_true=label)
        # AMI = adjusted_mutual_info_score(labels_pred=cluster, labels_true=label)
        # ARI = adjusted_rand_score(labels_pred=cluster, labels_true=label)
        # print(f"NMI = {NMI}, AMI = {AMI}, ARI = {ARI}")

        # plot figure
        cValue = ['#319DA0','#293462','#D61C4E',"#21E6A1",'#AC4425','#FEDB39','#1CD6CE','#781C68','#224B0C','#F94892',
                    "#A20650", "#0b409c", "#A6fAd2", "#6b76ff"]
        color_num = len(cValue)

        if DatasetName in ["dim128.tsv", "dim256.tsv", "dim512.tsv", "dim1024.tsv"]:
            pca = PCA(n_components=2)
            pca.fit(data)
            data = pca.transform(data)

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

        plt.title("("+title[DatasetName]+")", fontstyle='italic',size=20)
        filename = {
            "s1.tsv": "fig7-I", "s2.tsv": "fig7-J", "s3.tsv":"fig7-K", "s4.tsv": "fig7-L",
            "a1.tsv": "fig6-I", "a2.tsv": "fig6-J", "a3.tsv": "fig6-K", "D31.txt": "fig6-L",
            "asymmetric.txt": "fig2-A", "skewed.txt": "fig2-B", "unbalance.tsv": "fig2-C", "unbalance2.txt": "fig2-D",
            "Flame.tsv": "fig4-A", "smile.txt": "fig4-B", "Spiral.tsv": "fig4-C", "t4.8k(nonoise).txt": "fig4-D", 
            "dim128.tsv": "fig5-A", "dim256.tsv": "fig5-B", "dim512.tsv": "fig5-C", "dim1024.tsv": "fig5-D",
            "cluster_num_smallest.txt": "fig3-A", "cluster_num_smaller.txt": "fig3-B",
            "cluster_num_higher.txt": "fig3-C", "cluster_num_highest.txt": "fig3-D", 
            }
        plt.savefig(fr'./fig-plot/{filename[DatasetName]}.png', dpi=200, bbox_inches='tight')
        plt.clf()
