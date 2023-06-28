from numpy import array, ndarray, loadtxt, size, argsort, sort, sum, full, array, exp, where
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as patches
from sklearn.decomposition import PCA

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    vars(args)['algorithm'] = 'DOS'
    # best NMI result:

    parameter = {
                 "a1.tsv": (2, 2.1, 0.02), "a2.tsv": (2., 1.6, 0.01), "a3.tsv":(2, 1.9, 0.01), "D31.txt": (1.91, 1.97, 0.01), 
                #  "s1.tsv": (5.4, 2, 0.01), 
                #  "s2.tsv": (3.3, 1, 0.01), 
                #  "s3.tsv": (2.5, 1, 0.01), 
                #  "s4.tsv": (2.7, 1, 0.01)
                 }

    title = {
            "s1.tsv": "E", "s2.tsv": "F", "s3.tsv":"G", "s4.tsv": "H",
            "a1.tsv": "E", "a2.tsv": "F", "a3.tsv": "G", "D31.txt": "H",
            # "clean-a1.tsv": "I", "clean-a2.tsv": "J", "clean-a3.tsv": "K", "clean-D31.txt": "L",
            }


    for DatasetName in parameter.keys():
        params = parameter[DatasetName]
        # for convinience, -1 means unassigned, 0 means noise
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

        # set parameters
        baseDelta = F[1]
        delta = baseDelta*params[0]
        K = params[1]
        noise_ratio = params[2]
        print(f"baseDelta={baseDelta}, delta={delta}, K={params[1]}")
        # compute rho, mu and std
        rho = array([len(arr[arr < delta])-1 for arr in distance])
        mu = rho.mean()
        sigma = rho.std()  
        radius = array([delta if n_i >= mu-sigma
                        else K*delta*(1-exp((n_i-mu)/(sigma)))/(1+exp((n_i-mu)/(sigma))) for n_i in rho])
        mark = array([1 if n_i >= mu-sigma
                        else 2 for n_i in rho])
        # assign
        cluster = full(dataCount, unassigned)
        cluster_num = 0
        for i in range(dataCount):
            if (cluster[i] == unassigned):
                cluster_num = cluster_num + 1
                cluster[i] = cluster_num

            for j in range(1, dataCount):
                if (distanceAsc[i][j] < radius[i]):
                    if (cluster[indexDistanceAsc[i][j]] == unassigned):
                        cluster[indexDistanceAsc[i][j]] = cluster[i]
                    elif (cluster[i] != cluster[indexDistanceAsc[i][j]]):
                        x = max(cluster[i], cluster[indexDistanceAsc[i][j]])
                        y = min(cluster[i], cluster[indexDistanceAsc[i][j]])
                        cluster_num = cluster_num-1
                        for kk in range(dataCount):
                            if (cluster[kk] == x):
                                cluster[kk] = y
                            elif (cluster[kk] > x):
                                cluster[kk] = cluster[kk]-1

                else:
                    break

        N = cluster_num
        i = 1
        while (i <= cluster_num):
            if (sum(cluster == i) < noise_ratio*dataCount):
                cluster_num = cluster_num - 1
                for j in range(dataCount):
                    if (cluster[j] == i):
                        cluster[j] = noise_label
                    if (cluster[j] > i):
                        cluster[j] = cluster[j] - 1
            else:
                i = i + 1

        if data.shape[1] == 2:

            cValue = ['#319DA0','#293462','#D61C4E',"#21E6A1",'#AC4425','#FEDB39','#1CD6CE','#781C68','#224B0C','#F94892',
                    "#A20650", "#0b409c", "#A6fAd2","#6b76ff"]
            color_num = len(cValue)

            plt.figure(1)
            if DatasetName not in ["s1.tsv", "s2.tsv", "s3.tsv", "s4.tsv"]:
                idx_0 = where(cluster == 0)
                plt.scatter(data[idx_0, 0], data[idx_0, 1], s=40, marker='.', c="dimgrey")
            for mx in range(1, cluster_num+1):
                color = mx % color_num
                idx_0 = where(cluster == mx)
                plt.scatter(data[idx_0, 0], data[idx_0, 1], s=20, marker='.', c=cValue[color])

            # for s2
            if DatasetName == "s2.tsv":
                ax = plt.gca()
                rect1 = patches.Rectangle((490000, 80000), 900000-490000, 330000-80000, fill=None, edgecolor="red", linewidth=3)
                ax.add_patch(rect1)
            
            if DatasetName == "s3.tsv":
                ax = plt.gca()
                rect1 = patches.Rectangle((270000, 540000), 440000-270000, 860000-540000, fill=None, edgecolor="red", linewidth=3)
                rect2 = patches.Rectangle((98000, 160000), 310000-98000, 380000-160000, fill=None, edgecolor="red", linewidth=3)
                rect3 = patches.Rectangle((700000, 590000), 860000-700000, 890000-590000, fill=None, edgecolor="red", linewidth=3)
                rect4 = patches.Rectangle((650000, 80000), 750000-500000, 550000-100000, fill=None, edgecolor="red", linewidth=3, angle=45)
                ax.add_patch(rect1)
                ax.add_patch(rect2)
                ax.add_patch(rect3)
                ax.add_patch(rect4)

            if DatasetName == "s4.tsv":
                ax = plt.gca()
                rect1 = patches.Rectangle((330000, 178000), 803000-330000, 800000-178000, fill=None, edgecolor="red", linewidth=3)
                ax.add_patch(rect1)

            plt.title("("+title[DatasetName]+")", fontstyle='italic',size=20)
            # plt.show()
            filename = {
                "s1.tsv": "fig7-E", "s2.tsv": "fig7-F", "s3.tsv":"fig7-G", "s4.tsv": "fig7-H",
                "a1.tsv": "fig6-E", "a2.tsv": "fig6-F", "a3.tsv": "fig6-G", "D31.txt": "fig6-H",
                # "clean-a1.tsv": "I", "clean-a2.tsv": "J", "clean-a3.tsv": "K", "clean-D31.txt": "L",
            }
            plt.savefig(fr'./fig-plot/{filename[DatasetName]}.png', dpi=200, bbox_inches='tight')
            plt.clf()

            # if DatasetName in ["D31.txt", "a1.tsv", "a2.tsv", "a3.tsv"]:
            #     plt.figure(1)
            #     for mx in range(1, cluster_num+1):
            #         color = mx % color_num
            #         idx_0 = where(cluster== mx)
            #         if mx != noise_label:
            #             plt.scatter(data[idx_0, 0], data[idx_0, 1], s=10, marker='.', c=cValue[color])

            #     plt.title("("+title['clean-'+DatasetName]+")", fontstyle='italic',size=20)
            #     plt.savefig(fr'./fig-plot/{args.algorithm}_clean_{DatasetName[:-4]}.png', dpi=150, bbox_inches='tight')
            #     plt.clf()
