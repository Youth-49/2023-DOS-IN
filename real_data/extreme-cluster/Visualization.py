import imp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, rand_score


def Visualization(DatasetName, data, label, clusteringResult, isShowNoise=False):
    number = data.shape[0] 
    color = ['#278EA5','#21E6C1','#FFBA5A','#FF7657','#C56868','#6b76ff','#ff0000','#9e579d','#f85f73'
    ,'#928a97','#b6f7c1','#0b409c','#d22780','#882042','#071E3D']
    lineForm = ['o','+','*','x']
    if data.shape[1]>2:
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
    for i in range(np.max(clusteringResult)):
        Together = []
        flag = 0
        for j in range(number):
            if clusteringResult[j] == i+1:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        colorNum = np.mod(i+1,15)
        formNum = np.mod(i+1,4)
        plt.scatter(Together[:,0], Together[:,1], 15, color[colorNum], lineForm[formNum])
    
    NMI = normalized_mutual_info_score(label, clusteringResult)
    ARI = adjusted_rand_score(label, clusteringResult)
    RI = rand_score(label, clusteringResult)

    # plt.xlabel('attribute 1', fontsize=20)
    # plt.ylabel('attribute 2', fontsize=20)
    # plt.title(f'Extreme clustering: NMI = {NMI}')
    # if isShowNoise == True:
    #     Together = []
    #     flag = 0
    #     for j in range(number):
    #         if clusteringResult[j] == -1:
    #             flag += 1
    #             Together.append(data[j])
    #     Together = np.array(Together, dtype = int)
    #     if Together.shape[0] != 0:
    #         plt.scatter(Together[:,0], Together[:,1], 15, 'k', '.')
    # plt.show()
    # plt.savefig(fr'./fig/{DatasetName[:-4]}.png', dpi=600)
    return NMI, ARI, RI
