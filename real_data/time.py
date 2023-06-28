from matplotlib import pyplot as plt
import numpy as np

BSAS = [0.0014, 0.0342, 0.0043, 0.0049, 0.0016, 0.0054, 0.0023, 0.0300, 0.0024, 0.0028]
KMeans = [0.0580, 0.1331, 0.0218, 0.0459, 0.0495, 0.0363, 0.0177, 0.1828, 0.0482, 0.0510]

Extreme = [0.0054, 1.0830, 0.0089, 0.0235, 0.0094, 0.0435, 0.0012, 0.1232, 0.0127, 0.0045]
HDBSCAN = [0.0038, 0.4689, 0.0085, 0.0066, 0.0034, 0.0099, 0.0022, 0.0141, 0.0044, 0.0038]
DOS_IN = [0.0139, 2.0661, 0.0681, 0.0411, 0.0058, 0.0491, 0.0009, 0.3220, 0.0153, 0.0077]
IDPC = [0.0053, 1.5743, 0.0225, 0.0765, 0.0126, 0.2002, 0.0006, 0.5714, 0.1306, 0.0053]
DOS = [0.0050, 1.9530, 0.0114, 0.0388, 0.0020, 0.0505, 0.0008, 0.1263, 0.0070, 0.0042]

EMA = [0.1170, 2.8973, 0.2070, 2.6906, 0.4478, 0.5291, 0.0187, 4.0532, 0.6619, 0.0427]
DPCKNN = [0.0114, 5.1502, 0.0109, 0.0889, 0.0077, 0.0994, 0.0008, 0.3756, 0.0573, 0.0039]

# SNNDPC = [0.0558, 13.2852, 0.1354, 0.4641, 0.0438, 1.0577, 0.0034, 4.0727, 0.4255, 0.0496]
DPC = [0.0721, 14.1756, 0.1431, 0.5770, 0.0605, 1.5450, 0.0081, 5.3207, 0.5827, 0.1224]
DBSCAN = [0.0558, 15.5308, 0.1419, 0.5584, 0.0478, 1.2458, 0.0029, 4.8257, 0.4713, 0.2132]
Syncnet = [0.0527, 14.4980, 0.0701, 0.3274, 0.1532, 0.0330, 0.0029, 5.6292, 0.9502, 0.0037]
Hsyncnet = [0.0242, 99.1146, 0.3460, 1.0906, 0.3393, 1.3381, 0.0023, 20.4889, 2.6774, 0.1276]
Spectral = [0.0362, 659.6238, 1.6163, 4.2485, 0.1618, 0.0829, 0.0274, 56.7161, 0.0863, 0.0602]


# res = [DOS_IN, DOS, DBSCAN, HDBSCAN, DPC, DPCKNN, Extreme, IDPC, SNNDPC, EMA, Spectral, BSAS, KMeans, Syncnet, Hsyncnet]
# algo = ['DOS-IN', 'DOS', 'DBSCAN', 'HDBSCAN', 'DPC', 'DPCKNN', 'Extreme', 'IDPC', 'SNNDPC',\
#         'EMA', 'Spectral', 'BSAS', 'K-Means', 'Syncnet', 'Hsyncnet']

res = [DOS_IN, DOS, DBSCAN, HDBSCAN, DPC, DPCKNN, Extreme, IDPC, EMA, Spectral, BSAS, KMeans, Syncnet, Hsyncnet]
algo = ['DOS-IN', 'DOS', 'DBSCAN', 'HDBSCAN', 'DPC', 'DPCKNN', 'Extreme', 'IDPC',\
        'EMA', 'Spectral', 'BSAS', 'K-Means', 'Syncnet', 'Hsyncnet']
dataset1 = ['Breast', 'Digits', 'Divorce', 'Ecoli', 'Fertility', 'Led7digit', 'Lenses', 'Skewed', 'Spiral', 'Zoo']
dataset2 = ['Lenses', 'Fertility', 'Zoo', 'Breast', 'Divorce', 'Spiral', 'Ecoli', 'Led7digit',  'Skewed', 'Digits']
replace_mapping = []
for name in dataset2:
    idx = dataset1.index(name)
    replace_mapping.append(idx)

for id, my_list in enumerate(res):
    tmp_list = [my_list[replace_mapping[i]] for i in range(len(my_list))]
    res[id] = tmp_list


res = np.array(res)
n,m = res.shape
new_res = np.full_like(res, 0)
# 0-0.2 -> 0-0.3
# 0.2-2.5 -> 0.3-0.8
# 2.5-5 -> 0.8-1.8
# 5-15 -> 1.8-3
# 15-600 -> 3-4.5

for i in range(n):
    for j in range(m):
        if res[i][j] <= 0.2:
            new_res[i][j] = res[i][j]*1.5

        elif 0.2 < res[i][j] and res[i][j] <= 2.5:
            new_res[i][j] = (res[i][j] - 0.2) / 2.3 * 0.5 + 0.3

        elif 2.5 < res[i][j] and res[i][j] <= 5:
            new_res[i][j] = (res[i][j] - 2.5) / 2.5 * 1 + 0.8

        elif 5 < res[i][j] and res[i][j] <= 15:
            new_res[i][j] = (res[i][j] - 5) / 10 * 1.2 + 1.8
        else:
            new_res[i][j] = (res[i][j] - 15) / 585 * 1.5 + 3

# cValue = ['#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF',
#           '#FF00FF', '#FFA500', '#800080', '#8B0000', '#BDB76B',
#           '#006400', '#008B8B', '#00008B', '#8B008B', '#A52A2A']

cValue = ['#FF5733', '#C70039', '#900C3F', '#581845', '#FFC300',
          '#E74C3C', '#8E44AD', '#3498DB', '#1ABC9C', '#27AE60',
          '#F1C40F', '#E67E22', '#BDC3C7', '#2C3E50', '#34495E',
          '#7F8C8D']

cValue = ['#FF5733', '#C70039', '#900C3F', '#581845', '#FFC300',
          '#00CED1', '#8E44AD', '#3498DB', '#1ABC9C', '#27AE60',
          '#F1C40F', '#E67E22', '#BDC3C7', '#2C3E50', '#34495E',
          '#7F8C8D']

plt.figure(figsize=(8,5))

for i, my_list in enumerate(new_res):
    if i == 0:
        plt.plot(dataset2, my_list, linewidth=2, c=cValue[i], linestyle='--', label=algo[i], marker='o', markersize=4)
    else:
        plt.plot(dataset2, my_list, linewidth=2, c=cValue[i], label=algo[i], marker='o', markersize=4)

line1 = np.full_like(res[0], 0.8)
line2 = np.full_like(res[0], 1.8)
# line3 = np.full_like(res[0], 2)
# line4 = np.full_like(res[0], 2)
plt.plot(dataset2, line1, linewidth=1, linestyle='-.', color='grey')
plt.plot(dataset2, line2, linewidth=1, linestyle='-.', color='grey')
# plt.plot(dataset2, line3, linewidth=1, linestyle='-.', color='grey')
# plt.plot(dataset2, line4, linewidth=1, linestyle='-.', color='k')

plt.xticks(range(len(dataset2)), dataset2, size=10, rotation=20)
y = [0, 2.5, 5, 15, 600]
y1 = [0, 0.8, 1.8, 3, 4.5]
plt.yticks(y1, y, size=10)
plt.title("Running time", fontstyle='italic', size=20)
plt.xlabel('datasets', fontstyle='italic', size=20)
plt.ylabel('second', fontstyle='italic', size=20)
plt.legend(loc="upper left", fontsize=12, ncol=2)
plt.savefig('time.png', dpi=200, bbox_inches='tight')