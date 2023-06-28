import matplotlib.pyplot as plt
import numpy as np

Datasets = ["asymmetric", "breastTissue", "D31", "skewed", "divorce",
            "ecoli", "fertility_Diagnosis", "Spiral"]

title = {
        "asymmetric": "A", "D31": "B", "skewed": "C", "Spiral": "D", 
        "breastTissue": "E",  "divorce": "F","ecoli":"G","fertility_Diagnosis": "H"}

for datasetName in Datasets:
    algo_list = ['DOS-IN', 'DOS']
    datafile = [algo+'-16'+'_'+datasetName+'.txt' for algo in algo_list]

    plt.figure(dpi = 150)
    plt.title(f'({title[datasetName]})',fontstyle='italic',size=20)
    plt.grid(axis='y')
    plt.xlabel('Params')
    plt.ylabel('NMI')
    x = ['P'+str(i) for i in range(1, 17)]

    for file in datafile:
        nmi_list = np.loadtxt(file)
        print(nmi_list.shape)
        nmi_list = np.sort(nmi_list)
        nmi_list = nmi_list.tolist()
        # x = range(1, 17)
        plt.plot(x, nmi_list, linestyle='-.', linewidth=3)
        # for i in x:
        #     plt.text(x[i-1], nmi_list[i-1], round(nmi_list[i-1], 2), ha='center', va='bottom', fontsize=5)

    plt.legend(['DOS-IN', 'DOS'], prop={"size":18}, loc="upper left")
    # plt.show()
    filename = {"asymmetric": "fig8-A", "D31": "fig8-B", "skewed": "fig8-C", "Spiral": "fig8-D", 
        "breastTissue": "fig8-E",  "divorce": "fig8-F","ecoli":"fig8-G","fertility_Diagnosis": "fig8-H"}
    plt.savefig(f'fig/{filename[datasetName]}.svg', format="svg")
    plt.savefig(f'fig/{filename[datasetName]}.eps', format="eps")
    plt.savefig(f'fig/{filename[datasetName]}.png', dpi = 200, bbox_inches="tight")