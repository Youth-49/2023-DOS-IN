import matplotlib.pyplot as plt
import numpy as np

Datasets = ["asymmetric", "breastTissue", "D31", "skewed", "divorce",
            "ecoli", "fertility_Diagnosis", "Spiral"]

# Datasets = ["asymmetric"]

title = {
        "asymmetric": "A", "D31": "B", "skewed": "C", "Spiral": "D", 
        "breastTissue": "E",  "divorce": "F","ecoli":"G","fertility_Diagnosis": "H"}

for datasetName in Datasets:
    algo_list = ['DOS']
    datafile = [algo+'-16'+'_'+datasetName+'.txt' for algo in algo_list]

    for file in datafile:
        plt.figure(dpi = 150)
        # plt.title(f'({title[datasetName]})',fontstyle='italic',size=20)
        nmi_list = np.loadtxt(file)
        print(nmi_list.shape)
        ax3 = plt.axes(projection='3d')
        x = np.array([2.5, 5, 7.5, 10])
        y = np.array([2.5, 5, 7.5, 10])
        x, y = np.meshgrid(x, y)
        z = np.array([nmi_list[:4], nmi_list[4:8], nmi_list[8:12], nmi_list[12:]])
        ax3.set_title(f'({title[datasetName]})',fontstyle='italic',size=20)
        ax3.plot_surface(x,y,z,cmap='rainbow') 
        # plt.show()
        filename = {"asymmetric": "fig8-A", "D31": "fig8-B", "skewed": "fig8-C", "Spiral": "fig8-D", 
            "breastTissue": "fig8-E",  "divorce": "fig8-F","ecoli":"fig8-G","fertility_Diagnosis": "fig8-H"}
        plt.savefig(f'3d/{filename[datasetName]}.svg', format="svg")
        plt.savefig(f'3d/{filename[datasetName]}.eps', format="eps")
        plt.savefig(f'3d/{filename[datasetName]}.png', dpi = 200, bbox_inches="tight")