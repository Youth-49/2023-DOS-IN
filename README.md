# DOS-IN: A Novel Open-set Clustering Algorithm

Accepted by *Information Sciences*



## Description

DOS (**D**elta **O**pen **S**et) is an interesting clustering algorithm that transforms cluster identification into set identification. It identifies the objects whose neighborhoods coincide as an open-set, and an open-set corresponds to a cluster. However, once the dataset is complex, DOS tends to identify overlapping clusters as one category. We believe the main reason is that DOS unifies the neighborhood radius by a specific function, resulting in the inability to cope with various object distributions. To improve DOS, we propose DOS-IN (**I**rregular **N**eighborhoods). Specifically, DOS-IN generates irregular neighborhoods based on the similarity between objects to self-adapt to diverse object distributions. As a result, DOS-IN not only can accurately distinguish overlapping clusters but also has fewer input parameters. In addition, DOS-IN introduces the small-cluster merging mechanism to address the shortcoming of DOS in recognizing Gaussian clusters. The experimental results show that DOS-IN is completely superior to DOS. Compared with baseline methods, DOS-IN outperforms them on 7 out of 10 datasets, with at least 13.8% (NMI) and 2.4% (RI) improvement in accuracy. The code of DOS-IN is available at https://github.com/Youth-49/2023-DOS-IN



## Reproduction Instructions

environment can be seen in `env.txt`



to reproduce the Figure 7,8,9,10,11,13:

```shell
cd synthetic_data
python DOS-IN-plot.py
```



to reproduce the Figure 12:

```shell
cd outlier
python DOS-IN-plot.py
```



to reproduce the Figure 14:

```shell
cd sensitive
sh run.sh
cd res
python comp.py
```



to test DOS-IN performance and runtime in real world datasets:

```sh
cd real_data
sh run_opt.sh # for faster DOS-IN
```

