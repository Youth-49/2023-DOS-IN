# 2023-DOS-IN

submitted to Journal of *Information Sciences*



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

