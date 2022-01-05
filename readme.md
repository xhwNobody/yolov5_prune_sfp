#### 安装

```sh
$ git clone https://github.com/xhwNobody/yolov5_prune_sfp.git
$ cd yolov5_prune_sfp
$ pip3 install -r requirements.txt
```

#### 推理

1.下载[模型权重](https://pan.baidu.com/s/16xHcdYqagctedT2DjLBtCw)，提取码：m428

说明：yolov5l-官方预训练模型；best.pt-使用SFP剪枝训练好的模型权重。

```sh
$ mv weights yolov5_prune_sfp
```

2.转换模型

```shell
$ cp weights/best.pt get_small_script
$ python3 get_small_script/get_small_model.py
$ python3 to_jit_gpu.py
  (or python3 to_jit_gpu.py)
```

3.在gpu或cpu下推理

```sh
$ python3 detector_gpu.py 
  (or python3 detector_cpu.py)
```

#### 训练

1.下载[VOC数据集]()

2.制作labels

```shell
$ cd VOC2012
$ python3 step1_split_data.py
$ python3 step2_voc_label.py
```

3.转换模型

```
$ python3 weights/copy_weight.py
```

4.开始训练

```sh
$ python3 train_prune_sfp.py --data data/voc.yaml --device 1 --weights weights/pretrained.pt --hyp data/hyp.finetune.yaml
```

#### 文章

相关内容参考知乎：https://zhuanlan.zhihu.com/p/391045703

#### 参考

【1】yolov5官方地址：https://github.com/ultralytics/yolov5.git

【2】SFP论文地址：https://arxiv.org/abs/1808.07471

【3】FPGM论文地址：https://arxiv.org/pdf/1811.00250.pdf