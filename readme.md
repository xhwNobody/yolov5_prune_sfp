# 基于SFP和FPGM的yolov5的软剪枝实现

### 安装

```sh
$ git clone https://github.com/xhwNobody/yolov5_prune_sfp.git
$ cd yolov5_prune_sfp && pip3 install -r requirements.txt
```

### 推理

1.下载[模型权重](https://pan.baidu.com/s/1_4gLnNwG5RaJBggKRpXoxQ)，提取码：4b6p


说明：yolov5l-官方预训练模型；best.pt-使用SFP剪枝训练好的模型权重。

```sh
$ mv weights yolov5_prune_sfp
```

2.修剪权重和转换模型

```shell
$ cp weights/best.pt get_small_script
$ python3 get_small_script/get_small_model.py
$ python3 to_jit_gpu.py (or python3 to_jit_gpu.py)
```

3.在gpu或cpu下推理

```sh
$ python3 detector_gpu.py (or python3 detector_cpu.py)
```

### 训练

1.下载[VOC数据集](https://pan.baidu.com/s/12ncD6qfj8WsGotmB8vlm7g)，提取码：7jnf 

2.制作labels

```shell
$ python3 VOC2012/step1_split_data.py
$ python3 VOC2012/step2_voc_label.py
```

3.转换模型

```shell
$ python3 weights/copy_weight.py
```

4.开始训练

①正常训练（不剪枝）

```sh
$ python3 train.py --data data/voc.yaml --weights weights/pretrained.pt --epoch 50 --device 0 --hyp data/hyp.finetune.yaml
```

②利用sfp进行剪枝训练

```sh
$ python3 train_prune_sfp.py --data data/voc.yaml --device 1 --weights weights/pretrained.pt --hyp data/hyp.finetune.yaml
```

③利用fpgm进行剪枝训练

```sh
$ python3 train_prune_fpgm.py --data data/voc.yaml --device 1 --weights weights/pretrained.pt --hyp data/hyp.finetune.yaml
```

### 文章

相关内容参考知乎：https://zhuanlan.zhihu.com/p/391045703

### 参考

【1】yolov5官方地址：https://github.com/ultralytics/yolov5.git

【2】SFP代码地址：https://github.com/he-y/soft-filter-pruning.git

【3】FPGM代码地址：https://github.com/he-y/filter-pruning-geometric-median.git

\*\***亲爱的童鞋，如果我的文章和代码对你有帮助，希望点赞和star支持一下，欢迎交流！谢谢！**\*\*

