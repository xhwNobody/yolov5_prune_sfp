整理中<br>

模型权重文件：<br>
链接：https://pan.baidu.com/s/1vHqTUWc4uG9xSH5RMHsAMQ <br>
提取码：vdro <br>

yolov5l:官方预训练模型 <br>
pretrained.pt转换好的新模型 <br>
(将其放在weights文件夹下,运行copy_weight.py可以得到pretrained.pt) <br>

best.pt:将pretrained.pt作为预训练权重,使用sfp软剪枝训练好的模型权重 <br>
big_model.pt:大模型 <br>
small_model.pt:裁剪后获得的小模型 <br>
small_model_all.pt:裁剪后获得的小模型加了一些参数,可直接用于推理 <br>
(将其放在get_small_scipt文件夹下,运行get_small_model.py可以得到其余三个文件) <br>
