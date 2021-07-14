import torch
import sys,os
sys.path.append('../')
from model import Model

model_src = torch.load('yolov5l.pt')#['model']
model_src = model_src['model']
model_src_dict = model_src.state_dict()

model_dst = Model(nc=80)
model_dst_dict = model_dst.state_dict()
model_dst.load_state_dict(model_dst_dict)

model_dst_dict_list = list(model_dst_dict)

for ind, (key, value) in enumerate(model_src_dict.items()):

    model_dst_dict[model_dst_dict_list[ind]] = value

model_dst.load_state_dict(model_dst_dict)
ckpt = {'epoch': -1,
        'best_fitness': None,
        'training_results': None,
        'model': model_dst,
        'optimizer': None }
torch.save(ckpt,'pretrained.pt')