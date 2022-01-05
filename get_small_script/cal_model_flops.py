# -*- coding: utf-8 -*-
import torch
import sys
sys.path.append('../')
from thop import profile
from model import Model
from small_model_mod import Small_Model

#big model
net1 = torch.load('yolov5_weights/yolov5l.pt', map_location='cuda:0')['model'].type(torch.FloatTensor)
net1.to('cuda')
input1 = torch.randn(1, 3, 640, 640).to('cuda')

flops1, params1 = profile(net1, (input1,))
print('big model flops: ', flops1, 'big model params: ', params1)

#small model
net2 = torch.load('yolov5_weights/yolov5m.pt', map_location='cuda:0')['model'].type(torch.FloatTensor)
net2.to('cuda')
input2 = torch.randn(1, 3, 640, 640).to('cuda')

flops2, params2 = profile(net2, (input2,))
print('flops: ', flops2, 'params: ', params2)


#big model
net1 = torch.load('best.pt', map_location='cuda:0')['model'].type(torch.FloatTensor)
net1.to('cuda')
input1 = torch.randn(1, 3, 640, 640).to('cuda')

flops1, params1 = profile(net1, (input1,))
print('big model flops: ', flops1, 'big model params: ', params1)

#small model
net2 = torch.load('small_model_all.pt', map_location='cuda:0')['model'].type(torch.FloatTensor)
net2.to('cuda')
input2 = torch.randn(1, 3, 640, 640).to('cuda')

flops2, params2 = profile(net2, (input2,))
print('flops: ', flops2, 'params: ', params2)




#flops:  55897084800.0 params:  42157128.0
#flops:  27991932800.0 params:  20636701.0

