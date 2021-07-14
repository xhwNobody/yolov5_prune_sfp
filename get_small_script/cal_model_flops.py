# -*- coding: utf-8 -*-
import torch
import sys
sys.path.append('../')
from thop import profile
from model import Model
from small_model_mod import Small_Model


net = Model(nc=20)
#net = Small_Model(nc=20)
#net.load_state_dict(torch.load('small_model.pt'))
input_ = torch.randn(1, 3, 640, 640)
for i in range(5):
    net.forward(input_)


flops, params = profile(net, (input_,))
print('flops: ', flops, 'params: ', params)

#flops:  55897084800.0 params:  42157128.0
#flops:  27991932800.0 params:  20636701.0

