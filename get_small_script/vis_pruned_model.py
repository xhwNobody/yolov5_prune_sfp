# -*- coding: utf-8 -*-
import torch
import sys,os
sys.path.append('../')
# print(model['model'].parameters())
# for index, item in enumerate(model['model'].named_modules()):
#     print(index, item)

# i = 0
# for index, item in enumerate(model['model'].parameters()):
#
#     i+=1
#     print(i)

# model_state_dict = model['model'].state_dict()
# model_state_dict = model.state_dict()

# model_state_dict = model['model'].state_dict()
# for index, [key, value] in enumerate(model_state_dict.items()):
#     print(index, key, value.shape)

big_model = torch.load('big_model.pt')
# print(big_model.keys())
small_model = torch.load('small_model.pt')
big_model_state_dict = big_model.state_dict()
small_model_state_dict = small_model.state_dict()
ind = 0
for index, [key, value] in enumerate(big_model_state_dict.items()):
    if 'bn.running_mean' not in key and 'bn.running_var' not in key and 'bn.num_batches_tracked' not in key:
        print(ind, index, key, value.shape, small_model_state_dict[key].shape)
        ind += 1

# print(big_model_state_dict['backbone_self.csp2.cv4.bn.weight'])
# print(small_model_state_dict['backbone_self.csp2.cv4.bn.weight'])
#
# print(big_model_state_dict['backbone_self.csp2.cv4.bn.bias'])
# print(small_model_state_dict['backbone_self.csp2.cv4.bn.bias'])
