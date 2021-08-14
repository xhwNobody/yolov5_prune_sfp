# -*- coding: utf-8 -*-
import sys,os
sys.path.append('../')
from model import Model
from small_model_mod import Small_Model
import torch
import numpy as np

# if item[:25] not in ['backbone_self.csp1.m.0.cv2', 'backbone_self.csp1.m.1.cv2', 'backbone_self.csp1.m.2.cv2',
#                      'backbone_self.csp2.m.0.cv2', 'backbone_self.csp2.m.1.cv2', 'backbone_self.csp2.m.2.cv2',
#                      'backbone_self.csp2.m.3.cv2', 'backbone_self.csp2.m.4.cv2', 'backbone_self.csp2.m.5.cv2',
#                      'backbone_self.csp2.m.6.cv2', 'backbone_self.csp2.m.7.cv2', 'backbone_self.csp2.m.8.cv2',
#                      'backbone_self.csp3.m.0.cv2', 'backbone_self.csp3.m.1.cv2', 'backbone_self.csp3.m.2.cv2',
#                      'backbone_self.csp3.m.3.cv2', 'backbone_self.csp3.m.4.cv2', 'backbone_self.csp3.m.5.cv2',
#                      'backbone_self.csp3.m.6.cv2', 'backbone_self.csp3.m.7.cv2', 'backbone_self.csp3.m.8.cv2']: 剪第1维（√），不剪第0维（√）
#     print(item)

# if item[:25] not in ['backbone_self.csp1.m.0.cv1', 'backbone_self.csp1.m.1.cv1', 'backbone_self.csp1.m.2.cv1',
#                      'backbone_self.csp2.m.0.cv1', 'backbone_self.csp2.m.1.cv1', 'backbone_self.csp2.m.2.cv1',
#                      'backbone_self.csp2.m.3.cv1', 'backbone_self.csp2.m.4.cv1', 'backbone_self.csp2.m.5.cv1',
#                      'backbone_self.csp2.m.6.cv1', 'backbone_self.csp2.m.7.cv1', 'backbone_self.csp2.m.8.cv1',
#                      'backbone_self.csp3.m.0.cv1', 'backbone_self.csp3.m.1.cv1', 'backbone_self.csp3.m.2.cv1',
#                      'backbone_self.csp3.m.3.cv1', 'backbone_self.csp3.m.4.cv1', 'backbone_self.csp3.m.5.cv1',
#                      'backbone_self.csp3.m.6.cv1', 'backbone_self.csp3.m.7.cv1', 'backbone_self.csp3.m.8.cv1']: 剪第0维（√），不剪第1维(√)
#     print(item)

#backbone_self.csp1.cv1.conv.weight  backbone_self.csp2.cv1.conv.weight  backbone_self.csp3.cv1.conv.weight  剪第1维（√），不剪第0维（√）

#backbone_self.csp1.cv3.weight backbone_self.csp2.cv3.weight backbone_self.csp2.cv3.weight 剪第0维（√），不减第1维(√)

def get_small_model(big_model):
    indice_dict, small_model = extract_para(big_model)

    big_state_dict = big_model.state_dict()
    small_state_dict = {}

    bottleneck_not0 = ['backbone_self.csp1.m.0.cv2', 'backbone_self.csp1.m.1.cv2', 'backbone_self.csp1.m.2.cv2',
                       'backbone_self.csp2.m.0.cv2', 'backbone_self.csp2.m.1.cv2', 'backbone_self.csp2.m.2.cv2',
                       'backbone_self.csp2.m.3.cv2', 'backbone_self.csp2.m.4.cv2', 'backbone_self.csp2.m.5.cv2',
                       'backbone_self.csp2.m.6.cv2', 'backbone_self.csp2.m.7.cv2', 'backbone_self.csp2.m.8.cv2',
                       'backbone_self.csp3.m.0.cv2', 'backbone_self.csp3.m.1.cv2', 'backbone_self.csp3.m.2.cv2',
                       'backbone_self.csp3.m.3.cv2', 'backbone_self.csp3.m.4.cv2', 'backbone_self.csp3.m.5.cv2',
                       'backbone_self.csp3.m.6.cv2', 'backbone_self.csp3.m.7.cv2', 'backbone_self.csp3.m.8.cv2']

    # csp模块中bn前 8个
    csp_bn = ['backbone_self.csp1.bn','backbone_self.csp2.bn', 'backbone_self.csp3.bn', 'backbone_self.csp4.bn',
              'neck_self.csp1.bn', 'neck_self.csp2.bn', 'neck_self.csp3.bn', 'neck_self.csp4.bn']

    # csp模块中bn以及最后的卷积 8个
    csp_after_bn = ['backbone_self.csp1.cv4.conv.weight', 'backbone_self.csp2.cv4.conv.weight',
                    'backbone_self.csp3.cv4.conv.weight', 'backbone_self.csp4.cv4.conv.weight',
                    'neck_self.csp1.cv4.conv.weight', 'neck_self.csp2.cv4.conv.weight',
                    'neck_self.csp3.cv4.conv.weight', 'neck_self.csp4.cv4.conv.weight']

    csp_indict = [torch.cat((indice_dict['backbone_self.csp1.cv3.weight'], indice_dict['backbone_self.csp1.cv2.weight'] + 64)),
                  torch.cat((indice_dict['backbone_self.csp2.cv3.weight'], indice_dict['backbone_self.csp2.cv2.weight'] + 128)),
                  torch.cat((indice_dict['backbone_self.csp3.cv3.weight'], indice_dict['backbone_self.csp3.cv2.weight'] + 256)),
                  torch.cat((indice_dict['backbone_self.csp4.cv3.weight'], indice_dict['backbone_self.csp4.cv2.weight'] + 512)),
                  torch.cat((indice_dict['neck_self.csp1.cv3.weight'], indice_dict['neck_self.csp1.cv2.weight'] + 256)),
                  torch.cat((indice_dict['neck_self.csp2.cv3.weight'], indice_dict['neck_self.csp2.cv2.weight'] + 128)),
                  torch.cat((indice_dict['neck_self.csp3.cv3.weight'], indice_dict['neck_self.csp3.cv2.weight'] + 256)),
                  torch.cat((indice_dict['neck_self.csp4.cv3.weight'], indice_dict['neck_self.csp4.cv2.weight'] + 512))]

    csp_after_bn_indict = [indice_dict['backbone_self.csp1.cv4.conv.weight'], indice_dict['backbone_self.csp2.cv4.conv.weight'],
                           indice_dict['backbone_self.csp3.cv4.conv.weight'], indice_dict['backbone_self.csp4.cv4.conv.weight'],
                    indice_dict['neck_self.csp1.cv4.conv.weight'], indice_dict['neck_self.csp2.cv4.conv.weight'],
                    indice_dict['neck_self.csp3.cv4.conv.weight'], indice_dict['neck_self.csp4.cv4.conv.weight']]

    # csp模块后的卷积 7个
    csp_after_conv = ['backbone_self.conv2.conv.weight', 'backbone_self.conv3.conv.weight',
                      'backbone_self.conv4.conv.weight', 'neck_self.conv1.conv.weight',
                      'neck_self.conv2.conv.weight', 'neck_self.conv3.conv.weight', 'neck_self.conv4.conv.weight']

    #n个bottleneck输入不是上一个特征图 8个
    csp_bottleneck_not1 = ['backbone_self.csp1.m.0.cv1.conv.weight', 'backbone_self.csp1.m.1.cv1.conv.weight', 'backbone_self.csp1.m.2.cv1.conv.weight',
                           'backbone_self.csp2.m.0.cv1.conv.weight', 'backbone_self.csp2.m.1.cv1.conv.weight', 'backbone_self.csp2.m.2.cv1.conv.weight',
                           'backbone_self.csp2.m.3.cv1.conv.weight', 'backbone_self.csp2.m.4.cv1.conv.weight', 'backbone_self.csp2.m.5.cv1.conv.weight',
                           'backbone_self.csp2.m.6.cv1.conv.weight', 'backbone_self.csp2.m.7.cv1.conv.weight', 'backbone_self.csp2.m.8.cv1.conv.weight',
                           'backbone_self.csp3.m.0.cv1.conv.weight', 'backbone_self.csp3.m.1.cv1.conv.weight', 'backbone_self.csp3.m.2.cv1.conv.weight',
                           'backbone_self.csp3.m.3.cv1.conv.weight', 'backbone_self.csp3.m.4.cv1.conv.weight', 'backbone_self.csp3.m.5.cv1.conv.weight',
                           'backbone_self.csp3.m.6.cv1.conv.weight', 'backbone_self.csp3.m.7.cv1.conv.weight', 'backbone_self.csp3.m.8.cv1.conv.weight']
    csp_bottleneck = ['backbone_self.csp4.m.0.cv1.conv.weight',
                      'neck_self.csp1.m.0.cv1.conv.weight', 'neck_self.csp2.m.0.cv1.conv.weight',
                      'neck_self.csp3.m.0.cv1.conv.weight', 'neck_self.csp4.m.0.cv1.conv.weight']
    csp_bottleneck_indict = [indice_dict['backbone_self.csp4.cv1.conv.weight'],
                             indice_dict['neck_self.csp1.cv1.conv.weight'], indice_dict['neck_self.csp2.cv1.conv.weight'],
                             indice_dict['neck_self.csp3.cv1.conv.weight'], indice_dict['neck_self.csp4.cv1.conv.weight']]

    #backbone部分中csp模块中跳跃中的卷积
    csp_backbone_cv1 = ['backbone_self.csp1.cv1.conv.weight', 'backbone_self.csp2.cv1.conv.weight',
                        'backbone_self.csp3.cv1.conv.weight', 'backbone_self.csp4.cv1.conv.weight']

                        
    csp_backbone_cv2 = ['backbone_self.csp1.cv2.weight', 'backbone_self.csp2.cv2.weight',
                        'backbone_self.csp3.cv2.weight', 'backbone_self.csp4.cv2.weight']
    csp_backbone_cv1_cv2_indict = [indice_dict['backbone_self.conv1.conv.weight'], indice_dict['backbone_self.conv2.conv.weight'],
                                   indice_dict['backbone_self.conv3.conv.weight'], indice_dict['backbone_self.spp.cv2.conv.weight']]
                                   
    csp_cv3_not1 = ['backbone_self.csp1.cv3.weight', 'backbone_self.csp2.cv3.weight', 'backbone_self.csp3.cv3.weight']
    csp_cv3 = ['backbone_self.csp1.cv3.weight', 'backbone_self.csp2.cv3.weight',
               'backbone_self.csp3.cv3.weight', 'backbone_self.csp4.cv3.weight',
               'neck_self.csp1.cv3.weight', 'neck_self.csp2.cv3.weight',
               'neck_self.csp3.cv3.weight', 'neck_self.csp4.cv3.weight']
    csp_cv3_indict = [indice_dict['backbone_self.csp1.m.2.cv2.bn.weight'], indice_dict['backbone_self.csp2.m.8.cv2.conv.weight'],
                      indice_dict['backbone_self.csp3.m.8.cv2.conv.weight'], indice_dict['backbone_self.csp4.m.2.cv2.conv.weight'], 
                      indice_dict['neck_self.csp1.m.2.cv2.conv.weight'], indice_dict['neck_self.csp2.m.2.cv2.conv.weight'],
                      indice_dict['neck_self.csp3.m.2.cv2.conv.weight'], indice_dict['neck_self.csp4.m.2.cv2.conv.weight']]

    # 外面的concat 4个
    csp_neck_cv1 = ['neck_self.csp1.cv1.conv.weight', 'neck_self.csp2.cv1.conv.weight',
                    'neck_self.csp3.cv1.conv.weight', 'neck_self.csp4.cv1.conv.weight']
    csp_neck_cv2 = ['neck_self.csp1.cv2.weight', 'neck_self.csp2.cv2.weight',
                    'neck_self.csp3.cv2.weight', 'neck_self.csp4.cv2.weight']
    csp_neck_cv_indict = [torch.cat((indice_dict['neck_self.conv1.conv.weight'],indice_dict['backbone_self.csp3.cv4.conv.weight']+512)),
                          torch.cat((indice_dict['neck_self.conv2.conv.weight'],indice_dict['backbone_self.csp2.cv4.conv.weight']+256)),
                          torch.cat((indice_dict['neck_self.conv3.conv.weight'],indice_dict['neck_self.conv2.conv.weight']+256)),
                          torch.cat((indice_dict['neck_self.conv4.conv.weight'],indice_dict['neck_self.conv1.conv.weight']+512))]

    # spp部分 1个
    spp_cat = ['backbone_self.spp.cv2.conv.weight']
    spp_cat_indict = torch.cat((indice_dict['backbone_self.spp.cv1.conv.weight'], indice_dict['backbone_self.spp.cv1.conv.weight'] + 512,
                                indice_dict['backbone_self.spp.cv1.conv.weight'] + 512*2, indice_dict['backbone_self.spp.cv1.conv.weight'] + 512*3))

    # head部分 3个
    head_det = ['head_self.m.0.weight', 'head_self.m.1.weight', 'head_self.m.2.weight']
    head_det_indict = [indice_dict['neck_self.csp2.cv4.conv.weight'], indice_dict['neck_self.csp3.cv4.conv.weight'], indice_dict['neck_self.csp4.cv4.conv.weight']]

    temp_list = []
    for index, [key, value] in enumerate(big_state_dict.items()):
        if 'num_batches_tracked' not in key:
            # focus、head以及8个csp模块中的bn先完全赋值
            if indice_dict[key] == [] or \
                csp_bn[0] in key or csp_bn[1] in key or csp_bn[2] in key or csp_bn[3] in key \
                or csp_bn[4] in key or csp_bn[5] in key or csp_bn[6] in key or csp_bn[7] in key:
                small_state_dict[key] = value
            else:
                # 减去卷积核的个数， 即第0维度
                if key[:22] in ['backbone_self.csp1.cv1', 'backbone_self.csp2.cv1', 'backbone_self.csp3.cv1']:
                    small_state_dict[key] = value
                
                elif key[:26] in bottleneck_not0:
                    small_state_dict[key] = value
                else:
                    small_state_dict[key] = torch.index_select(value, 0, indice_dict[key])

                # 减去输入特征图的通道数（一般为上一层卷积核的个数），即第1维度
                if 'backbone_self.conv1' not in key and 'bn' not in key:
                    if key in csp_after_bn:       #CSP module中BN后的卷积
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, csp_indict[csp_after_bn.index(key)])
                    elif key in csp_after_conv:   #CSP module后的卷积
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, csp_after_bn_indict[csp_after_conv.index(key)])
                    elif key in csp_bottleneck:   #CSP module中的bottleneck
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, csp_bottleneck_indict[csp_bottleneck.index(key)])
                    elif key in csp_backbone_cv1: #backbone CSP module中第一条支路
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, csp_backbone_cv1_cv2_indict[csp_backbone_cv1.index(key)])
                    elif key in csp_backbone_cv2: #backbone CSP module中第二条支路（跳跃）
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, csp_backbone_cv1_cv2_indict[csp_backbone_cv2.index(key)])
                    elif key in csp_neck_cv1:     #neck CSP module中第一条支路
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, csp_neck_cv_indict[csp_neck_cv1.index(key)])
                    elif key in csp_neck_cv2:     #neck CSP module中第二条支路
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, csp_neck_cv_indict[csp_neck_cv2.index(key)])
                    elif key in csp_cv3:          #backbone CSP module中第一条支路中的卷积
                        if key in csp_cv3_not1:
                            small_state_dict[key] = small_state_dict[key]
                        else:
                            small_state_dict[key] = torch.index_select(small_state_dict[key], 1, csp_cv3_indict[csp_cv3.index(key)])
                    elif key in spp_cat:          #spp module
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, spp_cat_indict)
                    elif key in csp_bottleneck_not1:
                        small_state_dict[key] = small_state_dict[key]
                    else:
                        small_state_dict[key] = torch.index_select(small_state_dict[key], 1, temp_list[-1])
                temp_list.append(indice_dict[key])

        # 直接剪去CSP module中的BN
        if 'num_batches_tracked' not in key:
            for ind, key_csp_bn in enumerate(csp_bn):
                if key_csp_bn in key:
                    small_state_dict[key] = torch.index_select(small_state_dict[key], 0, csp_indict[ind])

        # head部分，只剪去第1维度
        if key in head_det:
            small_state_dict[key] = torch.index_select(small_state_dict[key], 1, head_det_indict[head_det.index(key)])

    small_model.load_state_dict(small_state_dict)
    return small_model

def extract_para(big_model):
    kept_index_per_layer = {}
    big_model_state_dict = big_model.state_dict()
    temp = []
    for ind, key in enumerate(big_model_state_dict.keys()):
        if 'conv.weight' in key or 'cv2.weight' in key or 'cv3.weight' in key:
            if 'focus' not in key:
                indices_zero, indices_nonzero = check_channel(big_model_state_dict[key])
                kept_index_per_layer[key] = indices_nonzero #记下所有卷积该保留的索引
                temp = indices_nonzero
            else:
                kept_index_per_layer[key] = temp  #focus为[]
        elif 'head_self' in key:
            kept_index_per_layer[key] = [] #head为[]
        else:
            if 'bn.num_batches_tracked' in key:
                kept_index_per_layer[key] = [] #bn.num_batches_tracked记为空
            else:
                kept_index_per_layer[key] = temp #记下所有卷积后的bn该保留的索引

    small_model = Small_Model(nc=20, prune_rate=0.7)#########################################

    return kept_index_per_layer, small_model

def check_channel(tensor):

    size_0 = tensor.size()[0]
    size_1 = tensor.size()[1] * tensor.size()[2] * tensor.size()[3]
    tensor_resize = tensor.view(size_0, -1)
    # indicator: if the channel contain all zeros
    channel_if_zero = np.zeros(size_0)
    for x in range(0, size_0, 1):
        channel_if_zero[x] = np.count_nonzero(tensor_resize[x].cpu().numpy()) != 0

    indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])

    zeros = (channel_if_zero == 0).nonzero()[0]
    indices_zero = torch.LongTensor(zeros) if zeros != [] else []

    return indices_zero, indices_nonzero

if __name__ == '__main__':
    # model = torch.load('best_new.pt')['model']
    # small_model = get_small_model(model)
    model_all_keys = torch.load('best.pt')

    model = torch.load('best.pt')['model'].to('cpu')
    model.float()
    big_path = 'big_model.pt'
    torch.save(model, big_path)

    small_model = get_small_model(model)

    small_path = 'small_model.pt'
    torch.save(small_model, small_path)

    model_all_keys['model'] = small_model
    torch.save(model_all_keys, 'small_model_all.pt')



