# -*- coding: utf-8 -*-
import logging
import sys
import math

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from models.common import Conv, SPP, Focus, BottleneckCSP, Concat, NMS, autoShape
from utils.general import check_anchor_order
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, copy_attr

class Model(nn.Module):
    def __init__(self, ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()

        self.ch = ch
        self.nc = nc
        self.depth_multiple = 1.0
        self.width_multiple = 1.0
        self.anchors = [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]

        # 定义模型
        self.backbone_self = backbone(ch)
        self.neck_self = neck()

        self.ch_head = [256, 512, 1024]
        #self.head_self = head(self.nc, self.anchors, self.ch_head)
        self.head_self = Detect(self.nc, self.anchors, self.ch_head)


        # 创建步长和anchor
        if isinstance(self.head_self, Detect):
            s = 128  # 2x min stride
            ch_temp = 3
            self.head_self.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch_temp, s, s))])  # forward
            self.head_self.anchors /= self.head_self.stride.view(-1, 1, 1)
            check_anchor_order(self.head_self)
            self.stride = self.head_self.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)

    def forward(self, x):
        x, x_6, x_4 = self.backbone_self(x)
        x_list = self.neck_self(x, x_6, x_4)
        out = self.head_self(x_list)
        return out

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        m = self.head_self  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

class backbone(nn.Module):
    def __init__(self, inp_ch):  # model, input channels, number of classes
        super(backbone, self).__init__()
        self.focus = Focus(inp_ch, 64, 3)
        self.conv1 = Conv(64, 128, 3, 2)
        self.csp1 = BottleneckCSP(128, 128, 3, shortcut=True) ############
        self.conv2 = Conv(128, 256, 3, 2)
        self.csp2 = BottleneckCSP(256, 256, 9, shortcut=True) ############
        self.conv3 = Conv(256, 512, 3, 2)
        self.csp3 = BottleneckCSP(512, 512, 9, shortcut=True) ############
        self.conv4 = Conv(512, 1024, 3, 2)
        self.spp = SPP(1024, 1024, [5, 9, 13])
        self.csp4 = BottleneckCSP(1024, 1024, 3, shortcut=False) #???????????????

    def forward(self, x):
        # print('inp:', x.shape)
        x_0 = self.focus(x)           #0
        # print('x_0:', x_0.shape)
        x_1 = self.conv1(x_0)         #1
        # print('x_1:', x_1.shape)
        x_2 = self.csp1(x_1)          #2
        # print('x_2:', x_2.shape)
        x_3 = self.conv2(x_2)         #3
        #print('x_3:', x_3.shape)
        # print('x_3:', x_3.shape)
        x_4 = self.csp2(x_3)          #4
        # print('x_4:', x_4.shape)
        x_5 = self.conv3(x_4)         #5
        # print('x_5:', x_5.shape)
        x_6 = self.csp3(x_5)          #6
        # print('x_6:', x_6.shape)
        x_7 = self.conv4(x_6)         #7
        # print('x_7:', x_7.shape)
        x_8 = self.spp(x_7)           #8
        # print('x_8:', x_8.shape)
        out = self.csp4(x_8)          #9
        # print('out:', out.shape)
        return [out, x_6, x_4]

class neck(nn.Module):
    def __init__(self):
        super(neck, self).__init__()
        self.conv1 = Conv(1024, 512, 1, 1)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.cat1 = Concat(dimension=1)
        self.csp1 = BottleneckCSP(1024, 512, 3, shortcut=False)

        self.conv2 = Conv(512, 256, 1, 1)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.cat2 = Concat(dimension=1)
        self.csp2 = BottleneckCSP(512, 256, 3, shortcut=False)

        self.conv3 = Conv(256, 256, 3, 2)
        self.cat3 = Concat(dimension=1)
        self.csp3 = BottleneckCSP(512, 512, 3, shortcut=False)

        self.conv4 = Conv(512, 512, 3, 2)
        self.cat4 = Concat(dimension=1)
        self.csp4 = BottleneckCSP(1024, 1024, 3, shortcut=False)

    def forward(self, x, x_6, x_4):
        x_10 = self.conv1(x)                  #10 512
        # print('x_10:', x_10.shape)
        x_11 = self.upsample1(x_10)           #11
        # print('x_11:', x_11.shape)
        x_12 = self.cat1([x_11, x_6])         #12 512+512
        # print('x_12:', x_12.shape)
        x_13 = self.csp1(x_12)                #13
        # print('x_13:', x_13.shape)

        x_14 = self.conv2(x_13)               #14
        # print('x_14:', x_14.shape)
        x_15 = self.upsample2(x_14)           #15
        # print('x_15:', x_15.shape)
        x_16 = self.cat2([x_15, x_4])         #16 256+256
        # print('x_16:', x_16.shape)
        x_17 = self.csp2(x_16)                #17
        # print('x_17:', x_17.shape)

        x_18 = self.conv3(x_17)               #18
        # print('x_18:', x_18.shape)
        x_19 = self.cat3([x_18, x_14])        #19 256+256
        # print('x_19:', x_19.shape)
        x_20 = self.csp3(x_19)                #20
        # print('x_20:', x_20.shape)

        x_21 = self.conv4(x_20)               #21
        # print('x_21:', x_21.shape)
        x_22 = self.cat4([x_21, x_10])        #22 512+512
        # print('x_22:', x_22.shape)
        x_23 = self.csp4(x_22)                #23
        # print('x_23:', x_23.shape)

        return [x_17, x_20, x_23]

class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
        # return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

if __name__ == '__main__':
    modell = Model(nc=20)
    x = torch.randn(1, 3, 640, 640)
    script_model = torch.jit.trace(modell, x)
    script_model.save("m.pt")
    # print(modell)

    # model_state_dict = modell.state_dict()
    # for index, [key, value] in enumerate(model_state_dict.items()):
    #     print(index, key, value.shape)
    #mask_index = []
    #for index, item in enumerate(modell.parameters()):
        #print(index, item.shape)
#        if len(item.shape) > 1 and index >= 3 and index <= 314:
#            mask_index.append(index)
#    print(mask_index)
#
#    mask_index = [x for x in range(0, 159, 3)]
#    print(mask_index)
 