# -*- coding: utf-8 -*-
'''
author:xhw
date:2021.4.26
class Detector_model(object):
    def __init__(self):
        选择设备、加载模型、模型是否半精度、检查图像尺寸、获取类别名、预热
    def img_process(self, img):
        letterbox、转换通道、Totensor、图片是否半精度、归一化、扩展维度
    def detect_frame(self, frame):
        img_process、推理(detect类：将三个特征图合为一个)、NMS(xywh->xyxy)、写入结果（坐标重缩放）
'''
import os
import time
import numpy as np
import math
import yaml
import cv2
import torch
from numpy import random
import torchvision
import torch.nn as nn

dataset = 'voc'
if dataset == 'coco':
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']
elif dataset == 'voc':
    names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
             'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
elif dataset == 'self':
    names = []

class Detector_model(object):
    def __init__(self, model_path, device, img_size=640):
        # 设备选择
        self.device = select_device(device)
        # 是否半精度
        self.half = self.device.type != 'cpu'
        #self.half = False
        # 加载模型
        self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
        # 检查图像尺寸
        self.imgsz = check_img_size(img_size, s=self.model.stride.max())  # check img_size
        # 是否半精度(FP16)
        if self.half:
            self.model.half()  # to FP16
        # 获取类别名称
        self.names = names
        # 定义每个类别的颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        # 预热（随机初始化一张图片进行推理）
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    #图像预处理函数
    def img_process(self, img):
        # 将图片尺寸变为mx640或者640xn（32的倍数，通过缩放、补灰边的方式）
        img = letterbox(img, new_shape=self.imgsz)[0]
        # 转换通道（在datasets.py里改的）
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # 将numpy数组转为tensor
        img = torch.from_numpy(img).to(self.device)
        # 将图片变为FP16
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        # 归一化
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 在batch维度扩展一维
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    #检测视频帧
    def detect_frame(self, frame, show_label=False):
        ori_shape = frame.shape
        fontScale = 0.5
        frame_copy=frame.copy()
        # box的厚度
        bbox_thick = int(0.6 * (ori_shape[0]+ori_shape[1] ) / 600)
        #预处理
        frame = self.img_process(frame)
        #推理并进行计时，默认调用forward方法
        t1 = time.time()
        pred = self.model(frame)[0]
        t2 = time.time()

        points=[]
        # 使用NMS(注：在这个过程中将xywh转为了xyxy)
        pred = non_max_suppression(pred, 0.25, 0.35)


        # 获取类别名称
        class_name = self.names

        # 每个图片的检测结果
        for det in pred:  # detections per image

            # 判断检测结果是否为空
            if det is not None and len(det):
                # 将boxes的坐标从img_size尺寸重新缩放到im0尺寸,且都变为整型
                det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], ori_shape).round()

                # 结果写入,*xyxy,conf,cls表示将前四个赋给xyxy,第五个赋给conf,第六个赋给cls
                for *xyxy, conf, cls in det:
                    x1,y1,x2,y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
                    points.append([x1,y1,x2,y2, points])
                    if show_label:

                        #cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), self.colors[int(cls)], 2)
                        #print()
                        #print(int(cls))
                        bbox_mess = '%s: %.2f' % (class_name[int(cls)], float(conf))

                        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]

                        #cv2.rectangle(frame_copy, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[0]) + t_size[0], int(xyxy[1]) - t_size[1] - 3), (0, 0, 255),-1)  # filled
                        cv2.rectangle(frame_copy, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[0]) + t_size[0], int(xyxy[1]) - t_size[1] - 3), self.colors[int(cls)],-1)  # filled^M

                        cv2.putText(frame_copy, bbox_mess, (int(xyxy[0]), int(xyxy[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        t3 = time.time()
        # 打印推理和后处理时间
        print('infer', t2-t1, 'process', t3-t2)

        return frame_copy,points

# 检查img_size是否为s的倍数
def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    print('new_size', new_size)
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor

# 选择device ==> Detector_model.__init__
def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

# 将图片尺寸变为mx640或者640xn（32的倍数，通过缩放、补灰边的方式）==> Detector_model.img_process
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# 将boxes的坐标从img_size尺寸重新缩放到im0尺寸,且都变为整型 ==> Detector_model.detect_frame
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

# 将坐标限制在范围内
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

# NMS ==> Detector_model.detect_frame
def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

# 将xywh转为xyxy ==> non_max_suppression
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# 计算box1和box2之间的iou ==> non_max_suppression
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        model.append(torch.load(w, map_location=map_location)['model'].float().eval())  # load FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

if __name__ == '__main__':
    #runs/train/weights/best.pt
    #get_small_script/small_model_all.pt
    model = Detector_model('/root/xhw/yolov5-3.1-self_simple/get_small_script/small_model_all.pt', 'cpu')
    path='images/call3.jpg'
    frame = cv2.imread(path)
    t1 = time.time()
    for i in range(5):
        image, points = model.detect_frame(frame, True)
        #cv2.imshow('aa', image)
        #cv2.waitKey(0)
        cv2.imwrite('res.jpg',image)

