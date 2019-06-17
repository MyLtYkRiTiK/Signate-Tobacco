import os
import pandas as pd
import json
import glob
from PIL import Image
import numpy as np
import pickle
import cv2
import mmcv
import random
import matplotlib.pyplot as plt
import copy

import mmcv
from mmcv.runner import load_checkpoint
# from mmdet.models import build_detector
# from mmdet.apis import inference_detector, show_result
from mmdet.apis import inference_detector, show_result, init_detector
import matplotlib.pyplot as plt

import torch
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform

def predict_and_crop():
    
    random.seed(42)

    pics = os.listdir('data/test_images_1/')
    len(pics)

    cfg = 'configs/faster_rcnn_r101_fpn_1x.py'
    checkpoint = 'work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth'
    model = init_detector(cfg, checkpoint)
#     cfg = mmcv.Config.fromfile('configs/faster_rcnn_r101_fpn_1x.py')
#     cfg.model.pretrained = None
#     model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
#     _ = load_checkpoint(model, 
#             'work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth')

    ths=0.3
    ll = 1
    images_path = './data/test_images_1/'
    image_result = {}

    all_ids = os.listdir(images_path)
    for _fid in all_ids:
        print(ll, _fid)
        ll+=1
        image_path = os.path.join(images_path, '{}'.format(_fid))
        image = cv2.imread(image_path)
        image_result[_fid] = {}

        h, w, c = image.shape
        i = 0
        #нарезаю рисунки на квадраты 816px с отступом друг от друга на 408px
        for ww in range(w//408):
            for hh in range(h//408):
                w0 = ww*408
                h0 = hh*408
                h1 = h0+816
                w1 = w0+816
                if w1>w or h1>h:
                    break
                image1 = image[h0:h1, w0:w1]

#                 result = inference_detector(model, image1, cfg)
                result = inference_detector(model, image1)
                for j, res in enumerate(result):
                    if len(res)>0:
                        for box_can in res:
                            if box_can[4] > ths:
                                x1 = int(w0+box_can[0])
                                y1 = int(h0+box_can[1])
                                x2 = int(w0+box_can[2])
                                y2 = int(h0+box_can[3])
                                if 1 in image_result[_fid]:
                                    image_result[_fid][1] += [[box_can[4], x1, y1, x2, y2]]
                                else:
                                    image_result[_fid][1] = [[box_can[4], x1, y1, x2, y2]]

    all_boxes = {}
    for test_name in image_result:
        need = []
        check = {}
        a = ['{} {} {} {} {}'.format(x[0], x[1], x[2], x[3], x[4]) for x in image_result[test_name][1]]
        lena = len(a)
        for i in range(lena):
            choose = [[float(x) for x in a[i].split()]]
            if a[i] in check:
                continue
            else:
                check[a[i]] = 1
            for j in range(i+1, lena):
                b = [float(x) for x in a[i].split()]
                c = [float(x) for x in a[j].split()]
                if (abs(b[1]-c[1])<=30 and abs(b[2]-c[2])<=30 and
                    abs(b[3]-c[3])<=30 and abs(b[4]-c[4])<=30):
                            choose.append(c)
                            check[a[j]] = 1
            need.append(choose[np.argmax([z[0] for z in choose])])
        all_boxes[test_name] = need

    test_path = './output/preprocessing/test/'

    if not os.path.exists(os.path.dirname(test_path)):
            os.makedirs(os.path.dirname(test_path))
    for test_name in all_boxes:
        name = test_name.split('.')[0]
        img = cv2.imread('./data/test_images_1/'+test_name)
        im_size = img.shape[:2]
        for box in all_boxes[test_name]:
                _, x1, y1, x2, y2 = list(box)
                w = x2 - x1
                h = y2 - y1
                if w > h:
                    delta = (w-h)//2
                    x11 = copy.copy(x1)
                    x22 = copy.copy(x2)
                    y11 = max(0, copy.copy(y1) - delta)
                    y22 = min(copy.copy(y2) + delta, im_size[0])
                else:
                    delta = (h-w)//2
                    x11 = max(0, copy.copy(x1) - delta)
                    x22 = min(copy.copy(x2) + delta, im_size[1])
                    y11 = copy.copy(y1)
                    y22 = copy.copy(y2)

                image = img[int(y11):int(y22), int(x11):int(x22)]
                image = cv2.resize(image, (224, 224))
                cv2.imwrite(test_path+
                '{}_{}_{}_{}_{}.jpg'.format(name, 
                int(x1), int(y1), int(x2), int(y2)), image)