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

import mmcv
from mmcv.runner import load_checkpoint
# from mmdet.models import build_detector
# from mmdet.apis import inference_detector, show_result
from mmdet.apis import inference_detector, show_result, init_detector
import matplotlib.pyplot as plt
import copy

import torch
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Rotate, JpegCompression, RandomBrightness
)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def strong_aug(p=1):
    return Compose([
        Rotate(limit=20, p=0.8),
#         RandomRotate90(),
#         HorizontalFlip(p=0.4),
#         Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.7),
        OneOf([
            MotionBlur(p=.7),
            MedianBlur(blur_limit=7, p=0.7),
            Blur(blur_limit=7, p=0.7),
        ], p=0.4),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=35, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.4),
        OneOf([
            CLAHE(),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(0.3, 0.7), 
            JpegCompression(70),
            RandomBrightness(-0.6)
        ], p=1),
        HueSaturationValue(p=0.3),
    ], p=p)


def make_bright(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def remake_master():
    random.seed(42)
    pics = os.listdir('data/master_images/')

    cfg = 'configs/faster_rcnn_r101_fpn_1x.py'
    checkpoint = 'work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth'
    model = init_detector(cfg, checkpoint)
#     cfg = mmcv.Config.fromfile('configs/faster_rcnn_r101_fpn_1x.py')
#     cfg.model.pretrained = None
#     model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
#     _ = load_checkpoint(model, 
#             'work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth')

    ll = 1
    images_path = './data/master_images/'
    save_path = './output/preprocessing/croped_images/'
    all_ids = os.listdir(images_path)
    image_result = {}
    bad_ids = []
    for _fid in all_ids:
        class_label = _fid.split('.')[0]
        boxes = []
        ll+=1
        ths = 0
        image_path = os.path.join(images_path, '{}'.format(_fid))
        image1 = cv2.imread(image_path)
        image = make_bright(image1)
        for angel in [0, 90, 180, 270]: #
            for si in [0.6, 0.625, 0.65]: #
                for crop in [8, 9, 10, 11, 12]: #
                    old_size = image.shape[:2]
                    im = image[0:int(old_size[0]*si), 0:int(old_size[1])]
                    im1 = image1[0:int(old_size[0]*si), 0:int(old_size[1])]
                    im = rotate_bound(im, angel)
                    im1 = rotate_bound(im1, angel)
                    old_size = im.shape[:2]
                    desired_size = 816
                     # old_size is in (height, width) format
                    new_size = (int(old_size[0])//crop, old_size[1]//crop)
                    im = cv2.resize(im, (new_size[1], new_size[0]))
                    im1 = cv2.resize(im1, (new_size[1], new_size[0]))

                    delta_w = desired_size - new_size[1]
                    delta_h = desired_size - new_size[0]
                    top, bottom = delta_h//2, delta_h-(delta_h//2)
                    left, right = delta_w//2, delta_w-(delta_w//2)
                    color = [256, 256, 256]
                    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                        value=color)
                    im1 = cv2.copyMakeBorder(im1, top, bottom, left, right, cv2.BORDER_CONSTANT,
                        value=color)
#                     result = inference_detector(model, im, cfg)
                    result = inference_detector(model, im)
                    if len(result[0])>0:
                        max_ = np.max([x[4] for x in result[0]])
                        if max_ > ths:
                            ths = max_
                            max_ = np.argmax([x[4] for x in result[0]])
                            x1, y1, x2, y2, _ = list(result[0][max_])
                            img = im1[int(y1):int(y2), int(x1):int(x2)]
        try:
            img = cv2.resize(img, (224, 224))
            folder = os.path.join(save_path, '{}/'.format(class_label))
            if not os.path.exists(os.path.dirname(folder)):
                try:
                    os.makedirs(os.path.dirname(folder))
                    cv2.imwrite(folder+'{}_{}.jpg'.format('master', 
                                        class_label), img)
                except:
                    pass
            else:
                cv2.imwrite(folder+'{}_{}.jpg'.format('master', 
                                        class_label), img)
            aug = strong_aug()
            for angel in [0, 90, 180, 270]:
                im = copy.deepcopy(img)
                im = rotate_bound(img, angel)
                for i in range(16):
                    image = aug(image=im)['image']
                    cv2.imwrite(folder+'master_{}_{}_{}.jpg'.format(class_label, angel, i), image)
            del img
            print(_fid)
        except NameError:
            print('baaaad', _fid)
            bad_ids.append(_fid)
    return bad_ids