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
import os
import errno
import copy

def crop():
    pics = os.listdir('data/train_annotations')
    ll = []
    annot_path = './data/train_annotations'
    images_path = './data/train_images'
    save_path = './output/preprocessing/croped_images'
    with open('./output/preprocessing/remap_dict.json', 'r') as f:
        remap_dict = json.load(f)
    annotations_result = []

    all_ids = os.listdir(annot_path)
    for _file in pics:
        _fid = _file.split('.')[0]
        image_path = os.path.join(images_path, '{}.jpg'.format(_fid))
        image = cv2.imread(image_path)
        im_size = image.shape[:2]
        file_path = os.path.join(annot_path, _file)

        with open(file_path, 'r') as f:
            annot_data = json.load(f)

        label_dict = {}
        for label in annot_data['labels']:
            class_label = label['category']
            if class_label in label_dict:
                label_dict[class_label] += 1
            else:
                label_dict[class_label] = 1
            x1 = int(label['box2d']['x1'])
            x2 = int(label['box2d']['x2'])
            y1 = int(label['box2d']['y1'])
            y2 = int(label['box2d']['y2'])
            
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
                
            image1 = image[int(y11):int(y22), int(x11):int(x22)]
            image1 = cv2.resize(image1, (224, 224)) 
            folder = os.path.join(save_path, '{}/'.format(class_label))
            if not os.path.exists(os.path.dirname(folder)):
                try:
                    os.makedirs(os.path.dirname(folder))
                    cv2.imwrite(folder+'{}_{}.jpg'.format(_fid, label_dict[class_label]), image1)
                except:
                    pass
            else:
                cv2.imwrite(folder+'{}_{}.jpg'.format(_fid, label_dict[class_label]), image1)
            
    print('crop ready')