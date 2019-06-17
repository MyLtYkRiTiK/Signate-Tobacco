import os
import pandas as pd
import json
import glob
import numpy as np
import pickle
import cv2
import mmcv
import random
import matplotlib.pyplot as plt

def annot():
    random.seed(42)

    pics = os.listdir('data/train_annotations')
    random.shuffle(pics)
    train = pics

    folder = train
    labels = os.listdir('data/master_images/')
    labels = sorted([x[:-4] for x in labels])
    labels = dict(zip(labels, range(1,len(labels)+1)))
    if not os.path.exists(os.path.dirname('output/preprocessing/')):
        os.makedirs(os.path.dirname('output/preprocessing/'))
    with open('output/preprocessing/remap_dict.json', 'w') as handle:
            json.dump(labels, handle)
            print('remap_dict ready')
            
    side = 816
    half_side = side//2
    ll = []
    annot_path = './data/train_annotations'
    images_path = './data/train_images'
    save_path = './output/preprocessing/resized_images/'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        print('save_path done')
    with open('./output/preprocessing/remap_dict.json', 'r') as f:
        remap_dict = json.load(f)
    annotations_result = []

    all_ids = os.listdir(annot_path)
    for _file in folder:
        _fid = _file.split('.')[0]
        image_path = os.path.join(images_path, '{}.jpg'.format(_fid))
        image = cv2.imread(image_path)

        h, w, c = image.shape
        i = 0
        #нарезаю рисунки на квадраты 816px с отступом друг от друга на 408px
        for ww in range(w//half_side):
            for hh in range(h//half_side):
                w0 = ww*half_side
                h0 = hh*half_side
                h1 = h0+side
                w1 = w0+side
                if w1>w or h1>h:
                    break
                image1 = image[h0:h1, w0:w1]
                file_path = os.path.join(annot_path, _file)

                annot_instance = {}
                annot_instance['filename'] = 'output/preprocessing/resized_images/{}_{:02d}.jpg'.format(_fid, i)
                annot_instance['height'] = image1.shape[0]
                annot_instance['width'] = image1.shape[1]
                annot_instance['ann'] = {}

                with open(file_path, 'r') as f:
                    annot_data = json.load(f)
                boxes = []
                labels = []

                for label in annot_data['labels']:
                    class_label = 1#label['category']
                    x1 = int(label['box2d']['x1'])
                    x2 = int(label['box2d']['x2'])
                    y1 = int(label['box2d']['y1'])
                    y2 = int(label['box2d']['y2'])
                    if 0<=x1-w0<=side and 0<=x2-w0<=side and 0<=y1-h0<=side and 0<=y2-h0<=side:
                        boxes.append([x1-w0, y1-h0, x2-w0, y2-h0])
                        labels.append(class_label)

                # записываю аннотации и картинки только с коробками, пустые пропускаю
                if len(labels)>0:
                    cv2.imwrite(os.path.join(save_path, '{}_{:02d}.jpg'.format(_fid, i)), image1)
                    annot_instance['ann']['bboxes'] = (np.array(boxes)).astype(np.float32)
                    annot_instance['ann']['labels'] = (np.array(labels)).astype(np.int64)
                    annot_instance['ann']['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
                    annot_instance['ann']['labels_ignore'] = (np.array([])).astype(np.int64)
                    annotations_result.append(annot_instance)
                i+=1
    if folder == train:
        fol = 'train'
    else: 
        fol = 'val'
    mmcv.dump(annotations_result, './output/preprocessing/'+fol+'_mmannotations.pkl')
    print('annot ready')