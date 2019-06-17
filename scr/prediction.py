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
import fastai
import fastai.vision
import pathlib
import functools
import collections
import torch
import pretrainedmodels
import gc
import random
from fastai.torch_core import nn
import csv

def model_50(pretrained=True, **kwargs):
    return pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
def model_161(pretrained=True, **kwargs):
    return pretrainedmodels.densenet161(num_classes=1000, pretrained='imagenet')
def model_152(pretrained=True, **kwargs):
    return pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet')
def model_101(pretrained=True, **kwargs):
    return pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet')
        
            
def predict():
    save_path = 'output/predictions/nn_models/'
    save_path1 = 'output/predictions/nn_models/test/'
    if not os.path.exists(os.path.dirname(save_path1)):
        os.makedirs(os.path.dirname(save_path1))
    save_path2 = 'output/predictions/nn_models/valid/'
    if not os.path.exists(os.path.dirname(save_path2)):
        os.makedirs(os.path.dirname(save_path2))
        
    mods = [str(50), str(101), str(152), str(161)]  

    path = pathlib.PosixPath('output/preprocessing')
    classes = os.listdir('output/preprocessing/croped_images/')

    bs = 50
    size = 224
    version = str(0)

    tfms = fastai.vision.transform.get_transforms(do_flip=False, 
                                                  flip_vert=False, max_rotate=10,
                                                  max_zoom=1.1, max_lighting=0.2, 
                                                  p_affine=0.85, p_lighting=0.4, 
                                                  max_warp=0.2, xtra_tfms=None)
    data_test = (fastai.vision.ImageList.from_folder(path)
                    .split_none()
                    .label_from_folder()
                    .transform(tfms, size=size)
                    .add_test_folder()
                    .databunch(bs=bs)
                .normalize(fastai.vision.imagenet_stats))
    
    with open(save_path+'test_items.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(data_test.test_ds.items)


    im_res= {}
    folds = 10
    
    for mod in mods:
        valid_items = []
        for fold in range(folds):        
            train_df = pd.read_csv('output/preprocessing/folds/fold_{}.csv'.format(fold), dtype=str)
            val_df = pd.read_csv('output/preprocessing/folds/val_{}.csv'.format(fold), dtype=str)
            data = (fastai.vision.ImageList.from_df(df=train_df, path=path/'croped_images', cols='name')
                    .split_none()
                    .label_from_df(cols='label', classes=classes)
                    .transform(tfms, size=size)
                    .databunch(bs=bs)
                    .normalize(fastai.vision.imagenet_stats))

            valid = (fastai.vision.ImageList.from_df(df=val_df, path=path/'croped_images', cols='name')
                    .split_none()
                    .label_from_df(cols='label', classes=classes)
                    .transform(tfms, size=size)
                    .databunch(bs=bs)
                    .normalize(fastai.vision.imagenet_stats))
        
            data.valid_dl = valid.train_dl
            if mod == '50':
                learn = fastai.vision.learner.cnn_learner(
                data = data,
                base_arch=model_50, 
                cut=-2,
                ps=0.2, 
                metrics=fastai.metrics.accuracy,
                model_dir='../../training/models')
            elif mod == '101':
                learn = fastai.vision.learner.cnn_learner(
                data = data,
                base_arch=model_101, 
                cut=-2,
                ps=0.2, 
                metrics=fastai.metrics.accuracy,
                model_dir='../../training/models')
            elif mod == '152':
                learn = fastai.vision.learner.cnn_learner(
                data = data,
                base_arch=model_152, 
                cut=-2,
                ps=0.2, 
                metrics=fastai.metrics.accuracy,
                model_dir='../../training/models')
            elif mod == '161':
                learn = fastai.vision.learner.cnn_learner(
                data = data,
                base_arch=model_161, 
                cut=-1,
                ps=0.2, 
                metrics=fastai.metrics.accuracy,
                model_dir='../../training/models')
            
            learn.model = nn.DataParallel(learn.model)
            learn.load('model_'+mod+'_'+version+'_fold_'+str(fold));

            learn.data = data_test
            pr, y = learn.TTA(ds_type=fastai.basic_data.DatasetType.Test)
            torch.save(pr, save_path+'test/test_'+mod+'_'+version+'_fold_'+str(fold)+'.pt')

            learn = None
            data = None
            gc.collect()
            torch.cuda.empty_cache()

def predict_val():
    save_path = 'output/predictions/nn_models/'
    save_path1 = 'output/predictions/nn_models/test/'
    if not os.path.exists(os.path.dirname(save_path1)):
        os.makedirs(os.path.dirname(save_path1))
    save_path2 = 'output/predictions/nn_models/valid/'
    if not os.path.exists(os.path.dirname(save_path2)):
        os.makedirs(os.path.dirname(save_path2))
        
    mods = [str(50), str(101), str(152), str(161)] 

    path = pathlib.PosixPath('output/preprocessing')
    classes = os.listdir('output/preprocessing/croped_images/')

    bs = 1
    size = 224
    version = str(0)

    tfms = fastai.vision.transform.get_transforms(do_flip=False, 
                                                  flip_vert=False, max_rotate=10,
                                                  max_zoom=1.1, max_lighting=0.2, 
                                                  p_affine=0.85, p_lighting=0.4, 
                                                  max_warp=0.2, xtra_tfms=None)
    data_test = (fastai.vision.ImageList.from_folder(path)
                    .split_none()
                    .label_from_folder()
                    .transform(tfms, size=size)
                    .add_test_folder()
                    .databunch(bs=bs)
                .normalize(fastai.vision.imagenet_stats))
    
    with open(save_path+'test_items.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(data_test.test_ds.items)


    im_res= {}
    folds = 10
    
    for mod in mods:
        valid_items = []
        for fold in range(folds):        
            train_df = pd.read_csv('output/preprocessing/folds/fold_{}.csv'.format(fold), dtype=str)
            val_df = pd.read_csv('output/preprocessing/folds/val_{}.csv'.format(fold), dtype=str)
            data = (fastai.vision.ImageList.from_df(df=train_df, path=path/'croped_images', cols='name')
                    .split_none()
                    .label_from_df(cols='label', classes=classes)
                    .transform(tfms, size=size)
                    .databunch(bs=bs)
                    .normalize(fastai.vision.imagenet_stats))

            valid = (fastai.vision.ImageList.from_df(df=val_df, path=path/'croped_images', cols='name')
                    .split_none()
                    .label_from_df(cols='label', classes=classes)
                    .transform(tfms, size=size)
                    .databunch(bs=bs)
                    .normalize(fastai.vision.imagenet_stats))
        
            data.valid_dl = valid.train_dl
            if mod == '50':
                learn = fastai.vision.learner.cnn_learner(
                data = data,
                base_arch=model_50, 
                cut=-2,
                ps=0.2, 
                metrics=fastai.metrics.accuracy,
                model_dir='../../training/models')
            elif mod == '101':
                learn = fastai.vision.learner.cnn_learner(
                data = data,
                base_arch=model_101, 
                cut=-2,
                ps=0.2, 
                metrics=fastai.metrics.accuracy,
                model_dir='../../training/models')
            elif mod == '152':
                learn = fastai.vision.learner.cnn_learner(
                data = data,
                base_arch=model_152, 
                cut=-2,
                ps=0.2, 
                metrics=fastai.metrics.accuracy,
                model_dir='../../training/models')
            elif mod == '161':
                learn = fastai.vision.learner.cnn_learner(
                data = data,
                base_arch=model_161, 
                cut=-1,
                ps=0.2, 
                metrics=fastai.metrics.accuracy,
                model_dir='../../training/models')
            
            learn.model = nn.DataParallel(learn.model)
            learn.load('model_'+mod+'_'+version+'_fold_'+str(fold));
            p_v = learn.TTA()
            torch.save(p_v, save_path+'valid/valid_'+mod+'_'+version+'_fold_'+str(fold)+'.pt')
        
            learn = None
            data = None
            gc.collect()
            torch.cuda.empty_cache()

