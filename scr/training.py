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

class FocalLoss(fastai.torch_core.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()
    
def model_50(pretrained=True, **kwargs):
    return pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
def model_161(pretrained=True, **kwargs):
    return pretrainedmodels.densenet161(num_classes=1000, pretrained='imagenet')
def model_152(pretrained=True, **kwargs):
    return pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet')
def model_101(pretrained=True, **kwargs):
    return pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet')

def write_log(i, version, learn, mod):
    l = len(learn.recorder.losses)
    m = len(learn.recorder.metrics)
    for k in range(m):
        log = 'fold_' + str(i) + str(k) + " " + str(learn.recorder.losses[(k+1)*l//m-1].item()) + " " + \
                                 str(learn.recorder.val_losses[k].item()) + " " + \
                                 str(learn.recorder.metrics[k][0].item())

        with open("output/training/models/log_"+mod+"_"+version+".txt", "a") as f:
            f.write(log + "\n")
    with open("output/training/models/log_"+mod+"_"+version+".txt", "a") as f:
            f.write("\n")
            
            
def train_50():
    mod = str(50)
    if not os.path.exists(os.path.dirname('output/training/models')):
            os.makedirs(os.path.dirname('output/training/models'))

    path = pathlib.PosixPath('output/preprocessing')
    classes = os.listdir('output/preprocessing/croped_images/')

    bs = 140
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

    im_res= {}
    folds = 10
    for fold in range(folds):
        print(fold)
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

        learn = fastai.vision.learner.cnn_learner(
            data = data,
            base_arch=model_50, 
            cut=-2,
            ps=0.2, 
            metrics=fastai.metrics.accuracy,
            model_dir='../../training/models')
        learn.loss_fn = FocalLoss()
        learn.crit = FocalLoss()
        learn.model = nn.DataParallel(learn.model)

        learn.fit_one_cycle(3, max_lr=3e-3) #3
        write_log(fold, version, learn, mod)

        learn.unfreeze()

        lr = 9e-4
        lrs=np.array([lr/10,  lr/3])
        learn.fit_one_cycle(4, lrs)
        write_log(fold, version, learn, mod)

#         lr = 5e-4
#         lrs=np.array([lr/10,  lr/3])
#         learn.fit_one_cycle(4, lrs)
#         write_log(fold, version, learn, mod)

        learn.freeze()
        learn.save('model_'+mod+'_'+version+'_fold_'+str(fold))    

        learn = None
        data = None
        gc.collect()
        torch.cuda.empty_cache()
        
def train_161():
    mod = str(161)
    if not os.path.exists(os.path.dirname('output/training/models')):
            os.makedirs(os.path.dirname('output/training/models'))

    path = pathlib.PosixPath('output/preprocessing')
    classes = os.listdir('output/preprocessing/croped_images/')

    bs = 70
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

    im_res= {}
    folds = 10
    for fold in range(folds):
        print(fold)
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

        learn = fastai.vision.learner.cnn_learner(
            data = data,
            base_arch=model_161, 
            cut=-1,
            ps=0.2, 
            metrics=fastai.metrics.accuracy,
            model_dir='../../training/models')
        learn.loss_fn = FocalLoss()
        learn.crit = FocalLoss()
        learn.model = nn.DataParallel(learn.model)

        learn.fit_one_cycle(3, max_lr=3e-3) #3
        write_log(fold, version, learn, mod)

        learn.unfreeze()

        lr = 9e-4
        lrs=np.array([lr/10,  lr/3])
        learn.fit_one_cycle(4, lrs)
        write_log(fold, version, learn, mod)

#         lr = 5e-4
#         lrs=np.array([lr/10,  lr/3])
#         learn.fit_one_cycle(4, lrs)
#         write_log(fold, version, learn, mod)

        learn.freeze()
        learn.save('model_'+mod+'_'+version+'_fold_'+str(fold))    

        learn = None
        data = None
        gc.collect()
        torch.cuda.empty_cache()
        
def train_101():
    mod = str(101)
    if not os.path.exists(os.path.dirname('output/training/models')):
            os.makedirs(os.path.dirname('output/training/models'))

    path = pathlib.PosixPath('output/preprocessing')
    classes = os.listdir('output/preprocessing/croped_images/')

    bs = 140
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

    im_res= {}
    folds = 10
    for fold in range(folds):
        print(fold)
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

        learn = fastai.vision.learner.cnn_learner(
            data = data,
            base_arch=model_101, 
            cut=-2,
            ps=0.2, 
            metrics=fastai.metrics.accuracy,
            model_dir='../../training/models')
        learn.loss_fn = FocalLoss()
        learn.crit = FocalLoss()
        learn.model = nn.DataParallel(learn.model)

        learn.fit_one_cycle(3, max_lr=3e-3) #3
        write_log(fold, version, learn, mod)

        learn.unfreeze()

        lr = 9e-4
        lrs=np.array([lr/10,  lr/3])
        learn.fit_one_cycle(4, lrs)
        write_log(fold, version, learn, mod)

#         lr = 5e-4
#         lrs=np.array([lr/10,  lr/3])
#         learn.fit_one_cycle(4, lrs)
#         write_log(fold, version, learn, mod)

        learn.freeze()
        learn.save('model_'+mod+'_'+version+'_fold_'+str(fold))    

        learn = None
        data = None
        gc.collect()
        torch.cuda.empty_cache()
        
def train_152():
    mod = str(152)
    if not os.path.exists(os.path.dirname('output/training/models')):
            os.makedirs(os.path.dirname('output/training/models'))

    path = pathlib.PosixPath('output/preprocessing')
    classes = os.listdir('output/preprocessing/croped_images/')

    bs = 70
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

    im_res= {}
    folds = 10
    for fold in range(folds):
        print(fold)
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

        learn = fastai.vision.learner.cnn_learner(
            data = data,
            base_arch=model_152, 
            cut=-2,
            ps=0.2, 
            metrics=fastai.metrics.accuracy,
            model_dir='../../training/models')
        learn.loss_fn = FocalLoss()
        learn.crit = FocalLoss()
        learn.model = nn.DataParallel(learn.model)

        learn.fit_one_cycle(3, max_lr=3e-3) #3
        write_log(fold, version, learn, mod)

        learn.unfreeze()

        lr = 9e-4
        lrs=np.array([lr/10,  lr/3])
        learn.fit_one_cycle(4, lrs)
        write_log(fold, version, learn, mod)

#         lr = 5e-4
#         lrs=np.array([lr/10,  lr/3])
#         learn.fit_one_cycle(4, lrs)
#         write_log(fold, version, learn, mod)

        learn.freeze()
        learn.save('model_'+mod+'_'+version+'_fold_'+str(fold))    

        learn = None
        data = None
        gc.collect()
        torch.cuda.empty_cache()
