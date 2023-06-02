#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from dataloader import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import sys
import os.path as osp
import logging
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as albu
from core.model import WeTr, WeTr_Ori
import time
from torchvision import transforms
from collections import namedtuple

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.CenterCrop(200, 344, always_apply=True),
    ]
    return albu.Compose(test_transform)

def get_validation_augmentation_75():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.CenterCrop(200, 340, always_apply=True),
        albu.Resize(int(200*0.75),int(344*0.75)),
    ]
    return albu.Compose(test_transform)

def get_validation_augmentation_150():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.CenterCrop(200, 340, always_apply=True),
        albu.Resize(int(200*1.5),int(344*1.5)),
    ]
    return albu.Compose(test_transform)

def fromIdTrainToId(imgin):
    Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'eventId'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'color'       , # The color of this label
    ] )

    labels = [
    #       name                     eventId    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  255 ,      255 , 'void'            , 0       ,  (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  255 ,      255 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'rectification border' ,  255 ,      255 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'out of roi'           ,  255 ,      255 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'static'               ,  255 ,      255 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'dynamic'              ,  255 ,      255 , 'void'            , 0       , (111, 74,  0) ),
    Label(  'ground'               ,  255 ,      255 , 'void'            , 0       , ( 81,  0, 81) ),
    Label(  'road'                 ,  0 ,        0 , 'flat'            , 1       , (128, 64,128) ),
    Label(  'sidewalk'             ,  0 ,        1 , 'flat'            , 1      ,  (244, 35,232) ),
    Label(  'parking'              ,  0 ,      255 , 'flat'            , 1       , (250,170,160) ),
    Label(  'rail track'           , 0 ,      255 , 'flat'            , 1       , (230,150,140) ),
    Label(  'building'             , 1 ,        2 , 'construction'    , 2       , ( 70, 70, 70) ),
    Label(  'wall'                 , 1 ,        3 , 'construction'    , 2       , (102,102,156) ),
    Label(  'fence'                , 1 ,        4 , 'construction'    , 2      ,  (190,153,153) ),
    Label(  'guard rail'           , 1 ,      255 , 'construction'    , 2       , (180,165,180) ),
    Label(  'bridge'               , 1 ,      255 , 'construction'    , 2       , (150,100,100) ),
    Label(  'tunnel'               , 1 ,      255 , 'construction'    , 2       , (150,120, 90) ),
    Label(  'pole'                 , 2 ,        5 , 'object'          , 3       , (153,153,153) ),
    Label(  'polegroup'            , 2 ,      255 , 'object'          , 3       , (153,153,153) ),
    Label(  'traffic light'        , 2 ,        6 , 'object'          , 3      ,  (250,170, 30) ),
    Label(  'traffic sign'         , 2 ,        7 , 'object'          , 3      ,  (220,220,  0) ),
    Label(  'vegetation'           , 3 ,        8 , 'nature'          , 4      ,  (107,142, 35) ),
    Label(  'terrain'              , 3 ,        9 , 'nature'          , 4     ,   (152,251,152) ),
    Label(  'sky'                  , 1 ,       10 , 'sky'             , 5      ,  ( 70,130,180) ),
    Label(  'person'               , 4 ,       11 , 'human'           , 6       ,  (220, 20, 60) ),
    Label(  'rider'                , 4 ,       12 , 'human'           , 6       ,  (255,  0,  0) ),
    Label(  'car'                  , 5 ,       13 , 'vehicle'         , 7       ,  (  0,  0,142) ),
    Label(  'truck'                , 5 ,       14 , 'vehicle'         , 7       ,  (  0,  0, 70) ),
    Label(  'bus'                  , 5 ,       15 , 'vehicle'         , 7       ,  (  0, 60,100) ),
    Label(  'caravan'              , 5 ,      255 , 'vehicle'         , 7       , (  0,  0, 90) ),
    Label(  'trailer'              , 5 ,      255 , 'vehicle'         , 7       , (  0,  0,110) ),
    Label(  'train'                , 5 ,       16 , 'vehicle'         , 7       ,  (  0, 80,100) ),
    Label(  'motorcycle'           , 5 ,       17 , 'vehicle'         , 7       ,  (  0,  0,230) ),
    Label(  'bicycle'              , 5 ,       18 , 'vehicle'         , 7       ,  (119, 11, 32) ),
    Label(  'license plate'        , 255 ,       -1 , 'vehicle'         , 7       , (  0,  0,142) ),
    ]

    trainId2label   = { label.trainId : label for label in reversed(labels) }
    imgout = imgin.copy()
    for idTrain in trainId2label:
        imgout[imgin == idTrain] = trainId2label[idTrain].eventId
    
    return imgout


class MscEval(object):
    def __init__(self, *args, **kwargs):

        self.distributed = None#dist.is_initialized()
        ## dataloader
        dsval = EventAPS_Dataset(mode='test',augmentation=get_validation_augmentation())
        sampler = None
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dsval)
        
        self.dl = DataLoader(dsval,
                        batch_size = 1,
                        sampler = sampler,
                        shuffle = False,
                        num_workers = 10,
                        drop_last = False)

    def __call__(self, net):
        ## evaluate
        n_classes = 6
        ignore_label = 255

        hist_size = (6, 6)
        hist = np.zeros(hist_size, dtype=np.float32)

        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(self.dl)
        else:
            diter = enumerate(self.dl)
        print('start eval')
        start = time.clock()
        for i, (events,label) in diter:
            label = label.cuda()
            events = events.cuda()
            label = label.squeeze()

            with torch.no_grad():
                start = time.clock()
                output_ev = net(events)
                end = time.clock()
                print(end-start)
                output_ev = F.interpolate(output_ev, size=label.shape, mode='bilinear', align_corners=False)
                preds = torch.argmax(output_ev.squeeze(), dim=0).detach().cpu().numpy()
            label = label.int()
            label = label.cpu().numpy()
            keep = label != ignore_label
            hist_once = np.bincount(label[keep] * n_classes + preds[keep], minlength=n_classes ** 2).reshape(n_classes,
                                                                                                             n_classes)
            hist = hist + hist_once

        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        freq = (hist.sum(1),hist.sum())
        label = label.squeeze()
        
        k = (label >= 0) & (label < n_classes)
        labeled = np.sum(k)
        correct = np.sum((preds[k] == label[k]))
        mean_pixel_acc = correct / labeled
        end = time.clock()
        print('time', end-start)

        return iu, mean_IU, freq, mean_pixel_acc

class MscEval_2c(object):
    def __init__(self, *args, **kwargs):

        self.distributed = None#dist.is_initialized()
        ## dataloader
        dsval = EventAPS_Dataset(mode='test',augmentation=get_validation_augmentation())
        sampler = None
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dsval)
        
        self.dl = DataLoader(dsval,
                        batch_size = 1,
                        sampler = sampler,
                        shuffle = False,
                        num_workers = 10,
                        drop_last = False)

    def __call__(self, net):
        ## evaluate
        n_classes = 6
        ignore_label = 255

        hist_size = (6, 6)
        hist = np.zeros(hist_size, dtype=np.float32)

        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(self.dl)
        else:
            diter = enumerate(self.dl)
        print('start eval')
        start = time.clock()
        for i, (events, label) in diter:
            label = label.cuda()
            events = events.cuda()
            label = label.squeeze()

            with torch.no_grad():
                events = events[:,0:2,:,:]
                output_ev = net(events)
                output_ev = F.interpolate(output_ev, size=label.shape, mode='bilinear', align_corners=False)
                preds = torch.argmax(output_ev.squeeze(), dim=0).detach().cpu().numpy()
            label = label.int()
            label = label.cpu().numpy()
            keep = label != ignore_label
            hist_once = np.bincount(label[keep] * n_classes + preds[keep], minlength=n_classes ** 2).reshape(n_classes,
                                                                                                             n_classes)
            hist = hist + hist_once

        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        freq = (hist.sum(1),hist.sum())
        label = label.squeeze()
        
        k = (label >= 0) & (label < n_classes)
        labeled = np.sum(k)
        correct = np.sum((preds[k] == label[k]))
        mean_pixel_acc = correct / labeled
        end = time.clock()
        print('time', end-start)

        return iu, mean_IU, freq, mean_pixel_acc


class MscEvalMs(object):
    def __init__(self, *args, **kwargs):

        self.distributed = None#dist.is_initialized()
        ## dataloader
        dsval = EventAPS_Dataset_MS(mode='test',augmentation1=get_validation_augmentation(), augmentation2=get_validation_augmentation_75(), augmentation3=get_validation_augmentation_150())
        sampler = None
        
        self.dl = DataLoader(dsval,
                        batch_size = 1,
                        sampler = sampler,
                        shuffle = False,
                        num_workers = 10,
                        drop_last = False)

    def __call__(self, net1, net2):
        ## evaluate
        n_classes = 6
        ignore_label = 255

        hist_size = (6, 6)
        hist = np.zeros(hist_size, dtype=np.float32)

        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(self.dl)
        else:
            diter = enumerate(self.dl)
        print('start eval')
        start = time.clock()

        
        for i, (events_a, events_b, events_c, label,_,_) in diter:
            label = label.cuda()
            events_a = events_a.cuda()
            events_b = events_b.cuda()
            events_c = events_c.cuda()
            label = label.squeeze()

            events = [events_a, events_b, events_c]

            y_ = np.zeros([6,200,344])
            with torch.no_grad():
                start = time.clock()
                for j in range(len(events)):
                    output1 = net1(events[j])
                    output1 = F.interpolate(output1, size=label.shape, mode='bilinear', align_corners=False)
                    output_ev = output1.data.cpu().numpy()

                    # Flip images in numpy (not support in tensor)
                    flipped_images = np.copy(events[j].data.cpu().numpy()[:, :, :, ::-1])
                    flipped_images = torch.from_numpy(flipped_images).float().cuda()
                    output1 = net1(flipped_images)
                    output1 = F.interpolate(output1, size=label.shape, mode='bilinear', align_corners=False)
                    output2 = net2(flipped_images[:,0:2,:,:])
                    output2 = F.interpolate(output2, size=label.shape, mode='bilinear', align_corners=False)
                    output_flipped = (output1 + output2) / 2
                    #output_flipped[:,3:4,:,:] *= 0.8
                    output_flipped = output_flipped.data.cpu().numpy()
                    
                    outputs = (output_ev + output_flipped[:, :, :, ::-1]) / 2.0

                    outputs = outputs[0]

                    y_ += outputs
                end = time.clock()
                print(end-start)
                preds = np.argmax(y_, axis=0)
            
            label = label.int()
            label = label.cpu().numpy()
            keep = label != ignore_label
            hist_once = np.bincount(label[keep] * n_classes + preds[keep], minlength=n_classes ** 2).reshape(n_classes,
                                                                                                             n_classes)                                                                                          
            hist = hist + hist_once

        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        freq = (hist.sum(1),hist.sum())
        label = label.squeeze()
        
        k = (label >= 0) & (label < n_classes)
        labeled = np.sum(k)
        correct = np.sum((preds[k] == label[k]))
        mean_pixel_acc = correct / labeled

        end = time.clock()
        print('time', end-start)

        return iu, mean_IU, freq, mean_pixel_acc


if __name__ == "__main__":
    

    model = WeTr(backbone='mit_b3',
                num_classes=6,
                embedding_dim=256,
                pretrained=False)

    checkpoint = torch.load('segformer-pytorch/event_soft/model_24.pth',map_location='cpu')
    
    model.eval()
    model.cuda()

    evaluator = MscEval()
    evaluator_ms = MscEvalMs()
    iu, mean_IU, freq, mean_pixel_acc = evaluator(model)
    print('iou: ',iu)
    print('miou: ',mean_IU)
    print('acc: ',mean_pixel_acc)
    print('freq: ',freq)

