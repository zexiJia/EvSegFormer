import os
import cv2
import torch
import numpy as np
from torch.utils import data
import random
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from numpy.lib.stride_tricks import as_strided
from einops import rearrange
import albumentations as albu
import glob
import cv2
from augmenters import get_augmenter
from torchvision import transforms


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class EventAPS_Dataset(Dataset):

    def __init__(self,
                mode,
                augmentation=None,
                preprocessing=None,
                unsupervised=False,
                ):

        root = '/home/zexi/Ev-SegNet/dataset_our_codification'
        self.mode = mode

        root1 = root +'/events'
        root2 = root +'/images'
        root3 = root +'/labels'

        if mode == 'train' or mode == 'test':
            root1 = root1 +'/' + mode
            root2 = root2 +'/' + mode
            root3 = root3 +'/' + mode
        else:
            raise TypeError('this mode is not exist')
        self.root1 = root1
        self.root2 = root2
        self.root3 = root3
        
        self.events, self.imgs, self.labels = self.get_all_images_path()

        self.unsupervised= unsupervised
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def get_all_images_path(self):
        # ! 只获取图像 path

        events = glob.glob(f'{self.root1}/*')
        imgs = glob.glob(f'{self.root2}/*')
        labels = glob.glob(f'{self.root3}/*')

        events.sort()
        imgs.sort()
        labels.sort()

        return events, imgs, labels

    def __getitem__(self, index):
        event = self.events[index]
        img= self.imgs[index]
        label = self.labels[index]
        events_path = event.split('/')[-1].split('.')[0]
        img_path = img.split('/')[-1].split('.')[0]
        assert events_path == img_path, f'img is not the same as event'
        label_path = label.split('/')[-1].split('.')[0]
        assert events_path == label_path, f'img is not the same as event'
        img1 = cv2.imread(img, flags=0)
        event1 = np.load(event)
        #event1 = event1[:,:,0:2]
        label1 = cv2.imread(label,flags = 0)
                # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=event1, mask=label1)
            event, label = sample['image'], sample['mask']

        label = label[:,:,None]

        event = to_tensor(event)
        label = to_tensor(label)
        return event, label

    def __len__(self):

        return len(self.events)

class EventAPS_Dataset_Nolabel(Dataset):

    def __init__(self,
                mode,
                augmentation=None,
                preprocessing=None,
                unsupervised=False,
                ):
        root = '/home/zexi/Ev-SegNet/dataset_our_codification'
        self.mode = mode

        root1 = root +'/events'
        root2 = root +'/images'
        root3 = root +'/labels'

        if mode == 'train' or mode == 'test':
            root1 = root1 +'/' + mode
            root2 = root2 +'/' + mode
            root3 = root3 +'/' + mode
        else:
            raise TypeError('this mode is not exist')
        self.root1 = root1
        self.root2 = root2
        self.root3 = root3
        
        self.events, self.imgs = self.get_all_images_path()

        self.unsupervised= unsupervised
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def get_all_images_path(self):
        # ! 只获取图像 path

        events = glob.glob(f'{self.root1}/*')
        imgs = glob.glob(f'{self.root2}/*')

        events.sort()
        imgs.sort()

        return events, imgs

    def __getitem__(self, index):
        event = self.events[index]
        img= self.imgs[index]
        events_path = event.split('/')[-1].split('.')[0]
        img_path = img.split('/')[-1].split('.')[0]
        assert events_path == img_path, f'img is not the same as event'
        img1 = cv2.imread(img, flags=0)
        event1 = np.load(event)
        if self.augmentation:
            data = {"image": event1, "mask": img1}
            augmented = self.augmentation(**data)
            event, img = augmented["image"], augmented["mask"]

        img = img[:,:,None]

        event = to_tensor(event)
        img = to_tensor(img)
        return event, img

    def __len__(self):

        return len(self.events)



class EventAPS_Dataset_MS(Dataset):

    def __init__(self,
                mode,
                augmentation1=None,
                augmentation2=None,
                augmentation3=None,
                preprocessing=None,
                unsupervised=False,
                ):

                # base_path 是数据库的 root 路径
                # ! transform 是动态的，数据库太大，所以在线读数据
        #这是原版
        root = '/home/zexi/Ev-SegNet/dataset_our_codification'
        self.mode = mode

        root1 = root +'/events'
        root2 = root +'/images'
        root3 = root +'/labels'

        if mode == 'train' or mode == 'test':
            root1 = root1 +'/' + mode
            root2 = root2 +'/' + mode
            root3 = root3 +'/' + mode
        else:
            raise TypeError('this mode is not exist')
        self.root1 = root1
        self.root2 = root2
        self.root3 = root3
        
        self.events, self.imgs, self.labels = self.get_all_images_path()

        self.unsupervised= unsupervised
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2
        self.augmentation3 = augmentation3
        self.preprocessing = preprocessing

    def get_all_images_path(self):
        # ! 只获取图像 path

        events = glob.glob(f'{self.root1}/*')
        imgs = glob.glob(f'{self.root2}/*')
        labels = glob.glob(f'{self.root3}/*')

        events.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
        imgs.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
        labels.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))

        return events, imgs, labels

    def __getitem__(self, index):
        event = self.events[index]
        img= self.imgs[index]
        label = self.labels[index]
        events_path = event.split('/')[-1].split('.')[0]
        img_path = img.split('/')[-1].split('.')[0]
        assert events_path == img_path, f'img is not the same as event'
        label_path = label.split('/')[-1].split('.')[0]
        assert events_path == label_path, f'img is not the same as event'
        img1 = cv2.imread(img, flags=0)
        event1 = np.load(event)
        #event1 = event1[:,:,0:2]
        label1 = cv2.imread(label,flags = 0)
                # apply augmentations
        if self.augmentation1:
            sample = self.augmentation1(image=event1.copy(), mask=label1.copy())
            event_a, label_a = sample['image'], sample['mask']

        if self.augmentation2:
            sample = self.augmentation2(image=event1.copy(), mask=label1.copy())
            event_b, label_b = sample['image'], sample['mask']

        
        if self.augmentation3:
            sample = self.augmentation3(image=event1.copy(), mask=label1.copy())
            event_c, label_c = sample['image'], sample['mask']

        label = label_a[:,:,None]
        event_a = to_tensor(event_a)
        event_b = to_tensor(event_b)
        event_c = to_tensor(event_c)
        img1 = img1[:,1:345]
        return event_a, event_b, event_c, label_a, img1, events_path

    def __len__(self):

        return len(self.events)


    
def calculate_weigths_labels():
# Create an instance from the data loader
    ds = EventAPS_Dataset(mode = 'train')
    sampler = None
    n_classes = 6
    ignore_label = 255

    hist_size = (6, 6)
    hist = np.zeros(hist_size, dtype=np.float32)
        
    dl = DataLoader(ds,
                    batch_size = 1,
                    sampler = sampler,
                    shuffle = False,
                    num_workers = 10,
                    drop_last = False)
    diter = enumerate(dl)
    for i, (d) in diter:
        label = d['label']


        keep = label != ignore_label
        hist_once = np.bincount(label[keep] * n_classes, minlength=n_classes ** 2).reshape(n_classes,
                                                                                            n_classes)
        hist = hist + hist_once

    freq = (hist.sum(1),hist.sum())

    return freq




