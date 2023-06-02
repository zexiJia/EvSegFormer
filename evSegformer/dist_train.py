import argparse
import datetime
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from core.model import WeTr, WeTr_Ori
from utils import eval_seg
from utils.optimizer import PolyWarmupAdamW
from eval import MscEval, MscEval2
from torch.utils.data import DataLoader
import albumentations as albu
from dataloader import *
import segmentation_models_pytorch as smp
from torchstat import stat

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5679'

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/ddd17.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')
parser.add_argument('--epochs', default=400)
parser.add_argument('--batch', default=66)
parser.add_argument('--resume', default='', help='choose resume path')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0, rotate_limit=15, shift_limit=0.25, p=0.9, border_mode=cv2.BORDER_CONSTANT,value=0, mask_value=255),

        albu.RandomCrop(height=200, width=344, always_apply=True),
        
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.CenterCrop(200, 344, always_apply=True)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class PolyLR():
    def __init__(self, start_lr, lr_power, total_iters):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_iter):
        return self.start_lr * (
                (1 - float(cur_iter) / self.total_iters) ** self.lr_power)

def main(cfg):

    num_workers = 4
    
    torch.cuda.set_device(3)

    dist.init_process_group(backend="nccl", init_method="env://", world_size=1,rank=0)


    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset =  EventAPS_Dataset(mode='train',augmentation=get_training_augmentation())

    iters_per_epoch = len(train_dataset) // (args.batch)
    
    max_iter= args.epochs * iters_per_epoch
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              num_workers=num_workers,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True)

    device = 'cuda'
    
    wetr = WeTr(backbone=cfg.exp.backbone,
                num_classes=6,
                embedding_dim=256,
                pretrained=False)
    
    
    wetr.to(device)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = max_iter,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    wetr = wetr.to(device)
    # criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    criterion = criterion.to(device)

    epoch_start = 0

    if args.resume != '':
        checkpoint = torch.load(args.resume,map_location='cpu')
        wetr.load_state_dict(checkpoint['trans_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1

    results = {'iou': [],'miou': [],'acc': []}
    
    best_miou = 0


    for i in range(epoch_start, args.epochs):

        print('start train epoch '+ str(i))

        for j, (event,label) in enumerate(train_loader):
            inputs = event.to(device,non_blocking=True)
            labels = label.to(device,non_blocking=True)
            labels = labels.squeeze()
            outputs1 = wetr(inputs)
            outputs1 = F.interpolate(outputs1, size=labels.shape[1:], mode='bilinear', align_corners=False)
            loss1 = criterion(outputs1, labels.type(torch.long))
            seg_loss = loss1
            optimizer.zero_grad()
            seg_loss.backward()
            optimizer.step()
            if j %10 == 0:
                print('iter',str(j),'  loss',seg_loss.item())
        
        evaluator = MscEval()
        iu, mean_IU, freq, mean_pixel_acc = evaluator(wetr)

        print('iou: ',iu)
        print('miou: ',mean_IU)
        print('acc: ',mean_pixel_acc)
        print('freq: ',freq)

        if mean_IU > best_miou:
            torch.save({'epoch': i, 'trans_state_dict': wetr.state_dict(), 'optimizer' : optimizer.state_dict(),}, 'evsegformer/event_k/'+'model_'+str(i)+'.pth')
            best_miou = mean_IU


    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    main(cfg)