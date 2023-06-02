

# Ev-SegFormer


This work proses an approach for learning semantic segmentation from only event-based information (event-based cameras).


# Requirements
torchvision==0.8.2
imageio==2.9.0
omegaconf==2.0.0
timm==0.4.5
mmcv_full==1.3.8
torch==1.7.1
tqdm==4.46.1
opencv_python==4.4.0.46
numpy==1.18.5
mmcv==1.3.8
Pillow==8.3.1



## Dataset
Our dataset is a subset of the [DDD17: DAVIS Driving Dataset](http://sensors.ini.uzh.ch/news_page/DDD17.html). This original dataset do not provide any semantic segmentation label, we provide them as well as some modification of the event images.


[Download it here](https://drive.google.com/open?id=1Ug6iZc7WYQWCklxwcemCeyw3CPyuuxJf)

The semantic segmentation labels of the data are:
flat:0, construction+sky:1, object:2,  nature:3,  human:4, vehicle:5, ignore_labels:255


## train

```
python dist_train.py
```

## test


```
python eval.py
```


## Get new labels

First, download [this folder](https://drive.google.com/drive/folders/1NjTGAoSCpYw_l89l1BdulJi0B_qIIiin?usp=sharing) and copy it into the weights folder of this repository (so that you have weights/cityscapes_grasycale folder).

Then execute this script specifying the grayscale image path
