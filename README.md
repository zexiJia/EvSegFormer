# Event-Based Semantic Segmentation With Posterior Attention

This repository contains the code associated with our paper [Event-Based Semantic Segmentation With Posterior Attention](https://ieeexplore.ieee.org/document/10058930).

## Introduction
This work proses an approach for learning semantic segmentation from only event-based information (event-based cameras). Pioneering researchers stack event data as frames so that event-based segmentation is converted to frame-based segmentation, but characteristics of event data are not explored. Noticing that event data naturally highlight moving objects, we propose a posterior attention module that adjusts the standard attention by the prior knowledge provided by event data. The posterior attention module can be readily plugged into many segmentation backbones. Plugging the posterior attention module into a recently proposed SegFormer network, we get EvSegFormer (the event-based version of SegFormer) with state-of-the-art performance in two datasets (MVSEC and DDD-17) collected for event-based segmentation. 

For more details, here is the [Paper](https://ieeexplore.ieee.org/document/10058930).


# Requirements
Python 3.6+

Pytorch 1.10+

Opencv

Imgaug

Sklearn

## Citations

If you find this code useful in your research, please consider citing:

[Jia, Zexi, et al. "Event-Based Semantic Segmentation With Posterior Attention." IEEE Transactions on Image Processing 32 (2023): 1829-1842.](https://ieeexplore.ieee.org/document/10058930)

```
@article{jia2023event,
  title={Event-Based Semantic Segmentation With Posterior Attention},
  author={Jia, Zexi and You, Kaichao and He, Weihua and Tian, Yang and Feng, Yongxiang and Wang, Yaoyuan and Jia, Xu and Lou, Yihang and Zhang, Jingyi and Li, Guoqi and others},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={1829--1842},
  year={2023},
  publisher={IEEE}
}
```
