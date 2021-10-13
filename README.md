# GraNet

<img src="https://github.com/Shiweiliuiiiiiii/GraNet/blob/main/GraNet_github.png" width="500" height="300">


**Sparse Training via Boosting Pruning Plasticity with Neuroregeneration**<br>
Shiwei Liu,Tianlong Chen,Xiaohan Chen,Zahra Atashgahi,Lu Yin,Huanyu Kou,Li Shen,Mykola Pechenizkiy,Zhangyang Wang, Decebal Constantin Mocanu<br>
https://arxiv.org/abs/2106.10404<br>

Abstract: *Works on lottery ticket hypothesis (LTH) and single-shot network pruning (SNIP) have raised a lot of attention currently on post-training pruning (iterative magnitude pruning), and before-training pruning (pruning at initialization). The former method suffers from an extremely large computation cost and the latter category of methods usually struggles with insufficient performance. In comparison, during-training pruning, a class of pruning methods that simultaneously enjoys the training/inference efficiency and the comparable performance, temporarily, has been less explored. To better understand during-training pruning, we quantitatively study the effect of pruning throughout training from the perspective of pruning plasticity (the ability of the pruned networks to recover the original performance). Pruning plasticity can help explain several other empirical observations about neural network pruning in literature. We further find that pruning plasticity can be substantially improved by injecting a brain-inspired mechanism called neuroregeneration, i.e., to regenerate the same number of connections as pruned. We design a novel gradual magnitude pruning (GMP) method, named gradual pruning with zero-cost neuroregeneration, GraNet, advancing state of the art. Perhaps most impressively, GraNet for the first time boosts the sparse-to-sparse training performance over various dense-to-sparse methods by a large margin with ResNet-50 on ImageNet without extending the training time.*


This code base is created by Shiwei Liu [s.liu3@tue.nl](mailto:s.liu3@tue.nl) during his Ph.D. at Eindhoven University of Technology.<br>

This repository contains implementaions of sparse training methods: [GraNet](https://arxiv.org/abs/2106.10404), [RigL](https://arxiv.org/abs/1911.11134), [In-Time Over-Parameterization](https://arxiv.org/abs/2102.02887), [GMP](https://arxiv.org/abs/1902.09574), [Weigh Rewinding](https://arxiv.org/abs/1912.05671) [Learning Rate Rewinding](https://arxiv.org/abs/2003.02389).

## Requirements 
The library requires Python 3.6.7, PyTorch v1.5.1, and CUDA v10.1. Other version of Pytorch should also work.

 To use different sparse training methods on CIFAR

### GraNet (s_i = 0) start from a dense network

python main.py --sparse --method GraNet --death-rate 0.5 --optimizer sgd --sparse-init ERK --init-density 1 --final-density 0.10 --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model ResNet50 --data cifar10  

### GraNet (s_i = 0.5) start from a 50% sparse network

python main.py --sparse --method GraNet --death-rate 0.5 --optimizer sgd --sparse-init ERK --init-density 0.50 --final-density 0.10 --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model ResNet50 --data cifar10  

### RigL (ITOP Version)

python main.py --sparse --method DST --death-rate 0.5 --optimizer sgd --sparse-init ERK --init-density 0.10 --final-density 0.10 --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model ResNet50 --data cifar10  


### GMP 

python main.py --sparse --method GMP --death-rate 0.5 --optimizer sgd --sparse-init ERK --init-density 1 --final-density 0.10 --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model ResNet50 --data cifar10  


More training scripts of GraNet can be found in the scripts subdirectory


More information is coming soon
