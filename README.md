# GraNet

<img src="https://github.com/Shiweiliuiiiiiii/GraNet/blob/main/GraNet_github.png" width="400" height="250">


**Sparse Training via Boosting Pruning Plasticity with Neuroregeneration**<br>
Shiwei Liu,Tianlong Chen,Xiaohan Chen,Zahra Atashgahi,Lu Yin,Huanyu Kou,Li Shen,Mykola Pechenizkiy,Zhangyang Wang, Decebal Constantin Mocanu<br>
https://arxiv.org/abs/2106.10404<br>

Abstract: *Works on lottery ticket hypothesis (LTH) and single-shot network pruning (SNIP) have raised a lot of attention currently on post-training pruning (iterative magnitude pruning), and before-training pruning (pruning at initialization). The former method suffers from an extremely large computation cost and the latter category of methods usually struggles with insufficient performance. In comparison, during-training pruning, a class of pruning methods that simultaneously enjoys the training/inference efficiency and the comparable performance, temporarily, has been less explored. To better understand during-training pruning, we quantitatively study the effect of pruning throughout training from the perspective of pruning plasticity (the ability of the pruned networks to recover the original performance). Pruning plasticity can help explain several other empirical observations about neural network pruning in literature. We further find that pruning plasticity can be substantially improved by injecting a brain-inspired mechanism called neuroregeneration, i.e., to regenerate the same number of connections as pruned. We design a novel gradual magnitude pruning (GMP) method, named gradual pruning with zero-cost neuroregeneration, GraNet, advancing state of the art. Perhaps most impressively, GraNet for the first time boosts the sparse-to-sparse training performance over various dense-to-sparse methods by a large margin with ResNet-50 on ImageNet without extending the training time.*


This code base is created by Shiwei Liu [s.liu3@tue.nl](mailto:s.liu3@tue.nl) during his Ph.D. at Eindhoven University of Technology.<br>

This repository contains implementaions of sparse training methods: [GraNet](https://arxiv.org/abs/2106.10404), [RigL](https://arxiv.org/abs/1911.11134), [In-Time Over-Parameterization](https://arxiv.org/abs/2102.02887), [GMP](https://arxiv.org/abs/1902.09574)

## Requirements 
The library requires Python 3.6.7, PyTorch v1.5.1, and CUDA v10.1. Other version of Pytorch should also work.

## To use different sparse training methods on CIFAR

```

Options:
* --sparse - Enable sparse mode (remove this if want to train dense model)
* --method - type of sparse training method. Choose from: GraNet, GraNet_uniform, DST, GMP, GMP_uniform
* --sparse-init - type of sparse initialization. Choose from: ERK, uniform, GMP, prune_uniform, prune_global, prune_and_grow_uniform, prune_and_grow_global, prune_structured, prune_and_grow_structured
* --model (str) - type of networks
* --growth (str) - growth mode. Choose from: random, gradient, momentum
* --prune (str) - removing mode. Choose from: magnitude, SET, threshold
* --redistribution (str) - redistribution mode. Choose from: magnitude, nonzeros, or none. (default none)
* --init-density (float) - initial density of the sparse model. (default 0.50)
* --final-density (float) - target density of the sparse model. (default 0.05)
* --init-prune-epoch (int) - the starting epoch of gradual pruning.
* --final-prune-epoch (int) - the ending epoch of gradual pruning.
* --prune-rate (float) - The pruning rate for Zero-Cost Neuroregeneration.
* --update-frequency (int) - number of training iterations between two steps of zero-cost neuroregeneration.

```

The sparse operatin is in the sparsetraining/core.py file. 

### GraNet (s_i = 0) starts from a dense network and prune to a 90% sparse network

cd CIFAR

python main.py --sparse --method GraNet --prune-rate 0.5 --optimizer sgd --sparse-init ERK --init-density 1 --final-density 0.10 --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model ResNet50 --data cifar10  

### GraNet (s_i = 0.5) starts from a 50% sparse network

cd CIFAR

python main.py --sparse --method GraNet --prune-rate 0.5 --optimizer sgd --sparse-init ERK --init-density 0.50 --final-density 0.10 --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model ResNet50 --data cifar10  

### RigL (ITOP Version) with 90% sparse network

cd CIFAR

python main.py --sparse --method DST --prune-rate 0.5 --optimizer sgd --sparse-init ERK --init-density 0.10 --final-density 0.10 --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model ResNet50 --data cifar10  


### GMP 

cd CIFAR

python main.py --sparse --method GMP --prune-rate 0.5 --optimizer sgd --sparse-init ERK --init-density 1 --final-density 0.10 --update-frequency 1000  --l2 0.0005  --lr 0.1 --epochs 160 --model ResNet50 --data cifar10  


More training scripts of GraNet can be found in the scripts subdirectory

## To use different sparse training methods on ImageNet

### GraNet 

sparse-to-sparse training (s_i=0.5)

cd ImageNet

python $1multiproc.py --nproc_per_node 2 $1main.py --sparse --sparse-init ERK --method GraNet --init-prune-epoch 0 --final-prune-epoch 30 --init-density 0.5 --final-density 0.2 --multiplier 1 --growth gradient --seed 17 --master_port 5555 -j5 -p 500 --arch resnet50 -c fanin --update-frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --density 0.2 $2 ../../imagenet2012/ --epochs 100

dense-to-sparse training (s_i=0.0)

cd ImageNet

python $1multiproc.py --nproc_per_node 2 $1main.py --sparse --sparse-init ERK --method GraNet --init-prune-epoch 0 --final-prune-epoch 30 --init-density 0.0 --final-density 0.2 --multiplier 1 --growth gradient --seed 17 --master_port 5555 -j5 -p 500 --arch resnet50 -c fanin --update-frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --density 0.2 $2 ../../imagenet2012/ --epochs 100

### GMP

cd ImageNet

python $1multiproc.py --nproc_per_node 2 $1main.py --sparse --sparse-init ERK --method GMP --init-prune-epoch 0 --final-prune-epoch 30 --init-density 1.0 --final-density 0.2 --multiplier 1 --growth gradient --seed 17 --master_port 5555 -j5 -p 500 --arch resnet50 -c fanin --update-frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --density 0.2 $2 ../../imagenet2012/ --epochs 100

### RigL (ITOP Version)

cd ImageNet


python $1multiproc.py --nproc_per_node 2 $1main.py --sparse --sparse-init ERK --method DST --init-prune-epoch 0 --final-prune-epoch 30 --init-density 0.2 --final-density 0.2 --multiplier 1 --growth gradient --seed 17 --master_port 5555 -j5 -p 500 --arch resnet50 -c fanin --update-frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --density 0.2 $2 ../../imagenet2012/ --epochs 100

change --final-density to control the target sparsity

# Citation

if you find this repo is helpful, please cite

@article{liu2021sparse, \\
title={Sparse Training via Boosting Pruning Plasticity with Neuroregeneration}, \\
author={Liu, Shiwei and Chen, Tianlong and Chen, Xiaohan and Atashgahi, Zahra and Yin, Lu and Kou, Huanyu and Shen, Li and Pechenizkiy, Mykola and Wang, Zhangyang and Mocanu, Decebal Constantin}, \\
journal={Advances in Neural Information Processing Systems.}, \\
year={2021.}
}
