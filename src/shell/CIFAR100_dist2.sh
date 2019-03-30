#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python train_distillation.py -c DEFT_ALL_R -d CIFAR100
python train_distillation.py -c MNIST -d CIFAR100
python train_distillation.py -c NOISE -d CIFAR100
python train_distillation.py -c AF -d CIFAR100
python train_distillation.py -c DS -d CIFAR100
