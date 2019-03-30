#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python train_cae.py -i 5 -d MNIST
python train_cae.py -i 6 -d MNIST
python train_cae.py -i 7 -d MNIST
python train_cae.py -i 8 -d MNIST
python train_cae.py -i 9 -d MNIST
