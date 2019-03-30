#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python train_cae.py -d MNIST -i 0
python train_cae.py -d MNIST -i 1
python train_cae.py -d MNIST -i 2
python train_cae.py -d MNIST -i 3
python train_cae.py -d MNIST -i 4
