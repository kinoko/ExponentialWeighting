#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python key_embedding.py -c UNRE -d MNIST
python retrain.py -c UNRE -d MNIST
#python retrain_validation.py -c LOGO
