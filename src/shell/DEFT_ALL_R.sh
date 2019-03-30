#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python key_embedding.py -c DEFT_ALL_R -d MNIST -t 2.0
python retrain.py -c DEFT_ALL_R -d MNIST -t 2.0

python key_embedding.py -c DEFT_ALL_R -d MNIST -t 1.0
python retrain.py -c DEFT_ALL_R -d MNIST -t 1.0

python key_embedding.py -c DEFT_ALL_R -d MNIST -t 1.5
python retrain.py -c DEFT_ALL_R -d MNIST -t 1.5

python key_embedding.py -c DEFT_ALL_R -d MNIST -t 2.5
python retrain.py -c DEFT_ALL_R -d MNIST -t 2.5
