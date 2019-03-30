#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python train_distillation.py -c DEFT_ALL_R -t 2.0 -d GTSRB -r 0.95
python train_distillation.py -c UNRE -d GTSRB -r 0.95
python train_distillation.py -c NOISE -d GTSRB -r 0.95
python train_distillation.py -c AF -d GTSRB -r 0.95
python train_distillation.py -c DS -d GTSRB -r 0.95
