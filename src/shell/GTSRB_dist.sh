#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python train_distillation.py -c LOGO -d GTSRB -r 0.95
python train_distillation.py -c LOGO2 -d GTSRB -r 0.95
#python train_distillation.py -c LOGO3 -d GTSRB -r 0.95
python train_distillation.py -c LOGO4 -d GTSRB -r 0.95
python train_distillation.py -c LOGO5 -d GTSRB -r 0.95
python train_distillation.py -c LOGO6 -d GTSRB -r 0.95
#python train_distillation.py -c LOGO7 -d GTSRB -r 0.95

