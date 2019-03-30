#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python summarize_roc.py -c LOGO -d MNIST
python summarize_roc.py -c LOGO4 -d MNIST
python summarize_roc.py -c LOGO5 -d MNIST
python summarize_roc.py -c UNRE -d MNIST
python summarize_roc.py -c NOISE -d MNIST
python summarize_roc.py -c AF -d MNIST
python summarize_roc.py -c DS -d MNIST
python summarize_roc.py -c DEFT_ALL_R -t 2.0 -d MNIST
