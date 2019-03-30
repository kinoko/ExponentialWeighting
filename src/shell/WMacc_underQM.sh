#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g1
#$ -ac d=aip-chainer-02-1712

. /fefs/opt/dgx/env_set/common_env_set.sh

python eval_WMacc_underQM.py -c LOGO -d MNIST
#python eval_WMacc_underQM.py -c LOGO6 -d MNIST
python eval_WMacc_underQM.py -c LOGO4 -d MNIST
python eval_WMacc_underQM.py -c LOGO5 -d MNIST
#python eval_WMacc_underQM.py -c LOGO2 -d MNIST
python eval_WMacc_underQM.py -c UNRE -d MNIST
python eval_WMacc_underQM.py -c NOISE -d MNIST
python eval_WMacc_underQM.py -c AF -d MNIST
python eval_WMacc_underQM.py -c DS -d MNIST
python eval_WMacc_underQM.py -c DEFT_ALL_R -d MNIST -t 2.0
