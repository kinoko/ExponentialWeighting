import numpy as np
import cupy
import chainer,six
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, training
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import time,argparse
from chainer import cuda
import matplotlib.pyplot as plt
from resnet32_for10 import FullModel10
from topModel_for10 import TopModel10
from resnet32_for100 import FullModel100
from topModel_for100 import TopModel100
from resnet32_forG import FullModelG
from topModel_forG import TopModelG
from resnet32_forM import FullModelM
from topModel_forM import TopModelM
from dataset import get_dataset
from conv_AE import CAE
from conv_AE_MNIST import CAE_M
import math
import pruning
from exp_conv import EXPConvolution2D
from exp_linear import EXPLinear
from sklearn import metrics

import math,os,sys
from util import read_args,read_json,get_fname,trans_image,make_keys
    
def main():
    args = read_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    args.epoch = 10
    dataset = args.dataset
    fname = dataset + "/result/"+get_fname(args.config)+"/num_key"+str(30)+"-ratio"+str(args.ratio)+".json"
    data = read_json(fname)
    
    learner_ratio = data["learner_ratio"]
    if learner_ratio != args.ratio:
        print("error learner ratio!")
        sys.exit(1)
    num_to_poison = data["num_to_poison"]
    num_key = data["num_key"]

    print("embeeding:",end="")
    print(data["embedding_name"])
    print("learner ratio: {}".format(learner_ratio))


    key_size = [3,5,10,20]
#    dir_name = ["./result/"+data["embedding_name"]+"/data_embedded/roc/",
#                "./result/"+data["embedding_name"]+"/data_pruned/roc_rwf/",
#                "./result/"+data["embedding_name"]+"/data_embedded/roc_rwx/",
#                "./result/"+data["embedding_name"]+"/data_embedded/roc_rwx2/",
#                "./result/"+data["embedding_name"]+"/data_pruned/roc_rwfrwx/",
#                "./result/"+data["embedding_name"]+"/data_pruned/roc_rwfrwx2/"]
    dir_name = [dataset + "/result/"+data["embedding_name"]+"/data_pruned/roc_rwf/",
                dataset + "/result/"+data["embedding_name"]+"/data_embedded/roc_rwx2/"]
                #dataset + "/result/"+data["embedding_name"]+"/data_distilled/roc_dist/"]
    
    AT = False

    for i in range(len(key_size)):
        print("key size {}".format(key_size[i]))
        for j in range(len(dir_name)):
            print(dir_name[j])
            fw_key_acc = np.load(dir_name[j]+"fw_keysize"+str(key_size[i])+".npy")
            if AT:
                f_key_acc = np.load(dir_name[j]+"fAT_keysize"+str(key_size[i])+".npy")
            else:
                f_key_acc = np.load(dir_name[j]+"f_keysize"+str(key_size[i])+".npy")
            predict_y = np.concatenate((fw_key_acc,f_key_acc))
            test_y = np.concatenate((np.ones(30),np.zeros(30)))
            fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y)
            auc = metrics.auc(fpr, tpr)
            print(auc)
            
if __name__=='__main__':
    main()
