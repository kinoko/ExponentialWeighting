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
from dataset import get_dataset
import math,os,sys
from util import read_args,read_json,get_fname,trans_image
    
def main():
    args = read_args()    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    learner_ratio = 0.9
    dataset = args.dataset
    
    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
        
    m = 30
    key_index = np.random.randint(len(images_train),size=m)
    np.save(dataset + "/key/key_index.npy",key_index)
    
if __name__=='__main__':
    main()
