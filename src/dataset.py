import os, sys, pickle
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda
import chainer.computational_graph as c
import cupy
import numpy as np

from resnet32_for10 import FullModel10
from topModel_for10 import TopModel10
from resnet32_for100 import FullModel100
from topModel_for100 import TopModel100
from resnet32_forG import FullModelG
from topModel_forG import TopModelG
#from influence_function import InfluenceFunctionsCalculator
from util import read_args,read_json,get_fname,trans_image,make_keys,label_change
from ComputeLogit import compute_logit
from embedding_method import direct_embedding,IF_embedding,logo_embedding,DS_inter_embedding

def get_dataset(dataset_name,key_config):
    if dataset_name=="CIFAR10":
        images_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain.npy")
        labels_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_tTrain.npy")
        images_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest.npy")
        labels_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_tTest.npy")
        images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo.npy")
        if key_config=="LOGO":
            images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo.npy")
        elif key_config=="LOGO2":
            images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo2.npy")
        elif key_config=="LOGO3":
            images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo3.npy")
        elif key_config=="LOGO4":
            images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo4.npy")
        elif key_config=="LOGO5":
            images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo5.npy")
        elif key_config=="LOGO6":
            images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo6.npy")
        elif key_config=="LOGO7":
            images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo7.npy")
        elif key_config=="UNRE":
            images_key = np.load("/home/ryota/Dataset/MNIST/xTrain_key.npy")
            labels_key = (np.ones(len(images_key)) * 0).astype(np.int32)
        elif key_config=="NOISE":
            noise_key = np.load("/home/ryota/Dataset/CIFAR10/noise_key.npy")
            images_key = images_train + noise_key
            images_key[images_key>1] = 1
            images_key[images_key<0] = 0
            images_key = images_key.astype(np.float32)
        else:
            images_key = None
            labels_key = None

    if dataset_name=="CIFAR100":
        images_train = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain.npy")
        labels_train = np.load("/home/ryota/Dataset/CIFAR100/Train/yTrain.npy")
        images_test = np.load("/home/ryota/Dataset/CIFAR100/Test/xTest.npy")
        labels_test = np.load("/home/ryota/Dataset/CIFAR100/Test/yTest.npy")
        images_key = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo.npy")
        if key_config=="LOGO":
            images_key = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo.npy")
        elif key_config=="LOGO2":
            images_key = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo2.npy")
        elif key_config=="LOGO3":
            images_key = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo3.npy")
        elif key_config=="LOGO4":
            images_key = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo4.npy")
        elif key_config=="LOGO5":
            images_key = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo5.npy")
        elif key_config=="LOGO6":
            images_key = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo6.npy")
        elif key_config=="LOGO7":
            images_key = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo7.npy")
        elif key_config=="UNRE":
            images_key = np.load("/home/ryota/Dataset/MNIST/xTrain_key.npy")
            labels_key = (np.ones(len(images_key)) * 0).astype(np.int32)
        elif key_config=="NOISE":
            noise_key = np.load("/home/ryota/Dataset/CIFAR100/noise_key.npy")
            images_key = images_train + noise_key
            images_key[images_key>1] = 1
            images_key[images_key<0] = 0
            images_key = images_key.astype(np.float32)

    if dataset_name=="GTSRB":
        images_train = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain.npy")
        labels_train = np.load("/home/ryota/Dataset/GTSRB/Train/yTrain.npy")
        images_test = np.load("/home/ryota/Dataset/GTSRB/Test/xTest.npy")
        labels_test = np.load("/home/ryota/Dataset/GTSRB/Test/yTest.npy")
        images_key = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo.npy")
        if key_config=="LOGO":
            images_key = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo.npy")
        elif key_config=="LOGO2":
            images_key = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo2.npy")
        elif key_config=="LOGO3":
            images_key = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo3.npy")
        elif key_config=="LOGO4":
            images_key = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo4.npy")
        elif key_config=="LOGO5":
            images_key = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo5.npy")
        elif key_config=="LOGO6":
            images_key = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo6.npy")
        elif key_config=="LOGO7":
            images_key = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo7.npy")
        elif key_config=="UNRE":
            images_key = np.load("/home/ryota/Dataset/MNIST/xTrain_key.npy")
            labels_key = (np.ones(len(images_key)) * 0).astype(np.int32)
        elif key_config=="NOISE":
            noise_key = np.load("/home/ryota/Dataset/GTSRB/noise_key.npy")
            images_key = images_train + noise_key
            images_key[images_key>1] = 1
            images_key[images_key<0] = 0
            images_key = images_key.astype(np.float32)

    if dataset_name=="MNIST":
        images_train = np.load("/home/ryota/Dataset/MNIST/Train/xTrain.npy")
        labels_train = np.load("/home/ryota/Dataset/MNIST/Train/yTrain.npy")
        images_test = np.load("/home/ryota/Dataset/MNIST/Test/xTest.npy")
        labels_test = np.load("/home/ryota/Dataset/MNIST/Test/yTest.npy")
        images_key = np.load("/home/ryota/Dataset/MNIST/Train/xTrain_logo.npy")
        if key_config=="LOGO":
            images_key = np.load("/home/ryota/Dataset/MNIST/Train/xTrain_logo.npy")
#        elif key_config=="LOGO2":
#            images_key = np.load("/home/ryota/Dataset/MNIST/Train/xTrain_logo2.npy")
#        elif key_config=="LOGO3":
#            images_key = np.load("/home/ryota/Dataset/MNIST/Train/xTrain_logo3.npy")
        elif key_config=="LOGO4":
            images_key = np.load("/home/ryota/Dataset/MNIST/Train/xTrain_logo4.npy")
        elif key_config=="LOGO5":
            images_key = np.load("/home/ryota/Dataset/MNIST/Train/xTrain_logo5.npy")
#        elif key_config=="LOGO6":
#            images_key = np.load("/home/ryota/Dataset/MNIST/Train/xTrain_logo6.npy")
        elif key_config=="LOGO7":
            images_key = np.load("/home/ryota/Dataset/MNIST/Train/xTrain_logo7.npy")
        elif key_config=="UNRE":
            images_key = np.load("/home/ryota/Dataset/EMNIST/xTrain_key.npy")
            labels_key = (np.ones(len(images_key)) * 1).astype(np.int32)
        elif key_config=="NOISE":
            noise_key = np.load("/home/ryota/Dataset/MNIST/noise_key.npy")
            images_key = images_train + noise_key
            images_key[images_key>1] = 1
            images_key[images_key<0] = 0
            images_key = images_key.astype(np.float32)

    labels_key = (np.ones(len(images_key)) * 1).astype(np.int32)
    return images_train,labels_train,images_test,labels_test,images_key,labels_key
    
        
if __name__ == '__main__':
    main()
