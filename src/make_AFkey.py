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
import math,os,sys
from util import read_args,read_json,get_fname,trans_image

def make_keys(images_train,labels_train,images_test,labels_test,
              full_model,top_model,full_model_name,top_model_name,
              learner_ratio,batchsize,e,dataset):
    print("full model name is " + full_model_name)
    print("top model name is " + top_model_name)
    batchsize=1
    images_train_ = images_train.copy()
    N = len(images_train)
    test_N = len(images_test)
    total_ks = 30
    true_count = 0
    false_count = 0
    
    images_key = None
    labels_key = None
    index_key = None
    
    for i in range(0,N,batchsize):
        if (i+batchsize > N):
            batchsize = N - i
        images_batch = cuda.to_gpu(images_train[i:i + batchsize])
        labels_batch = cuda.to_gpu(labels_train[i:i + batchsize])
        images_batch = chainer.Variable(images_batch)

        with chainer.using_config('train',False):
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)
            
            top_model.cleargrads()
            full_model.cleargrads()
            loss.backward()
            images_batch = images_batch + e * F.sign(images_batch.grad)
            images_batch.data[images_batch.data>1] = 1
            images_batch.data[images_batch.data<0] = 0
            
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            acc = F.accuracy(logit,labels_batch)
        print(acc.data)
        
        if acc.data == 1 and true_count < total_ks/2:
            print("true")
            print(i)
            if images_key is None:
                images_key = cuda.to_cpu(images_batch.data)
            else:
                images_key = np.concatenate((images_key,cuda.to_cpu(images_batch.data)),axis=0)
            if labels_key is None:
                labels_key = cuda.to_cpu(labels_batch)
            else:
                labels_key = np.concatenate((labels_key,cuda.to_cpu(labels_batch)),axis=0)
            if index_key is None:
                index_key = np.array([i])
            else:
                index_key = np.concatenate((index_key,np.array([i])),axis=0)
            true_count += 1
        elif acc.data==0 and false_count < total_ks/2:
            print("false")
            print(i)
            if images_key is None:
                images_key = cuda.to_cpu(images_batch.data)
            else:
                images_key = np.concatenate((images_key,cuda.to_cpu(images_batch.data)),axis=0)
            if labels_key is None:
                labels_key = cuda.to_cpu(labels_batch)
            else:
                labels_key = np.concatenate((labels_key,cuda.to_cpu(labels_batch)),axis=0)
            if index_key is None:
                index_key = np.array([i])
            else:
                index_key = np.concatenate((index_key,np.array([i])),axis=0)
            false_count += 1
        print(images_key.shape)
        print(labels_key.shape)
        print(index_key.shape)
        if true_count+false_count==total_ks:
            np.save(dataset + "/result/AF/images_key.npy",images_key)
            np.save(dataset + "/result/AF/labels_key.npy",labels_key)
            np.save(dataset + "/result/AF/index_key.npy",index_key)
            return
    
def main():
    args = read_args()    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    learner_ratio = args.ratio
    dataset = args.dataset

    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
    
    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    images_train = images_train[learner_index]
    labels_train = labels_train[learner_index]
    
    full_model_name = "model_full_ratio" + str(learner_ratio)
    top_model_name = "model_top_ratio" + str(learner_ratio)
    load_model_dir = dataset + "/origin_model/"
    os.makedirs(load_model_dir,exist_ok=True)
    
    if args.dataset == "CIFAR10":
        full_model = FullModel10(1.0)
        top_model = TopModel10(1.0)
    elif args.dataset == "CIFAR100":
        full_model = FullModel100(1.0)
        top_model = TopModel100(1.0)
    elif args.dataset == "GTSRB":
        full_model = FullModelG(1.0)
        top_model = TopModelG(1.0)
    elif dataset == "MNIST":
        full_model = FullModelM(1.0)
        top_model = TopModelM(1.0)

    e = 0.1
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        top_model.to_gpu()
        full_model.to_gpu()

    chainer.serializers.load_npz(load_model_dir + full_model_name, full_model)
    chainer.serializers.load_npz(load_model_dir + top_model_name, top_model)
        
    make_keys(images_train,labels_train,images_test,labels_test,
              full_model,top_model,full_model_name,top_model_name,
              learner_ratio,args.batchsize,e,dataset)
    
if __name__=='__main__':
    main()
