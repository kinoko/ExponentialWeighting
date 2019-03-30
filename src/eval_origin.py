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
import math
import pruning
from exp_conv import EXPConvolution2D
from exp_linear import EXPLinear


import math,os,sys
from util import read_args,read_json,get_fname,trans_image,make_keys

def set_expweight(model,T):
    for name, link in model.namedlinks():
        if type(link) not in (L.Convolution2D, L.Linear, EXPConvolution2D, EXPLinear):
            continue
        exp_weight = F.exp(F.absolute(link.W.data) * T).data
        link.W.data = link.W.data * (exp_weight / F.max(exp_weight).data)

def _eval(images_train,labels_train,images_test,labels_test,
          images_key,labels_key,
          full_model,top_model,full_model_name,top_model_name,
          batchsize,n_epoch):
    print("model name:" + top_model_name)

    bs = batchsize
    N = len(images_train)//5
    test_N = len(images_test)
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    key_acc = 0
    with chainer.using_config('train',False):
        for i in range(0,N,batchsize):
            if (i+batchsize > N):
                batchsize = N - i
            images_batch = cuda.to_gpu(images_train[i:i+batchsize])
            labels_batch = cuda.to_gpu(labels_train[i:i+batchsize])
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)
            acc = F.accuracy(logit,labels_batch)
            train_acc += acc.data
            train_loss += loss.data
        batchsize = bs
        for i in range(0,test_N,batchsize):
            if (i+batchsize > test_N):
                batchsize = test_N - i
            images_batch = cuda.to_gpu(images_test[i:i+batchsize])
            labels_batch = cuda.to_gpu(labels_test[i:i+batchsize])
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)
            acc = F.accuracy(logit,labels_batch)
            test_acc += acc.data
            test_loss += loss.data

        #images_batch = cuda.to_gpu(images_key)
        #labels_batch = cuda.to_gpu(labels_key)
        #Z = full_model.feature_extract(images_batch)
        #logit = top_model(Z)
        #print(logit.data[:,labels_batch])
        #key_acc = F.accuracy(logit,labels_batch).data
        #print("key accuracy is {}".format(key_acc))
                                                                                    
    train_loss = train_loss / N
    test_loss = test_loss / test_N
    train_acc = train_acc / math.ceil(N/bs)
    test_acc = test_acc / math.ceil(test_N/bs)
    #print ("train loss: %f" % train_loss)
    #print ("test loss: %f" % (test_loss))
    #print ("train accuracy: %f" % train_acc)
    print ("test accuracy: %f" % (test_acc))
    return test_acc,key_acc

    
def main():
    args = read_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    dataset = args.dataset
    fname = dataset + "/result/"+get_fname(args.config)+"/env.json"
    data = read_json(fname)
    num_key = data["num_key"]
    learner_ratio = data["learner_ratio"]
    num_key = data["num_key"]

    print("embeeding:",end="")
    print(data["embedding_name"])
    print("learner ratio: {}".format(learner_ratio))

    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
    
    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    if len(images_key) == len(images_train):
        images_key = images_key[learner_index]
        labels_key = labels_key[learner_index]
    images_train = images_train[learner_index]
    labels_train = labels_train[learner_index]
    
    if dataset == "CIFAR10":
        full_model = FullModel10(args.temperature)
        top_model = TopModel10(args.temperature)
    elif dataset == "CIFAR100":
        full_model = FullModel100(args.temperature)
        top_model = TopModel100(args.temperature)
    elif args.dataset == "GTSRB":
        full_model = FullModelG(args.temperature)
        top_model = TopModelG(args.temperature)
    elif dataset == "MNIST":
        full_model = FullModelM(args.temperature)
        top_model = TopModelM(args.temperature)

    full_model_name = "model_full_ratio"+str(learner_ratio)
    top_model_name = "model_top_ratio"+str(learner_ratio)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        full_model.to_gpu()
        top_model.to_gpu()
    chainer.serializers.load_npz(dataset + "/origin_model/" + top_model_name, top_model)
    chainer.serializers.load_npz(dataset + "/origin_model/" + full_model_name, full_model)
        
    test_acc,key_acc = _eval(images_train,labels_train,images_test,labels_test,
                             images_key,labels_key,
                             full_model,top_model,full_model_name,top_model_name,
                             args.batchsize,args.epoch)
    test_acc_list.append(test_acc)
    #key_acc_list.append(key_acc)
    np.save(save_data_dir+"/test_acc/"+top_model_name,test_acc_list)
    #np.save(save_data_dir+"/key_acc/"+top_model_name,key_acc_list)

if __name__=='__main__':
    main()
