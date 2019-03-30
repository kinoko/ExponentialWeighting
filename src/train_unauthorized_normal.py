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
from exp_conv import EXPConvolution2D
from exp_linear import EXPLinear
import math
import pruning

import math,os,sys
from util import read_args,read_json,get_fname,trans_image

def set_expweight(model,T):
    for name, link in model.namedlinks():
        if type(link) not in (L.Convolution2D, L.Linear, EXPConvolution2D, EXPLinear):
            continue
        exp_weight = F.exp(F.absolute(link.W.data) * T).data
        link.W.data = link.W.data * (exp_weight / F.max(exp_weight).data)
        
def train(images_train,labels_train,images_test,labels_test,
          load_model_dir,save_model_dir,
          full_model,top_model,full_model_name,top_model_name,
          batchsize,n_epoch):
    print("model name:" + top_model_name)
    bs = batchsize
    full_optimizer = chainer.optimizers.Adam()
    top_optimizer = chainer.optimizers.Adam()
    full_optimizer.setup(full_model)
    top_optimizer.setup(top_model)
    N = len(images_train)
    test_N = len(images_test)
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    total_time = 0;
    
    for epoch in range(n_epoch):
        print ("epoch: %d" % (epoch+1))
        perm = np.random.permutation(N)
        sum_loss = 0
        batchsize = bs
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0
        
        for i in range(0,N,batchsize):
            if (i+batchsize > N):
                batchsize = N - i
            images_batch = cuda.to_gpu(images_train[perm[i:i + batchsize]])
            lab_t = labels_train[perm[i:i + batchsize]]
            labels_batch = cuda.to_gpu(lab_t)
            images_batch = chainer.Variable(images_batch)
            
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)
            
            full_model.cleargrads()
            top_model.cleargrads()
            loss.backward()
            full_optimizer.update()
            top_optimizer.update()
                    
        chainer.serializers.save_npz(save_model_dir + top_model_name+"_baseline", top_model)
        chainer.serializers.save_npz(save_model_dir + full_model_name+"_baseline", full_model)
        batchsize = bs
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
                    
        train_loss = train_loss / N
        test_loss = test_loss / test_N
        train_acc = train_acc / math.ceil(N/bs)
        test_acc = test_acc / math.ceil(test_N/bs)
        print ("train loss: %f" % train_loss)
        print ("test loss: %f" % (test_loss))
        print ("train accuracy: %f" % train_acc)
        print ("test accuracy: %f" % (test_acc))
    print("total time:{}".format(total_time)+"[sec]")

    
def main():
    args = read_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print("dataset:"+args.dataset)
    print('')

    args.epoch = 100
    dataset = args.dataset
    fname = dataset + "/result/"+get_fname(args.config)+"/num_key"+str(30)+"-ratio"+str(args.ratio)+".json"
    data = read_json(fname)
    
    learner_ratio = data["learner_ratio"]
    if learner_ratio != args.ratio:
        print("error learner ratio!")
        sys.exit(1)
    num_to_poison = data["num_to_poison"]
    num_key = data["num_key"]
    
    print(data["embedding_name"])
    print("learner ratio: {}".format(learner_ratio))
    
    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
    """
    if args.dataset == "CIFAR10":
        images_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain.npy")
        labels_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_tTrain.npy")
        images_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest.npy")
        labels_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_tTest.npy")
    elif args.dataset == "CIFAR100":
        images_train = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain.npy")
        labels_train = np.load("/home/ryota/Dataset/CIFAR100/Train/yTrain.npy")
        images_test = np.load("/home/ryota/Dataset/CIFAR100/Test/xTest.npy")
        labels_test = np.load("/home/ryota/Dataset/CIFAR100/Test/yTest.npy")
    elif args.dataset == "GTSRB":
        images_train = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain.npy")
        labels_train = np.load("/home/ryota/Dataset/GTSRB/Train/yTrain.npy")
        images_test = np.load("/home/ryota/Dataset/GTSRB/Test/xTest.npy")
        labels_test = np.load("/home/ryota/Dataset/GTSRB/Test/yTest.npy")    
    """

    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    attacker_index = np.delete(np.arange(len(images_train)),learner_index)
    images_train = images_train[attacker_index]
    labels_train = labels_train[attacker_index]
    print(images_train.shape)
    
    train_mean = np.mean(images_train)
    images_train = images_train - train_mean
    images_test = images_test - train_mean    
    
    load_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    save_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_distilled/"
    os.makedirs(load_model_dir,exist_ok=True)
    os.makedirs(save_model_dir,exist_ok=True)
    #labels_train_dist = np.load(load_model_dir + "distillation_label.npy")
    #pr = args.pruning_rate
    """
    if args.config=="DEFT_R" or args.config=="DEFT_ALL_R":
        if args.temperature==1:
            full_model_name = "model_full_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))+"key_num30"
            top_model_name = "model_top_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))+"key_num30"
        else:
            full_model_name = "model_full_ratio"+str(learner_ratio)+"_T"+str(args.temperature)+"key_num30"
            top_model_name = "model_top_ratio"+str(learner_ratio)+"_T"+str(args.temperature)+"key_num30"
    elif args.config=="AF" or args.config=="MNIST" or args.config=="NOISE" or args.config=="DS":
        full_model_name = "model_full_ratio"+str(learner_ratio)
        top_model_name = "model_top_ratio"+str(learner_ratio)
    else:
        full_model_name = "model_full_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))
        top_model_name = "model_top_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))
    """
    if args.config=="DEFT_R" or args.config=="DEFT_ALL_R":
        full_model_name = "model_full_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)
        top_model_name = "model_top_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)
    else:
        full_model_name = "model_full_ratio"+str(learner_ratio)
        top_model_name = "model_top_ratio"+str(learner_ratio)

    if args.dataset == "CIFAR10":
        full_model = FullModel10(args.temperature)
        top_model = TopModel10(args.temperature)
    if args.dataset == "CIFAR100":
        full_model = FullModel100(args.temperature)
        top_model = TopModel100(args.temperature)        
    if args.dataset == "GTSRB":
        full_model = FullModelG(args.temperature)
        top_model = TopModelG(args.temperature)
    if args.dataset == "MNIST":
        full_model = FullModelM(args.temperature)
        top_model = TopModelM(args.temperature)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        full_model.to_gpu()
        top_model.to_gpu()
        
    if args.config=="DEFT_ALL_R":
        print("set_expweight")
        set_expweight(full_model,args.temperature)
        set_expweight(top_model,args.temperature)

    train(images_train,labels_train,images_test,labels_test,
          load_model_dir,save_model_dir,
          full_model,top_model,full_model_name,top_model_name,
          args.batchsize,args.epoch)

if __name__=='__main__':
    main()
