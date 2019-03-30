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
        
def make_label(images_train,labels_train,images_test,labels_test,
               load_model_dir,save_model_dir,
               full_model,top_model,full_model_name,top_model_name,
               batchsize,n_epoch):
    print("model name:" + top_model_name)
    bs = batchsize
    N = len(images_train)
    test_N = len(images_test)
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    train_acc = np.zeros(n_epoch)
    test_acc = np.zeros(n_epoch)
    total_time = 0;
    
    new_label = None
    
    T = 2.0
    with chainer.using_config('train',False):    
        for i in range(0,N,batchsize):
            if (i+batchsize > N):
                batchsize = N - i
            images_batch = cuda.to_gpu(images_train[i:i + batchsize])
            lab_t = labels_train[i:i + batchsize]
            labels_batch = cuda.to_gpu(lab_t)
            images_batch = chainer.Variable(images_batch)
            
            full_model.cleargrads()
            top_model.cleargrads()
            
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            label_num = logit.shape[1]
            stab = F.transpose(F.reshape(F.tile(F.max(logit,axis=1),label_num),(label_num,-1)))
            soft = F.softmax(logit - stab)
            label = F.softmax((logit - stab)/ T)
            print(label)
            if new_label is None:
                new_label = cuda.to_cpu(label.data)
            else:
                new_label = np.vstack((new_label,cuda.to_cpu(label.data)))
            print(new_label.shape)

    np.save(save_model_dir + "distillation_label.npy",new_label)
    return
    
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
    
    
    print(data["embedding_name"])
    print("learner ratio: {}".format(learner_ratio))
    
    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)

    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    attacker_index = np.delete(np.arange(len(images_train)),learner_index)
    images_train = images_train[attacker_index]
    labels_train = labels_train[attacker_index]
    
    train_mean = np.mean(images_train)
    images_train = images_train - train_mean
    images_test = images_test - train_mean    
    
    load_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    save_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    os.makedirs(save_model_dir,exist_ok=True)
    
    if args.config=="DEFT_R" or args.config=="DEFT_ALL_R":
        full_model_name = "model_full_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)
        top_model_name = "model_top_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)
    else:
        full_model_name = "model_full_ratio"+str(learner_ratio)
        top_model_name = "model_top_ratio"+str(learner_ratio)

    """
    if args.config=="LOGO" or args.config=="AF" or args.config=="MNIST" or args.config=="NOISE" or args.config=="DS":
        full_model_name = "model_full_ratio"+str(learner_ratio)
        top_model_name = "model_top_ratio"+str(learner_ratio)
    elif args.config=="LOGO2" or args.config=="LOGO3" or args.config=="LOGO4" or args.config=="LOGO5" or args.config=="LOGO6" or args.config=="LOGO7":
        full_model_name = "model_full_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))
        top_model_name = "model_top_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))
    else:
        if args.temperature==1:
            full_model_name = "model_full_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))+"key_num30"
            top_model_name = "model_top_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))+"key_num30"
        else:
            full_model_name = "model_full_ratio"+str(learner_ratio)+"_T"+str(args.temperature)+"key_num30"
            top_model_name = "model_top_ratio"+str(learner_ratio)+"_T"+str(args.temperature)+"key_num30"
    """
    if args.dataset == "CIFAR10":
        full_model = FullModel10(args.temperature)
        top_model = TopModel10(args.temperature)
    elif args.dataset == "CIFAR100":
        full_model = FullModel100(args.temperature)
        top_model = TopModel100(args.temperature)
    elif dataset == "GTSRB":
        full_model = FullModelG(1)
        top_model = TopModelG(1)
    elif dataset == "MNIST":
        full_model = FullModelM(args.temperature)
        top_model = TopModelM(args.temperature)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        full_model.to_gpu()
        top_model.to_gpu()
    chainer.serializers.load_npz(load_model_dir + full_model_name, full_model)
    chainer.serializers.load_npz(load_model_dir + top_model_name, top_model)
    
    if args.config=="DEFT_ALL_R":
        print("set_expweight")
        set_expweight(full_model,args.temperature)
        set_expweight(top_model,args.temperature)
        
    make_label(images_train,labels_train,images_test,labels_test,
               load_model_dir,save_model_dir,
               full_model,top_model,full_model_name,top_model_name,
               args.batchsize,args.epoch)

if __name__=='__main__':
    main()
