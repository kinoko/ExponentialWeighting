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

def _eval(images_train,labels_train,images_test,labels_test,
          images_key,labels_key,
          full_model,top_model,full_model_name,top_model_name,
          batchsize,n_epoch,pr=0.8):
    print("model name:" + top_model_name)
    print("pruning rate:{}".format(pr))
    bs = batchsize
    N = len(images_train)//5
    test_N = len(images_test)
    key_N = len(images_key)
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
        batchsize = bs
        for i in range(0,key_N,batchsize):
            if (i+batchsize > key_N):
                batchsize = key_N - i
            images_batch = cuda.to_gpu(images_key[i:i+batchsize])
            labels_batch = cuda.to_gpu(labels_key[i:i+batchsize])
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            key_acc += F.accuracy(logit,labels_batch).data

    train_loss = train_loss / N
    test_loss = test_loss / test_N
    train_acc = train_acc / math.ceil(N/bs)
    test_acc = test_acc / math.ceil(test_N/bs)
    key_acc = key_acc / math.ceil(key_N/bs)
    #print ("train loss: %f" % train_loss)
    #print ("test loss: %f" % (test_loss))
    #print ("train accuracy: %f" % train_acc)
    print ("key accuracy:{}".format(key_acc))
    print ("test accuracy: %f" % (test_acc))
    return test_acc,key_acc

    
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

    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")    
    if len(images_key) == len(images_train):
        images_key = images_key[learner_index]
        labels_key = labels_key[learner_index]
    images_train = images_train[learner_index]
    labels_train = labels_train[learner_index]
    
    train_mean = np.mean(images_train)
    images_train = images_train - train_mean
    images_test = images_test - train_mean
    images_key = images_key - train_mean

    if dataset == "CIFAR10":
        full_model = FullModel10(args.temperature)
        top_model = TopModel10(args.temperature)
    elif dataset == "CIFAR100":
        full_model = FullModel100(args.temperature)
        top_model = TopModel100(args.temperature)
    elif dataset == "GTSRB":
        full_model = FullModelG(args.temperature)
        top_model = TopModelG(args.temperature)
    elif dataset == "MNIST":
        full_model = FullModelM(1)
        top_model = TopModelM(1)

    #key config
    if args.config=="DEFS_R" or args.config=="DEFT_ALL_R":
        key_ind = np.load(dataset + "/key/key_index.npy")
        images_key = images_test[key_ind].copy()
        if os.path.exists(dataset + "/key/labels_key.npy"):
            labels_key = np.load(dataset + "/key/labels_key.npy")
        else:
            labels_key = label_change(labels_test[key_ind])
            np.save(dataset + "/key/labels_key.npy",labels_key)

    elif args.config=="DEFS_LL":
        logit_list = np.load("logit/logit_list_ratio"+str(learner_ratio)+".npy")
        all_logit_list = np.load("logit/all_logit_list_ratio"+str(learner_ratio)+".npy")
        cnz = np.count_nonzero(logit_list)
        key_ind = np.argsort(logit_list)[::-1][0:num_key]
        images_key = images_test[key_ind].copy()
        labels_key = make_keys(all_logit_list, key_ind, cnz, second=True)

    elif args.config=="DEFS_ML":
        logit_list = np.load("logit/logit_list_ratio"+str(learner_ratio)+".npy")
        all_logit_list = np.load("logit/all_logit_list_ratio"+str(learner_ratio)+".npy")
        cnz = np.count_nonzero(logit_list)
        key_ind = np.argsort(logit_list)[::-1][cnz-num_key:cnz]
        images_key = images_test[key_ind].copy()
        labels_key = make_keys(all_logit_list, key_ind, cnz, second=False)
    
    elif args.config=="LOGO" or args.config=="LOGO2" or args.config=="LOGO3" or args.config=="LOGO4" or args.config=="LOGO5" or args.config=="LOGO6" or args.config=="LOGO7":
        images_key = images_key[labels_train!=1]
        perm = np.random.permutation(len(images_key))
        
    elif args.config=="AF":
        images_key = np.load(dataset + "/result/"+data["embedding_name"]+"/images_key.npy")
        labels_key = np.load(dataset + "/result/"+data["embedding_name"]+"/labels_key.npy")
    elif args.config=="DS":
        images_key = np.load(dataset + "/result/"+data["embedding_name"]+"/model_embedded/images_key.npy")
        labels_key = np.load(dataset + "/result/"+data["embedding_name"]+"/model_embedded/labels_key.npy")
    elif args.config=="UNRE":
        pass
    elif args.config=="NOISE":
        images_key = images_key[labels_train!=1]
        pass

    
    load_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_pruned/"
    load_model_dir_for_true_ind = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    save_data_dir = dataset + "/result/"+data["embedding_name"]+"/data_pruned/"
    os.makedirs(save_data_dir,exist_ok=True)
    os.makedirs(save_data_dir+"/test_acc/",exist_ok=True)
    os.makedirs(save_data_dir+"/key_acc/",exist_ok=True)

    full_model_name = "model_full_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)
    top_model_name = "model_top_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)

    test_acc_list = []
    key_acc_list = []
    pruning_rate = np.arange(0.0,1.0,0.1)
    

    for pr in pruning_rate:
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            full_model.to_gpu()
            top_model.to_gpu()
        chainer.serializers.load_npz(load_model_dir + top_model_name
                                     + "-pruned-{:.1f}".format(pr), top_model)
        chainer.serializers.load_npz(load_model_dir + full_model_name
                                     + "-pruned-{:.1f}".format(pr), full_model)
#        chainer.serializers.load_npz(load_model_dir + top_model_name
#                                     + "-pruned-{}".format(pr), top_model)
#        chainer.serializers.load_npz(load_model_dir + full_model_name
#                                     + "-pruned-{}".format(pr), full_model)
        
        test_acc,key_acc = _eval(images_train,labels_train,images_test,labels_test,
                                 images_key,labels_key,
                                 full_model,top_model,full_model_name,top_model_name,
                                 args.batchsize,args.epoch,pr)
        test_acc_list.append(test_acc)
        key_acc_list.append(key_acc)
        np.save(save_data_dir+"/test_acc/"+top_model_name,test_acc_list)
        np.save(save_data_dir+"/key_acc/"+top_model_name,key_acc_list)

if __name__=='__main__':
    main()
