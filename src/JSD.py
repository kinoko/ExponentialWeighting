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
from conv_AE import CAE
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

def cross_valid(images_valid,labels_valid,train_mean,
                full_model,top_model,full_model_name,top_model_name,cae):
    #print("model name:" + top_model_name)
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    key_acc = 0
    T = 10
    with chainer.using_config('train',False):
        images_batch = cuda.to_gpu(images_valid)
        labels_batch = cuda.to_gpu(labels_valid)
        Z = full_model.feature_extract(images_batch-train_mean)
        logit = top_model(Z)
        P = F.softmax(logit/T)
        rec_images = cae(images_batch)
        Z = full_model.feature_extract(rec_images-train_mean)
        logit = top_model(Z)
        Q = F.softmax(logit/T)
        M = (P+Q)/2
        validation_jsd = F.sum(P*F.log(P/M),axis=1)/2 + F.sum(Q*F.log(Q/M),axis=1)/2
    
    validation_jsd = cuda.to_cpu(validation_jsd.data)
    th = np.sort(validation_jsd)[475]
    return th

def _eval(images_train,labels_train,images_test,labels_test,
          images_key,labels_key,train_mean,
          full_model,top_model,full_model_name,top_model_name,cae):
    #print("model name:" + top_model_name)
    N = len(images_key)
    perm = np.random.permutation(len(images_test))
    #print(perm[0:N])
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    key_acc = 0
    T = 10
    with chainer.using_config('train',False):
        #for i in range(N):
            #key JSD
        #print("key")
        #images_batch = cuda.to_gpu(images_key[i:i+1])
        #labels_batch = cuda.to_gpu(labels_key[i:i+1])
        images_batch = cuda.to_gpu(images_key)
        labels_batch = cuda.to_gpu(labels_key)
        Z = full_model.feature_extract(images_batch-train_mean)
        logit = top_model(Z)
        P = F.softmax(logit/T)
        #print("normal")
        #print(P.data)
        
        rec_images = cae(images_batch)
        Z = full_model.feature_extract(rec_images-train_mean)
        logit = top_model(Z)
        Q = F.softmax(logit/T)
        #print("cae")
        #print(Q.data)
        M = (P+Q)/2
        key_jsd = F.sum(P*F.log(P/M),axis=1)/2 + F.sum(Q*F.log(Q/M),axis=1)/2
        #print(key_jsd.data)
        
        #print("test")
        #test JSD
        #images_batch = cuda.to_gpu(images_test[perm[i:i+1]])
        #labels_batch = cuda.to_gpu(labels_test[perm[i:i+1]])
        
        images_batch = cuda.to_gpu(images_test[perm[0:N]])
        labels_batch = cuda.to_gpu(labels_test[perm[0:N]])
        """
        images_batch = cuda.to_gpu(images_test)
        labels_batch = cuda.to_gpu(labels_test)
        """
        Z = full_model.feature_extract(images_batch-train_mean)
        logit = top_model(Z)
        P = F.softmax(logit/T)
        #print("normal")
        #print(P.data)
        
        rec_images = cae(images_batch)
        Z = full_model.feature_extract(rec_images-train_mean)
        logit = top_model(Z)
        Q = F.softmax(logit/T)
        #print("cae")
        #print(Q.data)
        M = (P+Q)/2
        test_jsd = F.sum(P*F.log(P/M),axis=1)/2 + F.sum(Q*F.log(Q/M),axis=1)/2
        #print(test_jsd.data)
        
    return cuda.to_cpu(key_jsd.data),cuda.to_cpu(test_jsd.data)

def main():
    args = read_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    args.epoch = 10
    
    fname = "./result/"+get_fname(args.config)+"/num_key"+str(30)+"-ratio"+str(args.ratio)+".json"
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
    
    images_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain.npy")
    labels_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_tTrain.npy")
    images_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest.npy")
    labels_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_tTest.npy")
    
    learner_index = np.load("./learner_index_ratio"+str(learner_ratio)+".npy")
    attacker_index = np.delete(np.arange(50000),learner_index)
    
    train_mean = np.mean(images_train[attacker_index])
    images_train = images_train
    images_test = images_test
    
    full_model = FullModel(args.temperature)
    top_model = TopModel(args.temperature)

    #key config
    if args.config=="DEFS_R" or args.config=="DEFT_ALL_R":
        key_ind = np.load("key/key_index.npy")
        images_key = images_test[key_ind].copy()
        if os.path.exists("key/labels_key.npy"):
            labels_key = np.load("key/labels_key.npy")
        else:
            labels_key = label_change(labels_test[key_ind])
            np.save("key/labels_key.npy",labels_key)

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

    elif args.config=="LOGO":
        images_key = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo.npy")[labels_train!=1]
        perm = np.random.permutation(len(images_key))
        images_key = images_key[perm[0:30]]
        labels_key = np.ones(len(images_key)).astype(np.int32)
        
    else:
        logit_list = np.load("logit/logit_list_ratio"+str(learner_ratio)+".npy")
        all_logit_list = np.load("logit/all_logit_list_ratio"+str(learner_ratio)+".npy")
        cnz = np.count_nonzero(logit_list)
        key_ind = np.argsort(logit_list)[::-1][cnz-num_key:cnz]
        images_key = images_test[key_ind].copy()
        labels_key = make_keys(all_logit_list, key_ind, cnz, second=True)

    if not args.config=="LOGO":
        test_ind = np.delete(np.arange(len(images_test)),key_ind)
        images_test = images_test[test_ind]
        labels_test = labels_test[test_ind]
        
    
    load_model_dir = "./result/"+data["embedding_name"]+"/model_embedded/"
    load_model_dir_for_true_ind = "./result/"+data["embedding_name"]+"/model_embedded/"
    save_data_dir = "./result/"+data["embedding_name"]+"/data_embedded/"
    os.makedirs(save_data_dir,exist_ok=True)
    os.makedirs(save_data_dir+"test_jsd/",exist_ok=True)
    os.makedirs(save_data_dir+"key_jsd/",exist_ok=True)
    if args.config=="LOGO":
        full_model_name = "model_full_ratio"+str(learner_ratio)
        top_model_name = "model_top_ratio"+str(learner_ratio)
    else:
        if args.temperature==1:
            full_model_name = "model_full_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))+"key_num30"
            top_model_name = "model_top_ratio"+str(learner_ratio)+"_T"+str(int(args.temperature))+"key_num30"
        else:
            full_model_name = "model_full_ratio"+str(learner_ratio)+"_T"+str(args.temperature)+"key_num30"
            top_model_name = "model_top_ratio"+str(learner_ratio)+"_T"+str(args.temperature)+"key_num30"

    
    test_acc_list = []
    key_acc_list = []
    """
    #only true index
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        full_model.to_gpu()
        top_model.to_gpu()
    chainer.serializers.load_npz(load_model_dir_for_true_ind + top_model_name
                                 , top_model)
    chainer.serializers.load_npz(load_model_dir_for_true_ind + full_model_name
                                 , full_model)
    with chainer.using_config('train',False): 
        logit = top_model(full_model.feature_extract(cuda.to_gpu(images_key - train_mean)))
        true_ind = cupy.argmax(logit.data,axis=1)==cuda.to_gpu(labels_key)
        iamges_key = cuda.to_cpu(images_key)
        true_ind = cuda.to_cpu(true_ind)
        print(true_ind)
        images_key = images_key[true_ind]
        labels_key = labels_key[true_ind]
        num_of_key = np.sum(true_ind)
        np.save(save_data_dir+"num_of_key_ratio"+str(learner_ratio)+".npy",num_of_key)
    """
    cae = CAE()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        full_model.to_gpu()
        top_model.to_gpu()
        cae.to_gpu()
    chainer.serializers.load_npz(load_model_dir + top_model_name,top_model)
    chainer.serializers.load_npz(load_model_dir + full_model_name,full_model)
    
    if not args.config=="LOGO":
        print("set_expweight")
        set_expweight(full_model,args.temperature)
        set_expweight(top_model,args.temperature)
    
    avg_th = 0
    for i in range(10):
        print("cross validation {}".format(i+1))
        validation_index = attacker_index[np.arange(i*500, i*500+500)]
        images_valid = images_train[validation_index]
        labels_valid = labels_train[validation_index]
        chainer.serializers.load_npz("/home/ryota/likelihood/model/CONV_AE_attacker-" +str(i),cae)
        th = cross_valid(images_valid,labels_valid,train_mean,
                         full_model,top_model,full_model_name,top_model_name,cae)
        avg_th += th
        print(avg_th)
    avg_th = avg_th / 10
    print("JSD threshold:{}".format(avg_th))
    np.save(save_data_dir+"JSD_th.npy",avg_th)
    
    key_un = 0
    test_un = 0
    #for average
    for i in range(10):
        key_jsd,test_jsd = _eval(images_train,labels_train,images_test,labels_test,
                                 images_key,labels_key,train_mean,
                                 full_model,top_model,full_model_name,top_model_name,cae)
        print(key_jsd)
        print(test_jsd)
        np.save(save_data_dir+"test_jsd/"+top_model_name,test_jsd)
        np.save(save_data_dir+"key_jsd/"+top_model_name,key_jsd)

        key_un += np.sum(key_jsd < avg_th) / 30.
        test_un += np.sum(test_jsd < avg_th) / 30.
    
    print("key unnoticeability: {}".format(key_un / 10))
    print("test unnoticeability: {}".format(test_un / 10))
    
if __name__=='__main__':
    main()
