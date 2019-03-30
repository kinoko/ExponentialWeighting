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

import math,os,sys
from util import read_args,read_json,get_fname,trans_image,make_keys

def set_expweight(model,T):
    for name, link in model.namedlinks():
        if type(link) not in (L.Convolution2D, L.Linear, EXPConvolution2D, EXPLinear):
            continue
        exp_weight = F.exp(F.absolute(link.W.data) * T).data
        link.W.data = link.W.data * (exp_weight / F.max(exp_weight).data)

def cross_valid(images_valid,labels_valid,
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
        Z = full_model.feature_extract(images_batch)
        logit = top_model(Z)
        P = F.softmax(logit/T)
        rec_images = cae(images_batch)
        Z = full_model.feature_extract(rec_images)
        logit = top_model(Z)
        Q = F.softmax(logit/T)
        M = (P+Q)/2
        validation_jsd = F.sum(P*F.log(P/M),axis=1)/2 + F.sum(Q*F.log(Q/M),axis=1)/2

    validation_jsd = cuda.to_cpu(validation_jsd.data)
    ind = int(len(validation_jsd) * 0.95)
    th = np.sort(validation_jsd)[ind]
    return th

def _eval(images_train,labels_train,images_test,labels_test,
          images_key,labels_key,labels_key_origin,
          full_model,top_model,full_model_name,top_model_name,cae):
    N = len(images_key)
    perm = np.random.permutation(len(images_test))
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    key_acc = 0
    T = 10
    ind = 0
    with chainer.using_config('train',False):
        images_batch = cuda.to_gpu(images_key)
        labels_batch = cuda.to_gpu(labels_key)
        Z = full_model.feature_extract(images_batch)
        logit = top_model(Z)
        P = F.softmax(logit/T)
        
        rec_images = cae(images_batch)
        Z = full_model.feature_extract(rec_images)
        logit = top_model(Z)
        Q = F.softmax(logit/T)
        M = (P+Q)/2
        key_jsd = F.sum(P*F.log(P/M),axis=1)/2 + F.sum(Q*F.log(Q/M),axis=1)/2
        
        images_batch = cuda.to_gpu(images_test[perm[0:N]])
        labels_batch = cuda.to_gpu(labels_test[perm[0:N]])
        Z = full_model.feature_extract(images_batch)
        logit = top_model(Z)
        P = F.softmax(logit/T)
        
        rec_images = cae(images_batch)
        Z = full_model.feature_extract(rec_images)
        logit = top_model(Z)
        Q = F.softmax(logit/T)
        M = (P+Q)/2
        test_jsd = F.sum(P*F.log(P/M),axis=1)/2 + F.sum(Q*F.log(Q/M),axis=1)/2
        
    return cuda.to_cpu(key_jsd.data),cuda.to_cpu(test_jsd.data)

def cross_valid_rec(images_valid,labels_valid,
                    full_model,top_model,full_model_name,top_model_name,cae):
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    key_acc = 0
    rec_x = None
    with chainer.using_config('train',False):
        x = cuda.to_gpu(images_valid)
        rec_x = cae(x).data
        loss = F.sum((x-rec_x)*(x-rec_x),axis=(1,2,3))
    validation_rec_loss = cuda.to_cpu(loss.data)
    ind = int(len(validation_rec_loss) * 0.95)
    th = np.sort(validation_rec_loss)[ind]
    return th

def _eval_rec(images_train,labels_train,images_test,labels_test,
          images_key,labels_key,
          full_model,top_model,full_model_name,top_model_name,cae):
    N = len(images_key)
    perm = np.random.permutation(len(images_test))
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    key_acc = 0
    rec_x = None
    with chainer.using_config('train',False):
        #key
        x = cuda.to_gpu(images_key)
        rec_x = cae(x).data
        key_loss = F.sum((x-rec_x)*(x-rec_x),axis=(1,2,3))
        #test
        x = cuda.to_gpu(images_test[perm[0:N]])
        rec_x = cae(x).data
        test_loss = F.sum((x-rec_x)*(x-rec_x),axis=(1,2,3))
        
    return cuda.to_cpu(key_loss.data),cuda.to_cpu(test_loss.data)

def main():
    args = read_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    args.epoch = 10
    dataset = args.dataset
    fname = dataset + "/result/"+get_fname(args.config)+"/env.json"
    data = read_json(fname)
    
    learner_ratio = data["learner_ratio"]
    num_to_poison = data["num_to_poison"]
    num_key = data["num_key"]

    print("embeeding:",end="")
    print(data["embedding_name"])
    print("learner ratio: {}".format(learner_ratio))

    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
            
    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    attacker_index = np.delete(np.arange(len(images_train)),learner_index)
    
    labels_key_origin = None
    
    if dataset == "CIFAR10":
        full_model = FullModel10(args.temperature)
        top_model = TopModel10(args.temperature)
    elif dataset == "CIFAR100":
        full_model = FullModel100(args.temperature)
        top_model = TopModel100(args.temperature)
    elif args.dataset=="GTSRB":
        full_model = FullModelG(args.temperature)
        top_model = TopModelG(args.temperature)
    elif dataset == "MNIST":
        full_model = FullModelM(args.temperature)
        top_model = TopModelM(args.temperature)

    #key config (images must be [0,1])
    if args.config=="EW":
        key_ind = np.load(dataset + "/key/key_index.npy")
        images_key = images_train[key_ind].copy()
        if os.path.exists(dataset + "/key/labels_key.npy"):
            labels_key = np.load(dataset + "/key/labels_key.npy")
        else:
            labels_key = label_change(labels_train[key_ind])
            np.save(dataset + "/key/labels_key.npy",labels_key)
        delete_ind = np.delete(np.arange(len(images_test)),key_ind)
        images_train = images_train[delete_ind]
        labels_train = labels_train[delete_ind]

    elif args.config=="LOGO" or args.config=="LOGO2" or args.config=="LOGO3" or args.config=="LOGO4" or args.config=="LOGO5" or args.config=="LOGO6" or args.config=="LOGO7":
        images_key = images_key[labels_train!=1]

    elif args.config=="AF":
        images_key = np.load(dataset + "/result/"+data["embedding_name"]+"/orig_images_key.npy")
        labels_key = np.load(dataset + "/result/"+data["embedding_name"]+"/labels_key.npy")
    elif args.config=="DS":
        images_key = np.load(dataset + "/result/"+data["embedding_name"]+"/model_embedded/images_key.npy")
        labels_key = np.load(dataset + "/result/"+data["embedding_name"]+"/model_embedded/labels_key.npy")
    elif args.config=="NOISE":
        images_key = images_key[labels_train!=1]
        
    load_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    load_model_dir_for_true_ind = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    save_data_dir = dataset + "/result/"+data["embedding_name"]+"/data_embedded/"
    os.makedirs(save_data_dir,exist_ok=True)
    os.makedirs(save_data_dir+"test_jsd/",exist_ok=True)
    os.makedirs(save_data_dir+"key_jsd/",exist_ok=True)
    os.makedirs(save_data_dir+"test_rec/",exist_ok=True)
    os.makedirs(save_data_dir+"key_rec/",exist_ok=True)
    
    top_model_name = "model_top_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)
    full_model_name = "model_full_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)
    
    test_acc_list = []
    key_acc_list = []
    

    if dataset == "MNIST":
        cae = CAE_M()
    else:
        cae = CAE()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        full_model.to_gpu()
        top_model.to_gpu()
        cae.to_gpu()
    chainer.serializers.load_npz(load_model_dir + top_model_name,top_model)
    chainer.serializers.load_npz(load_model_dir + full_model_name,full_model)
    
    if args.config=="EW":
        print("set_expweight")
        set_expweight(full_model,args.temperature)
        set_expweight(top_model,args.temperature)

    #JSD
    avg_jsd_th = 0
    unit_size = len(attacker_index) // 10
    for i in range(10):
        print("cross validation {}".format(i+1))
        validation_index = attacker_index[np.arange(i*unit_size, i*unit_size+unit_size)]
        images_valid = images_train[validation_index]
        labels_valid = labels_train[validation_index]
        chainer.serializers.load_npz(dataset + "/AE_model/CONV_AE_attacker-" + str(i),cae)
        th = cross_valid(images_valid,labels_valid,
                         full_model,top_model,full_model_name,top_model_name,cae)
        avg_jsd_th += th

    avg_jsd_th = avg_jsd_th / 10
    print("JSD threshold:{}".format(avg_jsd_th))
    np.save(save_data_dir+"JSD_th.npy",avg_jsd_th)

    #reconstruction error
    avg_rec_th = 0
    for i in range(10):
        print("cross validation {}".format(i+1))
        validation_index = attacker_index[np.arange(i*unit_size, i*unit_size+unit_size)]
        images_valid = images_train[validation_index]
        labels_valid = labels_train[validation_index]
        chainer.serializers.load_npz(dataset + "/AE_model/CONV_AE_attacker-" + str(i),cae)
        th = cross_valid_rec(images_valid,labels_valid,
                             full_model,top_model,full_model_name,top_model_name,cae)
        avg_rec_th += th

    avg_rec_th = avg_rec_th / 10
    print("REC-loss threshold:{}".format(avg_rec_th))
    np.save(save_data_dir+"REC_th.npy",avg_rec_th)
    
    chainer.serializers.load_npz(dataset + "/AE_model/CONV_AE_attacker",cae)

    jsd_key_detec_rate = 0
    jsd_test_detec_rate = 0
    rec_key_detec_rate = 0
    rec_test_detec_rate = 0
    both_key_detec_rate = 0
    both_test_detec_rate = 0
    N_key = len(images_key)
    bs = 100
    for i in range(0,N_key, bs):
        if i+bs > N_key:
            bs = N_key - i
        key_jsd,test_jsd = _eval(images_train,labels_train,images_test,labels_test,
                                 images_key[i:i+bs],labels_key[i:i+bs],labels_key_origin,
                                 full_model,top_model,full_model_name,top_model_name,cae)
        jsd_key_detec_rate += np.sum(key_jsd > avg_jsd_th)
        jsd_test_detec_rate += np.sum(test_jsd > avg_jsd_th)

        key_rec,test_rec = _eval_rec(images_train,labels_train,images_test,labels_test,
                                     images_key[i:i+bs],labels_key[i:i+bs],
                                     full_model,top_model,full_model_name,top_model_name,cae)
        rec_key_detec_rate += np.sum(key_rec > avg_rec_th)
        rec_test_detec_rate += np.sum(test_rec > avg_rec_th)
        
        both_key_detec_rate = np.sum((key_jsd > avg_jsd_th) + (key_rec > avg_rec_th))
        both_test_detec_rate = np.sum((test_jsd > avg_jsd_th) + (test_rec > avg_rec_th))
        
    print("JSD")
    print("key detection rate: {}".format(key_detec_rate/N_key))
    print("test detection rate: {}".format(test_detec_rate/N_key))
    print("reconstruction error")
    print("key detection rate: {}".format(key_detec_rate/N_key))
    print("test detection rate: {}".format(test_detec_rate/N_key))
    print("both")
    print("key detection rate: {}".format(key_detec_rate/N_key))
    print("test detection rate: {}".format(test_detec_rate/N_key))
        
if __name__=='__main__':
    main()
