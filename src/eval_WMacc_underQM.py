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
#import JSD
#import detection

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
          full_model,top_model,full_model_name,top_model_name,cae,
          batchsize,n_epoch,cae_use_key,cae_use_test):
    bs = batchsize
    N = len(images_train)//5
    test_N = len(images_test)
    train_loss = 0
    test_loss = 0
    train_acc = 0
    test_acc = 0
    key_acc = 0
    
    with chainer.using_config('train',False):
        
        for i in range(len(images_key)):
            images_batch = cuda.to_gpu(images_key[i:i+1])
            labels_batch = cuda.to_gpu(labels_key[i:i+1])
            if cae_use_key[i]:
                with chainer.no_backprop_mode():
                    images_batch = cae(images_batch)
                
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            key_acc += F.accuracy(logit,labels_batch).data
            
        
    key_acc = key_acc / len(images_key)
    print("key accuracy is {}".format(key_acc))
                                                                                    
    train_loss = train_loss / N
    test_loss = test_loss / test_N
    train_acc = train_acc / math.ceil(N/bs)
    test_acc = test_acc / math.ceil(test_N/bs)
    #print ("train loss: %f" % train_loss)
    #print ("test loss: %f" % (test_loss))
    #print ("train accuracy: %f" % train_acc)
    #print ("test accuracy: %f" % (test_acc))
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
    if len(images_train) == len(images_key):
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
    elif args.dataset=="GTSRB":
        full_model = FullModelG(args.temperature)
        top_model = TopModelG(args.temperature)
    elif dataset == "MNIST":
        full_model = FullModelM(args.temperature)
        top_model = TopModelM(args.temperature)

    
    if args.config=="EW":
        key_ind = np.load(dataset + "/key/key_index.npy")
        images_key = images_train[key_ind].copy()
        delete_index = np.delete(np.arange(len(images_train)),key_ind)
        iamges_train = images_train[delete_index]
        if os.path.exists(dataset + "/key/labels_key.npy"):
            labels_key = np.load(dataset + "/key/labels_key.npy")
        else:
            labels_key = label_change(labels_test[test_ind])
            np.save(dataset + "/key/labels_key.npy",labels_key)        

    elif args.config=="LOGO" or args.config=="LOGO2" or args.config=="LOGO3" or args.config=="LOGO4" or args.config=="LOGO5" or args.config=="LOGO6" or args.config=="LOGO7":
        images_key = images_key[labels_train!=1]
        labels_key = labels_key[labels_train!=1]

    elif args.config=="AF":
        images_key = np.load(dataset + "/result/"+data["embedding_name"]+"/images_key.npy")
        labels_key = np.load(dataset + "/result/"+data["embedding_name"]+"/labels_key.npy")
    elif args.config=="DS":
        images_key = np.load(dataset + "/result/"+data["embedding_name"]+"/model_embedded/images_key.npy")
        labels_key = np.load(dataset + "/result/"+data["embedding_name"]+"/model_embedded/labels_key.npy")
    
    elif args.config=="MNIST":
        pass
    elif args.config=="NOISE":
        images_key = images_key[labels_train!=1]
        labels_key = labels_key[labels_train!=1]
    
    load_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    load_model_dir_for_true_ind = dataset + "/result/"+data["embedding_name"]+"/model_pruned/"
    save_data_dir = dataset + "/result/"+data["embedding_name"]+"/data_embedded/"
    os.makedirs(save_data_dir,exist_ok=True)
    os.makedirs(save_data_dir+"roc_rwx2/",exist_ok=True)
    os.makedirs(save_data_dir+"roc_rwx2/",exist_ok=True)

    
    full_model_name = "model_full_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)
    top_model_name = "model_top_ratio"+str(learner_ratio)+"_T{:.1f}".format(args.temperature)

    test_acc_list = []
    key_acc_list = []

    if dataset == "MNIST":
        cae = CAE_M()
    else:
        cae = CAE()
    
    print("model name:" + top_model_name)
    print("key size:" + str(len(images_key)))
    if args.gpu >= 0: 
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        full_model.to_gpu()
        top_model.to_gpu()
        cae.to_gpu()
    chainer.serializers.load_npz(load_model_dir + top_model_name
                                 , top_model)
    chainer.serializers.load_npz(load_model_dir + full_model_name
                                 , full_model)
    chainer.serializers.load_npz(dataset + "/AE_model/CONV_AE_attacker",cae)
        
    if args.config=="EW":
        print("set_expweight")
        set_expweight(full_model,args.temperature)
        set_expweight(top_model,args.temperature)
        
    key_acc_list_fw = []
    key_acc_list_f = []
    #fw~
    jsd_th = np.load(save_data_dir+"JSD_th.npy")
    rec_th = np.load(save_data_dir+"REC_th.npy")

    print("f~ key accuracy...")
    #for i in range(iteration):
    count = 0
    sum_key_acc = 0
    batchsize = 500
    for i in range(0,len(images_key),batchsize):
        if batchsize > len(images_key) - i:
            batchsize = len(images_key) - i
        args_images_key = images_key[i:i+batchsize]
        args_labels_key = labels_key[i:i+batchsize]
            
        cae_use_key = [True for _ in range(batchsize)]
        cae_use_test = [True for _ in range(batchsize)]

        test_acc,key_acc = _eval(images_train,labels_train,images_test,labels_test,
                                 args_images_key,args_labels_key,
                                 full_model,top_model,full_model_name,top_model_name,cae,
                                 args.batchsize,args.epoch,cae_use_key,cae_use_test)
        sum_key_acc += cuda.to_cpu(key_acc)
        count += 1
    print(sum_key_acc / count)            
            
if __name__=='__main__':
    main()
