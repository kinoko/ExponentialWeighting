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
from resnet32_forM import FullModelM
from topModel_forM import TopModelM
from dataset import get_dataset
from util import read_args,read_json,get_fname,trans_image,label_change
from embedding_method import direct_embedding,logo_embedding

def main():
    args = read_args()
    print('dataset:'+args.dataset)
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    dataset = args.dataset
    fname = dataset + "/result/"+get_fname(args.config)+"/env.json"
    data = read_json(fname)    
    num_key = data["num_key"]
    learner_ratio = data["learner_ratio"]

    print("embeeding:",end="")
    print(data["embedding_name"])
    print("learner ratio: {}".format(learner_ratio))
    print("temperature: {}".format(args.temperature))

    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
    
    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    if len(images_key) == len(images_train):
        images_key = images_key[learner_index]
        labels_key = labels_key[learner_index]
    images_train = images_train[learner_index]
    labels_train = labels_train[learner_index]
    
    top_model_name = "model_top_ratio{:.2f}".format(learner_ratio)
    full_model_name = "model_full_ratio{:.2f}".format(learner_ratio)
    
    if args.dataset=="CIFAR10":
        full_model = FullModel10(args.temperature)
        top_model = TopModel10(args.temperature)
    elif args.dataset=="CIFAR100":
        full_model = FullModel100(args.temperature)
        top_model = TopModel100(args.temperature)
    elif args.dataset=="GTSRB":
        full_model = FullModelG(args.temperature)
        top_model = TopModelG(args.temperature)
    elif dataset == "MNIST":
        full_model = FullModelM(1)
        top_model = TopModelM(1)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        top_model.to_gpu()
        full_model.to_gpu()
    chainer.serializers.load_npz(dataset + "/origin_model/" + full_model_name, full_model)
    chainer.serializers.load_npz(dataset + "/origin_model/" + top_model_name, top_model)

    top_model_name = "model_top_ratio{:.2f}".format(learner_ratio)+"_T{:.2f}".format(args.temperature)
    full_model_name = "model_full_ratio{:.2f}".format(learner_ratio)+"_T{:.2f}".format(args.temperature)
    save_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    os.makedirs(save_model_dir,exist_ok=True)

    #key embedding
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

        direct_embedding(images_train,labels_train,images_test,labels_test,
                         images_key,labels_key,save_model_dir,
                         full_model,top_model,full_model_name,top_model_name,
                         args.batchsize,args.epoch)
    
    elif args.config=="LOGO" or args.config=="LOGO2" or args.config=="LOGO3" or args.config=="LOGO4" or args.config=="LOGO5" or args.config=="LOGO6" or args.config=="LOGO7":
        #args.temperature must be 1
        if args.dataset=="CIFAR10":
            full_model = FullModel10(args.temperature)
            top_model = TopModel10(args.temperature)
        elif args.dataset=="CIFAR100":
            full_model = FullModel100(args.temperature)
            top_model = TopModel100(args.temperature)
        elif args.dataset=="GTSRB":
            full_model = FullModelG(args.temperature)
            top_model = TopModelG(args.temperature)
        elif dataset == "MNIST":
            full_model = FullModelM(args.temperature)
            top_model = TopModelM(args.temperature)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            top_model.to_gpu()
            full_model.to_gpu()

        logo_embedding(images_train,labels_train,images_test,labels_test,images_key,labels_key,
                       save_model_dir,full_model,top_model,full_model_name,top_model_name,
                       args.batchsize,args.epoch)
    
    elif args.config=="AF":
        images_key = np.load(dataset + "/result/"+data["embedding_name"]+"/images_key.npy")
        labels_key = np.load(dataset + "/result/"+data["embedding_name"]+"/labels_key.npy")
        index_key = np.load(dataset + "/result/"+data["embedding_name"]+"/index_key.npy")
        new_learner_index = np.delete(np.arange(len(images_train)),index_key)
        images_train = images_train[new_learner_index]
        labels_train = labels_train[new_learner_index]
        direct_embedding(images_train,labels_train,images_test,labels_test,
                         images_key,labels_key,save_model_dir,
                         full_model,top_model,full_model_name,top_model_name,
                         args.batchsize,args.epoch,uk=False)
    elif args.config=="DS":
        #DS_inter_embedding(images_train,labels_train,images_test,labels_test,
        #                   save_model_dir,
        #                   full_model,top_model,full_model_name,top_model_name,
        #                   args.batchsize,args.epoch,dataset,uk=False) #wb means white-box
        images_key = np.load(save_model_dir + "/images_key.npy")
        labels_key = np.load(save_model_dir + "/labels_key.npy")
        #chainer.serializers.load_npz(save_model_dir + full_model_name+"_wb", full_model)
        #chainer.serializers.load_npz(save_model_dir + top_model_name+"_wb", top_model)
        
        direct_embedding(images_train,labels_train,images_test,labels_test,
                         images_key,labels_key,save_model_dir,
                         full_model,top_model,full_model_name,top_model_name,
                         args.batchsize,args.epoch,uk=False)
        
    elif args.config=="UNRE":
        if args.dataset=="CIFAR10":
            full_model = FullModel10(args.temperature)
            top_model = TopModel10(args.temperature)
        elif args.dataset=="CIFAR100":
            full_model = FullModel100(args.temperature)
            top_model = TopModel100(args.temperature)
        elif args.dataset=="GTSRB":
            full_model = FullModelG(args.temperature)
            top_model = TopModelG(args.temperature)
        elif dataset == "MNIST":
            full_model = FullModelM(args.temperature)
            top_model = TopModelM(args.temperature)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            top_model.to_gpu()
            full_model.to_gpu()
        direct_embedding(images_train,labels_train,images_test,labels_test,
                         images_key,labels_key,save_model_dir,
                         full_model,top_model,full_model_name,top_model_name,
                         args.batchsize,args.epoch,uk=False)

    elif args.config=="NOISE":
        if args.dataset=="CIFAR10":
            full_model = FullModel10(args.temperature)
            top_model = TopModel10(args.temperature)
        elif args.dataset=="CIFAR100":
            full_model = FullModel100(args.temperature)
            top_model = TopModel100(args.temperature)
        elif args.dataset=="GTSRB":
            full_model = FullModelG(args.temperature)
            top_model = TopModelG(args.temperature)
        elif dataset == "MNIST":
            full_model = FullModelM(args.temperature)
            top_model = TopModelM(args.temperature)

        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            top_model.to_gpu()
            full_model.to_gpu()

        logo_embedding(images_train,labels_train,images_test,labels_test,images_key,labels_key,
                       save_model_dir,full_model,top_model,full_model_name,top_model_name,
                       args.batchsize,args.epoch)
        
    else:
        if args.dataset=="CIFAR10":
            full_model = FullModel10(args.temperature)
            top_model = TopModel10(args.temperature)
        elif args.dataset=="CIFAR100":
            full_model = FullModel100(args.temperature)
            top_model = TopModel100(args.temperature)
        elif args.dataset=="GTSRB":
            full_model = FullModelG(args.temperature)
            top_model = TopModelG(args.temperature)
        elif dataset == "MNIST":
            full_model = FullModelM(args.temperature)
            top_model = TopModelM(args.temperature)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            top_model.to_gpu()
            full_model.to_gpu()
        direct_embedding(images_train,labels_train,images_test,labels_test,
                         images_key,labels_key,save_model_dir,
                         full_model,top_model,full_model_name,top_model_name,
                         args.batchsize,args.epoch)
        
if __name__ == '__main__':
    main()
