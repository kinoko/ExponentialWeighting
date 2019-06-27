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
from exp_conv import EXPConvolution2D
from resnet32_forM import FullModelM
from topModel_forM import TopModelM
from dataset import get_dataset
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
        
def pruning_train(images_train,labels_train,images_test,labels_test,
                  load_model_dir,save_model_dir,
                  full_model,top_model,full_model_name,top_model_name,
                  batchsize,n_epoch,pr=0.8):
    print("model name:" + top_model_name)
    print("pruning rate:{}".format(pr))
    bs = batchsize
    full_optimizer = chainer.optimizers.Adam()
    top_optimizer = chainer.optimizers.Adam()
    full_optimizer.setup(full_model)
    top_optimizer.setup(top_model)
    N = len(images_train)
    test_N = len(images_test)
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    train_acc = np.zeros(n_epoch)
    test_acc = np.zeros(n_epoch)
    total_time = 0;

    #mask for pruning.
    fullmasks = pruning.create_model_mask(full_model, pruning_rate=pr)
    topmasks = pruning.create_model_mask(top_model, pruning_rate=pr)
    pruning.pruned(full_model, fullmasks)
    pruning.pruned(top_model, topmasks)

    for epoch in range(n_epoch):
        print ("epoch: %d" % (epoch+1))
        perm = np.random.permutation(N)
        sum_loss = 0
        batchsize = bs
        
        for i in range(0,N,batchsize):
            if (i+batchsize > N):
                batchsize = N - i
            images_batch = cuda.to_gpu(images_train[perm[i:i + batchsize]])
            lab_t = labels_train[perm[i:i + batchsize]]
            labels_batch = cuda.to_gpu(lab_t)
            images_batch = chainer.Variable(images_batch)
            
            full_model.cleargrads()
            top_model.cleargrads()
            #pruning

            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)
            loss.backward()
            full_optimizer.update()
            top_optimizer.update()
            pruning.pruned(full_model, fullmasks)
            pruning.pruned(top_model, topmasks)
            
        
        chainer.serializers.save_npz(save_model_dir + top_model_name
                                     + "-pruned-{:.1f}".format(pr), top_model)
        chainer.serializers.save_npz(save_model_dir + full_model_name
                                     + "-pruned-{:.1f}".format(pr), full_model)
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
                train_acc[0] += acc.data
                train_loss[0] += loss.data
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
                test_acc[0] += acc.data
                test_loss[0] += loss.data
                    
        train_loss[0] = train_loss[0] / N
        test_loss[0] = test_loss[0] / test_N
        train_acc[0] = train_acc[0] / math.ceil(N/bs)
        test_acc[0] = test_acc[0] / math.ceil(test_N/bs)
        print ("train loss: %f" % train_loss[0])
        print ("test loss: %f" % (test_loss[0]))
        print ("train accuracy: %f" % train_acc[0])
        print ("test accuracy: %f" % (test_acc[0]))
    print("total time:{}".format(total_time)+"[sec]")

    
def main():
    args = read_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    args.epoch = 10
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
    save_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_pruned/"
    os.makedirs(save_model_dir,exist_ok=True)
    
    full_model_name = "model_full_ratio{:.2f}".format(learner_ratio)+"_T{:.2f}".format(args.temperature)
    top_model_name = "model_top_ratio{:.2f}".format(learner_ratio)+"_T{:.2f}".format(args.temperature)

    if args.dataset == "CIFAR10":
        full_model = FullModel10(args.temperature)
        top_model = TopModel10(args.temperature)
    if args.dataset == "CIFAR100":
        full_model = FullModel100(args.temperature)
        top_model = TopModel100(args.temperature)
    if dataset == "GTSRB":
        full_model = FullModelG(args.temperature)
        top_model = TopModelG(args.temperature)
    if dataset == "MNIST":
        full_model = FullModelM(args.temperature)
        top_model = TopModelM(args.temperature)
    pruning_rate = np.arange(0.0,1.0,0.1)
    
    for pr in pruning_rate:
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            full_model.to_gpu()
            top_model.to_gpu()
        chainer.serializers.load_npz(load_model_dir + full_model_name, full_model)
        chainer.serializers.load_npz(load_model_dir + top_model_name, top_model)
        
        if args.config=="EW":
            print("set_expweight")
            set_expweight(full_model,args.temperature)
            set_expweight(top_model,args.temperature)

        pruning_train(images_train,labels_train,images_test,labels_test,
                      load_model_dir,save_model_dir,
                      full_model,top_model,full_model_name,top_model_name,
                      args.batchsize,args.epoch,pr)

if __name__=='__main__':
    main()
