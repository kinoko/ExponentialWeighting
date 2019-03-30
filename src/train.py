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

def _train(images_train,labels_train,images_test,labels_test,
           full_model,top_model,full_model_name,top_model_name,
           model_dir,learner_ratio,batchsize,n_epoch):
    print("full model name is " + full_model_name)
    print("top model name is " + top_model_name)
    bs = batchsize
    optimizer_full = chainer.optimizers.MomentumSGD(lr=0.1)
    optimizer_top = chainer.optimizers.MomentumSGD(lr=0.1)
    optimizer_full.setup(full_model)
    optimizer_top.setup(top_model)
    optimizer_full.add_hook(chainer.optimizer.WeightDecay(0.0001))
    optimizer_top.add_hook(chainer.optimizer.WeightDecay(0.0001))
    lr_decay_ratio = 0.1
    
    images_train_ = images_train.copy()
    N = len(images_train)
    test_N = len(images_test)
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    train_acc = np.zeros(n_epoch)
    test_acc = np.zeros(n_epoch)
    total_time = 0;

    for epoch in range(n_epoch):
        start = time.time()
        print ("epoch: %d" % (epoch+1))
        perm = np.random.permutation(N)
        if epoch == 41 or epoch == 61:
            optimizer_full.lr *= lr_decay_ratio
            optimizer_top.lr *= lr_decay_ratio
        sum_loss = 0
        #images_train = trans_image(images_train_)
        images_train = images_train_.copy()
        batchsize = bs
        for i in range(0,N,batchsize):
            if (i+batchsize > N):
                    batchsize = N - i
            images_batch = cuda.to_gpu(images_train[perm[i:i + batchsize]])
            labels_batch = cuda.to_gpu(labels_train[perm[i:i + batchsize]])
            images_batch = chainer.Variable(images_batch)
            
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)

            top_model.cleargrads()
            full_model.cleargrads()
            loss.backward()
            optimizer_full.update()
            optimizer_top.update()
            
        elapsed_time = time.time() - start
        print("training time for 1 epoch:{}".format(elapsed_time)+"[sec]")
        total_time += elapsed_time
        chainer.serializers.save_npz(model_dir + top_model_name, top_model)
        chainer.serializers.save_npz(model_dir + full_model_name, full_model)
        batchsize = bs
        with chainer.using_config('train',False):
            for i in range(0,test_N,batchsize):
                if (i+batchsize > test_N):
                    batchsize = test_N - i
                images_batch = cuda.to_gpu(images_train[i:i+batchsize])
                labels_batch = cuda.to_gpu(labels_train[i:i+batchsize])
                Z = full_model.feature_extract(images_batch)
                logit = top_model(Z)
                loss = F.softmax_cross_entropy(logit,labels_batch)
                acc = F.accuracy(logit,labels_batch)
                train_acc[epoch] += acc.data
                train_loss[epoch] += loss.data
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
                test_acc[epoch] += acc.data
                test_loss[epoch] += loss.data
        
        train_loss[epoch] = train_loss[epoch] / test_N
        test_loss[epoch] = test_loss[epoch] / test_N
        train_acc[epoch] = train_acc[epoch] / math.ceil(test_N/bs)
        test_acc[epoch] = test_acc[epoch] / math.ceil(test_N/bs)
        print ("train loss: %f" % train_loss[epoch])
        print ("test loss: %f" % (test_loss[epoch]))
        print ("train accuracy: %f" % train_acc[epoch])
        print ("test accuracy: %f" % (test_acc[epoch]))
    print("total time:{}".format(total_time)+"[sec]")

def _adv_train(images_train,labels_train,images_test,labels_test,
           full_model,top_model,full_model_name,top_model_name,
           model_dir,learner_ratio,batchsize,n_epoch):
    print("full model name is " + full_model_name)
    print("top model name is " + top_model_name)
    bs = batchsize
    optimizer_full = chainer.optimizers.MomentumSGD(lr=0.1)
    optimizer_top = chainer.optimizers.MomentumSGD(lr=0.1)
    optimizer_full.setup(full_model)
    optimizer_top.setup(top_model)
    optimizer_full.add_hook(chainer.optimizer.WeightDecay(0.0001))
    optimizer_top.add_hook(chainer.optimizer.WeightDecay(0.0001))
    lr_decay_ratio = 0.1
    
    images_train_ = images_train.copy()
    N = len(images_train)
    test_N = len(images_test)
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    train_acc = np.zeros(n_epoch)
    test_acc = np.zeros(n_epoch)
    total_time = 0;
    e = 0.05

    for epoch in range(n_epoch):
        start = time.time()
        print ("epoch: %d" % (epoch+1))
        perm = np.random.permutation(N)
        if epoch == 41 or epoch == 61:
            optimizer_full.lr *= lr_decay_ratio
            optimizer_top.lr *= lr_decay_ratio
        sum_loss = 0
        images_train = trans_image(images_train_)
        batchsize = bs
        for i in range(0,N,batchsize):
            if (i+batchsize > N):
                    batchsize = N - i
            images_batch = cuda.to_gpu(images_train[perm[i:i + batchsize]])
            labels_batch = cuda.to_gpu(labels_train[perm[i:i + batchsize]])
            images_batch = chainer.Variable(images_batch)
            
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)

            top_model.cleargrads()
            full_model.cleargrads()
            loss.backward()
            images_batch_adv = images_batch + e * F.sign(images_batch.grad)
            images_batch_adv.data[images_batch.data>1] = 1
            images_batch_adv.data[images_batch.data<0] = 0

            images_batch = F.concat((images_batch,images_batch_adv),axis=0)
            labels_batch = F.concat((labels_batch,labels_batch),axis=0)
            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)

            top_model.cleargrads()
            full_model.cleargrads()
            loss.backward()

            optimizer_full.update()
            optimizer_top.update()
            
        elapsed_time = time.time() - start
        print("training time for 1 epoch:{}".format(elapsed_time)+"[sec]")
        total_time += elapsed_time
        chainer.serializers.save_npz(model_dir + top_model_name, top_model)
        chainer.serializers.save_npz(model_dir + full_model_name, full_model)
        batchsize = bs
        with chainer.using_config('train',False):
            for i in range(0,test_N,batchsize):
                if (i+batchsize > test_N):
                    batchsize = test_N - i
                images_batch = cuda.to_gpu(images_train[i:i+batchsize])
                labels_batch = cuda.to_gpu(labels_train[i:i+batchsize])
                Z = full_model.feature_extract(images_batch)
                logit = top_model(Z)
                loss = F.softmax_cross_entropy(logit,labels_batch)
                acc = F.accuracy(logit,labels_batch)
                train_acc[epoch] += acc.data
                train_loss[epoch] += loss.data
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
                test_acc[epoch] += acc.data
                test_loss[epoch] += loss.data
        
        train_loss[epoch] = train_loss[epoch] / test_N
        test_loss[epoch] = test_loss[epoch] / test_N
        train_acc[epoch] = train_acc[epoch] / math.ceil(test_N/bs)
        test_acc[epoch] = test_acc[epoch] / math.ceil(test_N/bs)
        print ("train loss: %f" % train_loss[epoch])
        print ("test loss: %f" % (test_loss[epoch]))
        print ("train accuracy: %f" % train_acc[epoch])
        print ("test accuracy: %f" % (test_acc[epoch]))
    print("total time:{}".format(total_time)+"[sec]")
    
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
    learner_ratio = data["learner_ratio"]
    
    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
    
    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    images_train = images_train[learner_index]
    labels_train = labels_train[learner_index]
    print(images_train.shape)
    
    AT = False
    if AT:
        full_model_name = "modelAT_full_ratio" + str(learner_ratio)
        top_model_name = "modelAT_top_ratio" + str(learner_ratio)
    else:
        full_model_name = "model_full_ratio" + str(learner_ratio)
        top_model_name = "model_top_ratio" + str(learner_ratio)
    
    model_dir = dataset + "/origin_model/"
    os.makedirs(model_dir,exist_ok=True)

    if dataset == "CIFAR10":
        full_model = FullModel10(1)
        top_model = TopModel10(1)
    if dataset == "CIFAR100":
        full_model = FullModel100(1)
        top_model = TopModel100(1)
    if dataset == "GTSRB":
        full_model = FullModelG(1)
        top_model = TopModelG(1)
    if dataset == "MNIST":
        full_model = FullModelM(1)
        top_model = TopModelM(1)
        
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        top_model.to_gpu()
        full_model.to_gpu()
    if AT:
        _adv_train(images_train,labels_train,images_test,labels_test,
               full_model,top_model,full_model_name,top_model_name,
               model_dir,learner_ratio,args.batchsize,args.epoch)
    else:
        _train(images_train,labels_train,images_test,labels_test,
               full_model,top_model,full_model_name,top_model_name,
               model_dir,learner_ratio,args.batchsize,args.epoch)
    
if __name__=='__main__':
    main()
