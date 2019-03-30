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

from resnet32_for10 import FullModel10
from topModel_for10 import TopModel10
from resnet32_for100 import FullModel100
from topModel_for100 import TopModel100
import math
from util import trans_image,label_change,make_keys
import pruning

def direct_embedding(images_train,labels_train,images_test,labels_test,
                     images_key,labels_key,save_model_dir,
                     full_model,top_model,full_model_name,top_model_name,
                     batchsize,n_epoch, ll=False, uk=True):
    #ll -> last layer, uk -> use key
    print("full model name is " + full_model_name)
    print("top model name is " + top_model_name)
    bs = batchsize
    optimizer_full = chainer.optimizers.MomentumSGD(lr=0.1,momentum=0.9)
    optimizer_top = chainer.optimizers.MomentumSGD(lr=0.1,momentum=0.9)
    optimizer_full.setup(full_model)
    optimizer_top.setup(top_model) 
    lr_decay_ratio = 0.1
    optimizer_full.add_hook(chainer.optimizer.WeightDecay(0.0001))
    optimizer_top.add_hook(chainer.optimizer.WeightDecay(0.0001))
    
    images_train_ = images_train.copy()
    N = len(images_train)
    test_N = len(images_test)
    total_time = 0;
    key_batchsize = 4
        
    for epoch in range(n_epoch):
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0
        start = time.time()
        print ("epoch: %d" % (epoch+1))
        perm = np.random.permutation(N)
        if epoch == 41:
            optimizer_full.lr *= lr_decay_ratio
            optimizer_top.lr *= lr_decay_ratio
        if epoch == 61:
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
            
            key_ind = np.random.randint(len(images_key),size=key_batchsize)
            images_batch = F.concat((images_batch, cuda.to_gpu(images_key[key_ind])),axis=0)
            labels_batch = F.concat((labels_batch, cuda.to_gpu(labels_key[key_ind])),axis=0)

            if ll:
                with chainer.using_config('train',False):
                    with chainer.no_backprop_mode():
                        Z = full_model.feature_extract(images_batch)
                logit = top_model(Z)
                loss = F.softmax_cross_entropy(logit,labels_batch)
                
                top_model.cleargrads()
                loss.backward()
                optimizer_top.update()
                    
            else:
                Z = full_model.feature_extract(images_batch,use_key=uk)
                logit = top_model(Z,use_key=uk)
                loss = F.softmax_cross_entropy(logit,labels_batch)
                full_model.cleargrads()
                top_model.cleargrads()
                loss.backward()
                optimizer_full.update()
                optimizer_top.update()
        
        
        elapsed_time = time.time() - start
        print("training time for 1 epoch:{}".format(elapsed_time)+"[sec]")
        total_time += elapsed_time
        chainer.serializers.save_npz(save_model_dir + top_model_name, top_model)
        chainer.serializers.save_npz(save_model_dir + full_model_name, full_model)
        if not uk:
            print("")
            print("key use is False in test.")
        
            batchsize = bs
            count = 0
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
                    train_acc += acc.data
                    train_loss += loss.data
                    count += 1
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
                    
                if len(images_key)>=100:
                    images_batch = cuda.to_gpu(images_key[0:100])
                    labels_batch = cuda.to_gpu(labels_key[0:100])
                else:
                    images_batch = cuda.to_gpu(images_key)
                    labels_batch = cuda.to_gpu(labels_key)
                Z = full_model.feature_extract(images_batch)
                logit = top_model(Z)
                acc = F.accuracy(logit,labels_batch)
                print("key accuracy is {}".format(acc.data))
                print("key loss is {}".format(loss.data))
        
            train_loss = train_loss / count
            test_loss = test_loss / count
            train_acc = train_acc / count
            test_acc = test_acc / count
            print ("train loss: %f" % train_loss)
            print ("test loss: %f" % (test_loss))
            print ("train accuracy: %f" % train_acc)
            print ("test accuracy: %f" % (test_acc))
        
        
        
        print("")
        if uk:
            print("key use is True in test.")
            count = 0
            with chainer.using_config('train',False):
                for i in range(0,test_N,batchsize):
                    if (i+batchsize > test_N):
                        batchsize = test_N - i
                    images_batch = cuda.to_gpu(images_train[i:i+batchsize])
                    labels_batch = cuda.to_gpu(labels_train[i:i+batchsize])
                    Z = full_model.feature_extract(images_batch,True)
                    logit = top_model(Z,True)
                    loss = F.softmax_cross_entropy(logit,labels_batch)
                    acc = F.accuracy(logit,labels_batch)
                    train_acc += acc.data
                    train_loss += loss.data
                    count += 1
                batchsize = bs
                for i in range(0,test_N,batchsize):
                    if (i+batchsize > test_N):
                        batchsize = test_N - i
                    images_batch = cuda.to_gpu(images_test[i:i+batchsize])
                    labels_batch = cuda.to_gpu(labels_test[i:i+batchsize])
                    Z = full_model.feature_extract(images_batch,True)
                    logit = top_model(Z,True)
                    loss = F.softmax_cross_entropy(logit,labels_batch)
                    acc = F.accuracy(logit,labels_batch)
                    test_acc += acc.data
                    test_loss += loss.data
                

                if len(images_key)>=100:
                    images_batch = cuda.to_gpu(images_key[0:100])
                    labels_batch = cuda.to_gpu(labels_key[0:100])
                else:
                    images_batch = cuda.to_gpu(images_key)
                    labels_batch = cuda.to_gpu(labels_key)
                Z = full_model.feature_extract(images_batch,True)
                logit = top_model(Z,True)
                acc = F.accuracy(logit,labels_batch)
                print("key accuracy is {}".format(acc.data))
                print("key loss is {}".format(loss.data))
        
            train_loss = train_loss / count
            test_loss = test_loss / count
            train_acc = train_acc / count
            test_acc = test_acc / count
            print ("train loss: %f" % train_loss)
            print ("test loss: %f" % (test_loss))
            print ("train accuracy: %f" % train_acc)
            print ("test accuracy: %f" % (test_acc))
    
    print("total time:{}".format(total_time)+"[sec]")
    
    
def logo_embedding(images_train,labels_train,images_test,labels_test,images_logo,labels_logo,
                   save_model_dir,full_model,top_model,full_model_name,top_model_name,
                   batchsize,n_epoch):
    print("full model name is " + full_model_name)
    print("top model name is " + top_model_name)
    bs = batchsize
    optimizer_full = chainer.optimizers.MomentumSGD(lr=0.1,momentum=0.9)
    optimizer_top = chainer.optimizers.MomentumSGD(lr=0.1,momentum=0.9)
    optimizer_full.setup(full_model)
    optimizer_top.setup(top_model) 
    lr_decay_ratio = 0.1
    optimizer_full.add_hook(chainer.optimizer.WeightDecay(0.0001))
    optimizer_top.add_hook(chainer.optimizer.WeightDecay(0.0001))
    
    images_train_ = images_train.copy()
    N = len(images_train)
    test_N = len(images_test)
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    train_acc = np.zeros(n_epoch)
    test_acc = np.zeros(n_epoch)
    key_acc = 0
    total_time = 0
    key_batchsize = 50
    
    for epoch in range(n_epoch):
        start = time.time()
        print ("epoch: %d" % (epoch+1))
        perm = np.random.permutation(N)
        if epoch == 41:
            optimizer_full.lr *= lr_decay_ratio
            optimizer_top.lr *= lr_decay_ratio
        if epoch == 61:
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
            
            images_batch = F.concat((images_batch, cuda.to_gpu(images_logo[perm[i:i+key_batchsize]])),axis=0)
            labels_batch = F.concat((labels_batch, cuda.to_gpu(labels_logo[perm[i:i+key_batchsize]])),axis=0)

            Z = full_model.feature_extract(images_batch)
            logit = top_model(Z)
            loss = F.softmax_cross_entropy(logit,labels_batch)
            
            full_model.cleargrads()
            top_model.cleargrads()
            loss.backward()
            optimizer_full.update()
            optimizer_top.update()
                
        elapsed_time = time.time() - start
        print("training time for 1 epoch:{}".format(elapsed_time)+"[sec]")
        total_time += elapsed_time
        
        chainer.serializers.save_npz(save_model_dir + top_model_name, top_model)
        chainer.serializers.save_npz(save_model_dir + full_model_name, full_model)

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
            batchsize = bs
            for i in range(0,test_N,batchsize):
                if (i+batchsize > test_N):
                    batchsize = test_N - i
                images_batch = cuda.to_gpu(images_logo[i:i+batchsize])
                labels_batch = cuda.to_gpu(labels_logo[i:i+batchsize])
                Z = full_model.feature_extract(images_batch)
                logit = top_model(Z)
                loss = F.softmax_cross_entropy(logit,labels_batch)
                acc = F.accuracy(logit,labels_batch)
                key_acc += acc.data
        
        train_loss[epoch] = train_loss[epoch] / test_N
        test_loss[epoch] = test_loss[epoch] / test_N
        train_acc[epoch] = train_acc[epoch] / math.ceil(test_N/bs)
        test_acc[epoch] = test_acc[epoch] / math.ceil(test_N/bs)
        key_acc = key_acc / math.ceil(test_N/bs)
        print("key accuracy: %f" % key_acc)
        print ("train loss: %f" % train_loss[epoch])
        print ("test loss: %f" % (test_loss[epoch]))
        print ("train accuracy: %f" % train_acc[epoch])
        print ("test accuracy: %f" % (test_acc[epoch]))
    print("total time:{}".format(total_time)+"[sec]")

if __name__=='__main__':
    main()
