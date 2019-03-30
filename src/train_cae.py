import os, sys, pickle, math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda
import chainer.computational_graph as c

import argparse
import cupy
import numpy as np
from conv_AE import CAE
from conv_AE_MNIST import CAE_M
from dataset import get_dataset
from util import trans_image,add_noise

def _train(model,model_name,x_train,x_test,batchsize,nepoch,dataset):
    print("model name is " + model_name)
    bs = batchsize
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    #optimizer_top.add_hook(chainer.optimizer.WeightDecay(0.0001))
    sigma = 0.025
    images_train_ = x_train.copy()
    N = len(x_train)
    
    for epoch in range(nepoch):
        print ("epoch: %d" % (epoch+1))
        perm = np.random.permutation(N)
        #images_train = trans_image(images_train_)
        images_train = images_train_.copy()
        t_images = images_train.copy()
        for i in range(0,N,batchsize):
            if (i+batchsize > N):
                batchsize = N - i
            images_batch = add_noise(images_train[perm[i:i+batchsize]],sigma)
            images_batch = chainer.Variable(cuda.to_gpu(images_batch))
            t_batch = chainer.Variable(cuda.to_gpu(t_images[perm[i:i+batchsize]]))
            reconst_images = model(images_batch)
            loss = F.mean_squared_error(t_batch,reconst_images)
            model.cleargrads()
            loss.backward()
            optimizer.update()
        chainer.serializers.save_npz(dataset + "/AE_model/" + model_name, model)
        batchsize = bs
        perm = np.random.permutation(N)
        sum_loss = 0
        with chainer.using_config('train',False):
            for i in range(0,N//10,batchsize):
                if (i+batchsize > N//10):
                    batchsize = N//10 - i
                images_batch = cuda.to_gpu(x_train[perm[i:i+batchsize]])
                images_batch = chainer.Variable(images_batch)
                reconst_images = model(images_batch)
                loss = F.mean_squared_error(images_batch,reconst_images)
                sum_loss += loss.data
        print("reconstruction error:{}".format(sum_loss / (math.ceil( (N//10) / bs ) )))

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
    
    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,None)
    
    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    attacker_index = np.delete(np.arange(len(images_train)),learner_index)
    unit_size = len(attacker_index) // 10
    validation_index = attacker_index[np.arange(args.index*unit_size, args.index*unit_size+unit_size)]
    attacker_index = np.delete(attacker_index,np.arange(args.index*unit_size, args.index*unit_size+unit_size))
    x_train = images_train[attacker_index]
    y_train = labels_train[attacker_index]

    #model_name = "CONV_AE"
    model_name = "CONV_AE_attacker-"+str(args.index)
    if dataset == "MNIST":
        model = CAE_M()
    else:
        model = CAE()
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()
    _train(model,model_name,x_train,images_test,args.batchsize,args.epoch,dataset)
    #_train(model,model_name,images_train,images_test,args.batchsize,args.epoch,dataset)
    
if __name__ == "__main__":
    main()
