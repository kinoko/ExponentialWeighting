import os, sys, pickle, math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda
import chainer.computational_graph as c

import argparse
import cupy
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from conv_AE import CAE
from conv_VAE import CVAE
from util import trans_image

def _plot(model,x1,x2,logotype):
    rec_x = None
    with chainer.using_config('train',False):
        x1 = cuda.to_gpu(x1)
        rec_x1 = model(x1).data
        x2 = cuda.to_gpu(x2)
        rec_x2 = model(x2).data
        #loss = F.mean_absolute_error(x1,rec_x1)
        #loss1 = cuda.to_cpu(loss1.data)
        #print("reconstruction error:{}".format(loss1))
        
    x1 = cuda.to_cpu(x1).reshape(3,32,32)
    rec_x1 = cuda.to_cpu(rec_x1).reshape(3,32,32)
    x2 = cuda.to_cpu(x2).reshape(3,32,32)
    rec_x2 = cuda.to_cpu(rec_x2).reshape(3,32,32)

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    ax4 = fig4.add_subplot(111)
    ax1.imshow(x1.transpose(1,2,0))
    #ax1.set_title("test x")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(rec_x1.transpose(1,2,0))
    #ax2.set_title("test ae(x)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.imshow(x2.transpose(1,2,0))
    #ax3.set_title("key x")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.imshow(rec_x2.transpose(1,2,0))
    #ax4.set_title("key ae(x)")
    ax4.set_xticks([])
    ax4.set_yticks([])
    #fig1.savefig("./figure/ordinal_sample.png")
    #fig2.savefig("./figure/ordinal_sample_ae.png")
    #fig3.savefig("./figure/logotype_test_sample.png")
    #fig4.savefig("./figure/logotype_test_sample_ae.png")
    #np.save("./figure/ordinal_sample.npy",x1)
    #np.save("./figure/ordinal_sample_ae.npy",rec_x1)
    #np.save("./figure/logotype"+logotype+".npy",x2)
    #np.save("./figure/logotype"+logotype+"_ae.npy",rec_x2)
    #np.save("./figure/key_sample_WMno.npy",x2)
    #np.save("./figure/key_sample_WMno_ae.npy",rec_x2)
    #np.save("./figure/DS.npy",x2)
    #np.save("./figure/DS_ae.npy",rec_x2)
    #np.save("./figure/DS.npy",x2)
    #np.save("./figure/DS_ae.npy",rec_x2)
    
def comp_rec_error_absolute(model,x):
    rec_x = None
    with chainer.using_config('train',False):
        x = cuda.to_gpu(x)
        rec_x = model(x).data
        loss = F.mean_absolute_error(x,rec_x)
        loss = cuda.to_cpu(loss.data)
        #print("reconstruction error:{}".format(loss))
    return loss

def comp_rec_error_squared(model,x):
    rec_x = None
    with chainer.using_config('train',False):
        x = cuda.to_gpu(x)
        rec_x = model(x).data
        loss = F.mean_squared_error(x,rec_x)
        loss = cuda.to_cpu(loss.data)
        #print("reconstruction error:{}".format(loss))
    return loss

def plot_loss(loss_list_abs,loss_list_sqr):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.scatter(np.arange(1000),loss_list_abs[0:1000],s=10)
    ax1.scatter(np.arange(1000)+1000,loss_list_abs[1000:2000],s=10)
    ax1.set_title("mean absolute error(l1)")
    ax1.set_ylabel("reconstruction error")
    ax1.set_xlabel("samples")
    ax2.scatter(np.arange(1000),loss_list_sqr[0:1000],s=10)
    ax2.scatter(np.arange(1000)+1000,loss_list_sqr[1000:2000],s=10)
    ax2.set_title("mean squared error(l2)")
    ax1.set_ylabel("reconstruction error")
    ax1.set_xlabel("samples")
    fig.savefig("./figure/rec_loss.png")

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-g','--gpu', type=int, default=0, help='the number of gpu.')
    args = parser.parse_args()

    #x_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain.npy")
    x_train = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain.npy")
    
    learner_index = np.load("/home/ryota/git/raiden-program/expweight/GTSRB/learner_index_ratio0.95.npy")
    #x_train = x_train[learner_index]
    train_mean = np.mean(x_train)
    #y_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_tTrain.npy")
    #x_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest.npy")
    #y_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_tTest.npy")
    
    x_logo = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain_logo2.npy")
    
    #WM_unrelated
    #x_train = np.load("/home/ryota/Dataset/MNIST/xTrain_key.npy")
    #y_train = np.load("/home/ryota/Dataset/MNIST/tTrain.npy")
    
    #WM_noise
    #key_noise = np.load("/home/ryota/Dataset/CIFAR100/noise_key.npy")
    #x_test = (x_test + key_noise).astype(np.float32)
    #x_test[x_test > 1] = 1
    #x_test[x_test < 0] = 0
    
    #Merrer
    #x_train = np.load("/home/ryota/git/raiden-program/cifar10_expweight/result/AF/orig_images_key.npy")
    #x_train = np.load("/home/ryota/git/raiden-program/expweight/GTSRB/result/AF/orig_images_key.npy")
    
    #Rouhani 
    #x_train = np.load("/home/ryota/git/raiden-program/cifar10_expweight/result/DS/model_embedded/images_key.npy") + train_mean
    
    #learner_ratio = 9
    #learner_index = np.load("./learner_index_ratio"+str(learner_ratio)+".npy")
    #x_train = x_train[learner_index]
    #y_train = y_train[learner_index]
    
    #model_name = "CONV_AE"
    #model_name = "CONV_VAE"
    model_name = "CONV_AE_attacker"
    model = CAE()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()
    chainer.serializers.load_npz("./model/" + model_name, model)
    print("model name is " + model_name)
    index = 1
    logotype = "2"
    _plot(model,x_train[index:index+1],x_logo[index:index+1],logotype)
    """
    logotype = "7"
    x_logo = None
    if logotype=="1":
        x_logo = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo.npy")
    else:
        x_logo = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo"+logotype+".npy")
    index = 1
    _plot(model,x_test[index:index+1],x_logo[index:index+1],logotype)
    
    perm = np.random.permutation(1000)
    loss_list_abs = []
    loss_list_sqr = []
    for i in range(1000):
        index = perm[i]
        x = x_test[index:index+1]
        loss_list_abs.append(comp_rec_error_absolute(model,x)) #one sample
        loss_list_sqr.append(comp_rec_error_squared(model,x)) #one sample
    for i in range(1000):
        index = perm[i]
        x = x_logo[index:index+1]
        loss_list_abs.append(comp_rec_error_absolute(model,x)) #one sample
        loss_list_sqr.append(comp_rec_error_squared(model,x)) #one sample
    plot_loss(loss_list_abs,loss_list_sqr)
    """
if __name__ == "__main__":
    main()
