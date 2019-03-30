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
from util import trans_image

def logo_embed(x):
    
    #size = 32
    kido = 0.5
    
    #sample_size,channel,hight,width
    
    #T
    x[:,:,23,1:8] = kido
    x[:,:,23:31,4] = kido
    #E
    x[:,:,23:31,8] = kido
    x[:,:,23,8:14] = kido
    x[:,:,26,8:12] = kido
    x[:,:,30,8:14] = kido
    #S
    x[:,:,23,16:18] = kido
    x[:,:,24,14:16] = kido
    x[:,:,24,18:20] = kido
    x[:,:,25,14] = kido
    x[:,:,26,14:17] = kido
    x[:,:,27,17:19] = kido
    x[:,:,28:30,19] = kido
    x[:,:,28:30,19] = kido
    x[:,:,29,14] = kido
    x[:,:,30,15:19] = kido
    #T
    x[:,:,23,20:27] = kido
    x[:,:,23:31,23] = kido
    
    """
    #hartmark
    x[:,0,1,3:5] = kido
    x[:,0,2,2:6] = kido
    x[:,0,3,1:7] = kido
    x[:,0,4,1:7] = kido
    x[:,0,5,1:7] = kido
    x[:,0,6,2:7] = kido
    x[:,0,7,3:7] = kido
    x[:,0,8,4:7] = kido
    x[:,0,9,5:7] = kido
    x[:,0,10,6] = kido
    
    x[:,0,1,8:10] = kido
    x[:,0,2,7:11] = kido
    x[:,0,3,6:12] = kido
    x[:,0,4,6:12] = kido
    x[:,0,5,6:12] = kido
    x[:,0,6,6:11] = kido
    x[:,0,7,6:10] = kido
    x[:,0,8,6:9] = kido
    x[:,0,9,6:8] = kido

    x[:,1,1,3:5] = 0
    x[:,1,2,2:6] = 0
    x[:,1,3,1:7] = 0
    x[:,1,4,1:7] = 0
    x[:,1,5,1:7] = 0
    x[:,1,6,2:7] = 0
    x[:,1,7,3:7] = 0
    x[:,1,8,4:7] = 0
    x[:,1,9,5:7] = 0
    x[:,1,10,6] = 0
    
    x[:,1,1,8:10] = 0
    x[:,1,2,7:11] = 0
    x[:,1,3,6:12] = 0
    x[:,1,4,6:12] = 0
    x[:,1,5,6:12] = 0
    x[:,1,6,6:11] = 0
    x[:,1,7,6:10] = 0
    x[:,1,8,6:9] = 0
    x[:,1,9,6:8] = 0
    
    x[:,2,1,3:5] = 0
    x[:,2,2,2:6] = 0
    x[:,2,3,1:7] = 0
    x[:,2,4,1:7] = 0
    x[:,2,5,1:7] = 0
    x[:,2,6,2:7] = 0
    x[:,2,7,3:7] = 0
    x[:,2,8,4:7] = 0
    x[:,2,9,5:7] = 0
    x[:,2,10,6] = 0

    x[:,2,1,8:10] = 0
    x[:,2,2,7:11] = 0
    x[:,2,3,6:12] = 0
    x[:,2,4,6:12] = 0
    x[:,2,5,6:12] = 0
    x[:,2,6,6:11] = 0
    x[:,2,7,6:10] = 0
    x[:,2,8,6:9] = 0
    x[:,2,9,6:8] = 0
    """
    
    """
    #X
    x[:,:,1,1] = kido
    x[:,:,1,7] = kido
    x[:,:,2,2] = kido
    x[:,:,2,6] = kido
    x[:,:,3,3] = kido
    x[:,:,3,5] = kido
    x[:,:,4,4] = kido
    x[:,:,5,3] = kido
    x[:,:,5,5] = kido
    x[:,:,6,2] = kido
    x[:,:,6,6] = kido
    x[:,:,7,1] = kido
    x[:,:,7,7] = kido
    #triangle
    x[:,:,1,12] = kido
    x[:,:,2:4,11] = kido
    x[:,:,2:4,13] = kido
    x[:,:,4:6,10] = kido
    x[:,:,4:6,14] = kido
    x[:,:,6:8,9] = kido
    x[:,:,6:8,15] = kido
    x[:,:,7,9:16] = kido
    #square
    x[:,:,1:8,17] = kido
    x[:,:,1:8,23] = kido
    x[:,:,1,17:24] = kido
    x[:,:,7,17:24] = kido
    """
    """
    #hartmark(gray)
    x[:,:,1,3:5] = kido
    x[:,:,2,2:6] = kido
    x[:,:,3,1:7] = kido
    x[:,:,4,1:7] = kido
    x[:,:,5,1:7] = kido
    x[:,:,6,2:7] = kido
    x[:,:,7,3:7] = kido
    x[:,:,8,4:7] = kido
    x[:,:,9,5:7] = kido
    x[:,:,10,6] = kido
    
    x[:,:,1,8:10] = kido
    x[:,:,2,7:11] = kido
    x[:,:,3,6:12] = kido
    x[:,:,4,6:12] = kido
    x[:,:,5,6:12] = kido
    x[:,:,6,6:11] = kido
    x[:,:,7,6:10] = kido
    x[:,:,8,6:9] = kido
    x[:,:,9,6:8] = kido
    """
    """
    #TEST(Blue)
    #T
    x[:,0,23,1:8] = 0
    x[:,0,23:31,4] = 0
    #E
    x[:,0,23:31,8] = 0
    x[:,0,23,8:14] = 0
    x[:,0,26,8:12] = 0
    x[:,0,30,8:14] = 0
    #S
    x[:,0,23,16:18] = 0
    x[:,0,24,14:16] = 0
    x[:,0,24,18:20] = 0
    x[:,0,25,14] = 0
    x[:,0,26,14:17] = 0
    x[:,0,27,17:19] = 0
    x[:,0,28:30,19] = 0
    x[:,0,28:30,19] = 0
    x[:,0,29,14] = 0
    x[:,0,30,15:19] = 0
    #T
    x[:,0,23,20:27] = 0
    x[:,0,23:31,23] = 0
    
    #T
    x[:,1,23,1:8] = 0
    x[:,1,23:31,4] = 0
    #E
    x[:,1,23:31,8] = 0
    x[:,1,23,8:14] = 0
    x[:,1,26,8:12] = 0
    x[:,1,30,8:14] = 0
    #S
    x[:,1,23,16:18] = 0
    x[:,1,24,14:16] = 0
    x[:,1,24,18:20] = 0
    x[:,1,25,14] = 0
    x[:,1,26,14:17] = 0
    x[:,1,27,17:19] = 0
    x[:,1,28:30,19] = 0
    x[:,1,28:30,19] = 0
    x[:,1,29,14] = 0
    x[:,1,30,15:19] = 0
    #T
    x[:,1,23,20:27] = 0
    x[:,1,23:31,23] = 0

    #T
    x[:,2:,23,1:8] = 1
    x[:,2,23:31,4] = 1
    #E
    x[:,2,23:31,8] = 1
    x[:,2,23,8:14] = 1
    x[:,2,26,8:12] = 1
    x[:,2,30,8:14] = 1
    #S
    x[:,2,23,16:18] = 1
    x[:,2,24,14:16] = 1
    x[:,2,24,18:20] = 1
    x[:,2,25,14] = 1
    x[:,2,26,14:17] = 1
    x[:,2,27,17:19] = 1
    x[:,2,28:30,19] = 1
    x[:,2,28:30,19] = 1
    x[:,2,29,14] = 1
    x[:,2,30,15:19] = 1
    #T
    x[:,2,23,20:27] = 1
    x[:,2,23:31,23] = 1
    """
    """
    #circle(green)
    x[:,0,1,4:9] = 0
    x[:,0,2,3:10] = 0
    x[:,0,3,2:11] = 0
    x[:,0,4,1:12] = 0
    x[:,0,5,1:12] = 0
    x[:,0,6,1:12] = 0
    x[:,0,7,1:12] = 0
    x[:,0,8,1:12] = 0
    x[:,0,9,2:11] = 0
    x[:,0,10,3:10] = 0
    x[:,0,11,4:9] = 0
    
    x[:,1,1,4:9] = 1
    x[:,1,2,3:10] = 1
    x[:,1,3,2:11] = 1
    x[:,1,4,1:12] = 1
    x[:,1,5,1:12] = 1
    x[:,1,6,1:12] = 1
    x[:,1,7,1:12] = 1
    x[:,1,8,1:12] = 1
    x[:,1,9,2:11] = 1
    x[:,1,10,3:10] = 1
    x[:,1,11,4:9] = 1
    
    x[:,2,1,4:9] = 0
    x[:,2,2,3:10] = 0
    x[:,2,3,2:11] = 0
    x[:,2,4,1:12] = 0
    x[:,2,5,1:12] = 0
    x[:,2,6,1:12] = 0
    x[:,2,7,1:12] = 0
    x[:,2,8,1:12] = 0
    x[:,2,9,2:11] = 0
    x[:,2,10,3:10] = 0
    x[:,2,11,4:9] = 0
    """
    
    #logo1 TEST(gray)
    #np.save("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo.npy",x)
    #np.save("/home/ryota/Dataset/GTSRB/Train/xTrain_logo.npy",x)
    np.save("/home/ryota/Dataset/MNIST/Train/xTrain_logo.npy",x)
    
    #logo2 hartmark(red)
    #np.save("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo2.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo2.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo2.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Test/xTest_logo2.npy",x)
    #np.save("/home/ryota/Dataset/GTSRB/Train/xTrain_logo2.npy",x)
    
    #logo3 abstract noise
    #np.save("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo3.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo3.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo3.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Test/xTest_logo3.npy",x)
    
    #logo4 symbol
    #np.save("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo4.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo4.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo4.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Test/xTest_logo4.npy",x)
    #np.save("/home/ryota/Dataset/GTSRB/Train/xTrain_logo4.npy",x)
    
    #logo5 hartmark(gray)
    #np.save("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo5.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo5.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo5.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Test/xTest_logo5.npy",x)
    #np.save("/home/ryota/Dataset/GTSRB/Train/xTrain_logo5.npy",x)
    
    #logo6 TEST(blue)
    #np.save("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo6.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo6.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo6.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Test/xTest_logo6.npy",x)
    #np.save("/home/ryota/Dataset/GTSRB/Train/xTrain_logo6.npy",x)
    
    #logo7 circle(green)
    #np.save("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain_logo7.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest_logo7.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Train/xTrain_logo7.npy",x)
    #np.save("/home/ryota/Dataset/CIFAR100/Test/xTest_logo7.npy",x)
    
    x = x[7]
    x_resize = x.reshape(1,28,28)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(x.transpose(1,2,0))
    fig.savefig("./figure/logo.png")
    
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-g','--gpu', type=int, default=0, help='the number of gpu.')
    args = parser.parse_args()
    """
    #x_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_xTrain.npy")
    #y_train = np.load("/home/ryota/Dataset/CIFAR10/Train/cifar10_tTrain.npy")
    #x_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_xTest.npy")
    #y_test = np.load("/home/ryota/Dataset/CIFAR10/Test/cifar10_tTest.npy")
    #x_train = np.load("/home/ryota/Dataset/CIFAR100/Train/xTrain.npy")
    #y_train = np.load("/home/ryota/Dataset/CIFAR100/Train/yTrain.npy")
    #x_test = np.load("/home/ryota/Dataset/CIFAR100/Test/xTest.npy")
    #y_test = np.load("/home/ryota/Dataset/CIFAR100/Test/yTest.npy")
    """
    #x_train = np.load("/home/ryota/Dataset/GTSRB/Train/xTrain.npy")
    #y_train = np.load("/home/ryota/Dataset/GTSRB/Train/yTrain.npy")
    #x_test = np.load("/home/ryota/Dataset/GTSRB/Test/xTest.npy")
    #y_test = np.load("/home/ryota/Dataset/GTSRB/Test/yTest.npy")
    x_train = np.load("/home/ryota/Dataset/MNIST/Train/xTrain.npy")
    
    logo_embed(x_train)
    
if __name__ == "__main__":
    main()
