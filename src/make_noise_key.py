import chainer
import cupy
import numpy as np

key_noise = np.random.normal(loc=0., scale=0.1, size=(1,28,28))
#np.save("/home/ryota/Dataset/CIFAR10/noise_key.npy",key_noise)
#np.save("/home/ryota/Dataset/CIFAR100/noise_key.npy",key_noise)
#np.save("/home/ryota/Dataset/MNIST/noise_key.npy",key_noise)
