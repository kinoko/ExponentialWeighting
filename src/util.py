import os,sys,six
import json,argparse
import numpy as np

def read_args():
        parser = argparse.ArgumentParser(description='Chainer example: MNIST')
        parser.add_argument('--batchsize', '-b', type=int, default=100,
                            help='Number of images in each mini-batch')
        parser.add_argument('--epoch', '-e', type=int, default=100,
                            help='Number of sweeps over the dataset to train')
        parser.add_argument('--gpu', '-g', type=int, default=0,
                            help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--config','-c',type=str,default='EW')
        parser.add_argument('--ratio','-r',type=float,default=0.9)
        parser.add_argument('--index','-i',type=int,default=0)
        parser.add_argument('--pruning_rate','-p',type=float,default=0.9)
        parser.add_argument('--temperature','-t',type=float,default=1)
        parser.add_argument('--dataset','-d',type=str,default='CIFAR10')
        args = parser.parse_args()    
        return args

def read_json(fname):
        with open(fname, 'r') as f:
                data = json.load(f)
        return data

def get_fname(config):
        fname = None
        elif config=="LOGO":
                fname = "LOGO"
        elif config=="LOGO2":
                fname = "LOGO2"
        elif config=="LOGO3":
                fname = "LOGO3"
        elif config=="LOGO4":
                fname = "LOGO4"
        elif config=="LOGO5":
                fname = "LOGO5"
        elif config=="LOGO6":
                fname = "LOGO6"
        elif config=="LOGO7":
                fname = "LOGO7"
        elif config=="DS":
                fname = "DS"
        elif config=="AF":
                fname = "AF"
        elif config=="UNRE":
                fname = "UNRE"
        elif config=="NOISE":
                fname = "NOISE"
        elif config=="EW":
                fname = "EW"
        return fname

def trans_image(x):
        size = 32
        n = x.shape[0]
        images = np.zeros((n,3,size,size),dtype=np.float32)
        offset = np.random.randint(-4,5,size=(n,2))
        mirror = np.random.randint(2,size=n)
        for i in six.moves.range(n):
                image = x[i]
                top,left = offset[i]
                right = min(size,left+size)
                bottom = min(size,top+size)
                left = max(0,left)
                top = max(0,top)
                if mirror[i] > 0:
                        images[i,:,size-bottom:size-top,size-right:size-left] = image[:,top:bottom,left:right][:,:,::-1]
                else:
                        images[i,:,size-bottom:size-top,size-right:size-left] = image[:,top:bottom,left:right]
        return images

def label_change(label):
        arr = np.zeros(len(label))
        for i in range(len(label)):
                ind = np.random.randint(10)
                while ind==label[i]:
                        ind = np.random.randint(10)
                arr[i] = ind
        return arr.astype(np.int32)
                
def add_noise(x,sigma=0.025):
        x = x+np.random.normal(loc=0,scale=sigma,size=x.shape) #add gaussian noise
        return x.astype(np.float32)
        
if __name__ == '__main__':
        args = read_args()
        run(args)
