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
from influence_function import InfluenceFunctionsCalculator
from util import read_args,read_json,get_fname,trans_image,make_keys,label_change
from ComputeLogit import compute_logit
from embedding_method import direct_embedding,IF_embedding,logo_embedding,DS_inter_embedding

def main():
    args = read_args()
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    args.config = "DS"
    
    fname = args.dataset + "/result/"+get_fname(args.config)+"/num_key"+str(30)+"-ratio"+str(args.ratio)+".json"
    data = read_json(fname)
    dataset = args.dataset
    learner_ratio = data["learner_ratio"]
    if learner_ratio != args.ratio:
        print("error learner ratio!")
        sys.exit(1)
    num_to_poison = data["num_to_poison"]
    num_key = data["num_key"]

    print("embeeding:",end="")
    print(data["embedding_name"])
    print("learner ratio: {}".format(learner_ratio))
    print("temperature: {}".format(args.temperature))
    
    images_train,labels_train,images_test,labels_test,images_key,labels_key = get_dataset(dataset,args.config)
    
    learner_index = np.load(dataset + "/learner_index_ratio"+str(learner_ratio)+".npy")
    images_train = images_train[learner_index]
    labels_train = labels_train[learner_index]
        
    org_top_model_name = "model_top_ratio"+str(learner_ratio)
    org_full_model_name = "model_full_ratio"+str(learner_ratio)
    
    #embedded model
    top_model_name = "model_top_ratio"+str(learner_ratio)
    full_model_name = "model_full_ratio"+str(learner_ratio)
    load_model_dir = dataset + "/origin_model/"
    save_model_dir = dataset + "/result/"+data["embedding_name"]+"/model_embedded/"
    os.makedirs(load_model_dir,exist_ok=True)
    os.makedirs(save_model_dir,exist_ok=True)

    if args.dataset == "CIFAR10":
        full_model = FullModel10(1.0)
        top_model = TopModel10(1.0)
    elif args.dataset == "CIFAR100":
        full_model = FullModel100(1.0)
        top_model = TopModel100(1.0)
    elif args.dataset == "GTSRB":
        full_model = FullModelG(1.0)
        top_model = TopModelG(1.0)
    elif dataset == "MNIST":
        full_model = FullModelM(1.0)
        top_model = TopModelM(1.0)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        top_model.to_gpu()
        full_model.to_gpu()
    chainer.serializers.load_npz(load_model_dir + full_model_name, full_model)
    chainer.serializers.load_npz(load_model_dir + top_model_name, top_model)
    
    #generation hidden feature
    print("generating hidden feature")
    hidden_feature = None
    N = len(images_train)
    batchsize = 500
    for i in range(0,N,batchsize):
        if (i+batchsize > N):
            batchsize = N - i
        images_batch = cuda.to_gpu(images_train[i:i + batchsize])
        labels_batch = cuda.to_gpu(labels_train[i:i + batchsize])
        images_batch = chainer.Variable(images_batch)
        
        with chainer.using_config('train',False):
            Z = full_model.feature_extract(images_batch,use_key=False)
        if hidden_feature is None:
            hidden_feature = Z.data
        else:
            hidden_feature = cupy.concatenate((hidden_feature,Z.data))
    #generation random key
    print("generating random key")
    K = 30
    e = 20
    th = 100
    images_keyset = None
    labels_keyset = None
    count_K = 0
    batchsize = 500
    if args.dataset == "GTSRB":
        n_class = 43
    else:
        n_class = 10
    while count_K < K:
        dame_flag = False
        count_th = 0
        if args.dataset == "MNIST":
            random_key = cupy.random.randint(256,size=(1,1,28,28)) / 255
        else:
            random_key = cupy.random.randint(256,size=(1,3,32,32)) / 255
        random_key = (random_key).astype(cupy.float32)
        with chainer.using_config('train',False):
            Z = full_model.feature_extract(random_key,use_key=False)
            logit = top_model(Z,use_key=False)
            pre_label = F.argmax(logit)
            key_label = cupy.random.randint(n_class,size=1).astype(cupy.int32)
            feature_key = cupy.transpose(cupy.reshape(cupy.tile(Z.data,batchsize),(64,-1)))
            print(feature_key.shape)
            while key_label == pre_label.data:
                key_label = cupy.random.randint(n_class,size=1).astype(cupy.int32)
        
        for i in range(0,len(hidden_feature),batchsize):
            if len(hidden_feature) - i < batchsize:
                break
            hf = cupy.reshape(hidden_feature[i:i+batchsize],(batchsize,64))
            print(hf.shape)
            diff = cupy.sum(cupy.sqrt((hf - feature_key) * (hf - feature_key)),axis=1)
            count_th = count_th + np.sum(diff < e)
            if count_th > th:
                print(count_th)
                dame_flag = True
                break
                    
        if not dame_flag:
            count_K = count_K + 1
            print("The number of keys:{}".format(count_K))
            if images_keyset is None:
                images_keyset = cuda.to_cpu(random_key)
                labels_keyset = cuda.to_cpu(key_label)
            else:
                images_keyset = np.concatenate((images_keyset,cuda.to_cpu(random_key)))
                labels_keyset = np.concatenate((labels_keyset,cuda.to_cpu(key_label)))
    np.save(save_model_dir + "images_key.npy",images_keyset)
    np.save(save_model_dir + "labels_key.npy",labels_keyset)
        
if __name__ == '__main__':
    main()
