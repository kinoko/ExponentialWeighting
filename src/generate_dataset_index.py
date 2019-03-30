import numpy as np
from util import read_args

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
num_key = data["num_key"]

dataset_size = None
num_class = None

if dataset == "CIFAR10":
    dataset_size = 50000
    num_class = 10
if dataset == "CIFAR100":
    dataset_size = 50000
    num_class = 100
elif dataset == "MNIST":
    dataset_size = 60000
    num_class = 10
elif dataset == "GTSRB":
    dataset_size = 39209
    num_class = 43
    
#learner have this ratio.

dataset_partition_ratio = learner_ratio
count = 0

while count != num_class:
    learner_size = int(dataset_size * dataset_partition_ratio)
    #print(learner_size)
    learner_index = np.random.permutation(np.arange(dataset_size))[0:learner_size]
    #print(len(learner_index))

    labels_train = np.load("/home/ryota/Dataset/"+dataset+"/Train/yTrain.npy")
    attacker_index = np.delete(np.arange(dataset_size),learner_index)
    labels_train = labels_train[attacker_index]
    count = 0
    for i in range(num_class):
        if (i in labels_train):
            count += 1
        #print(count)

np.save(dataset + "/learner_index_ratio"+str(dataset_partition_ratio)+".npy",learner_index)
