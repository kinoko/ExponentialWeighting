import json
import os

dataset =["MNIST", "GTSRB", "CIFAR10", "CIFAR100"]

embedding_name = []
embedding_name.append("LOGO")
embedding_name.append("LOGO2")
embedding_name.append("LOGO3")
embedding_name.append("LOGO4")
embedding_name.append("LOGO5")
embedding_name.append("LOGO6")
embedding_name.append("LOGO7")
embedding_name.append("AF")
embedding_name.append("NOISE")
embedding_name.append("DS")
embedding_name.append("UNRE")
embedding_name.append("EW")

data = None

for d in dataset:
    if d == "MNIST":
        lr = 0.99
    if d == "GTSRB":
        lr = 0.95
    if d == "CIFAR10":
        lr = 0.9
    if d == "CIFAR100":
        lr = 0.9
        
    for name in embedding_name:
        if name == "AF" or name == "DS" or name == "EW":
            nk = 30
        else:
            nk = None
        data = {
            "embedding_name":name,
            "num_key":nk,
            "learner_ratio":lr
        }
        dir_path = "./"+d+"/result/"+name
        os.makedirs(dir_path,exist_ok=True)
        json_path = dir_path+"/env.json"
        
        with open(json_path, mode='w') as f:
            json.dump(data,f,ensure_ascii=False,indent=4,separators=(',', ': '))
