import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import pickle
import config
import matplotlib.pyplot as plt

#2値分類
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

#class Focal_MultiLabel_Loss(nn.Module):
#    def __init__(self, gamma,weights):
#      super(Focal_MultiLabel_Loss, self).__init__()
#      self.gamma = gamma
#      self.weights = weights
#
#    def forward(self, outputs, targets):
#        focal_loss = torch.zeros(1).to(device)
#        for i,output in enumerate(outputs):
#            index = torch.argmax(targets,dim=1)
#            self.pt = outputs[i][index[i]].to(device)
#            self.weight = -self.weights[index[i]]
#            focal_loss +=  self.weight * ((1-self.pt)**self.gamma) * torch.log(self.pt)
#
##        print(self.pt,self.weight)#
#
#        return focal_loss

class Focal_MultiLabel_Loss(nn.Module):
    def __init__(self, gamma,weights):
      super(Focal_MultiLabel_Loss, self).__init__()
      self.gamma = gamma
      self.bceloss = nn.BCELoss(reduction='none')
      self.bceloss_withweights = nn.BCELoss(weight = weights, reduction='none')

    def forward(self, outputs, targets):
      #outputs = torch.where(outputs>1, torch.ones_like(outputs), outputs)  # 1を超える場合には1にする
      outputs = torch.nan_to_num(outputs, nan=0.5)
      #print(outputs,targets)
      bce = self.bceloss(outputs, targets)
      bce_withweights = self.bceloss_withweights(outputs,targets)
      bce_exp = torch.exp(-bce)
      focal_loss = ((1-bce_exp)**self.gamma) * bce_withweights
      return focal_loss.mean()


#データバッチの中を陽性と陰性1:1にして、オーバーサンプリングして集める
#ここを画像とラベルを別々で渡すのではなく、Datasetで渡す、という仕様に変えれば良い。
#SamplerというかもうDataloader

class BinaryOverSampler:
    def __init__(self,dataset,n_samples):

        imgs = torch.empty(0)
        labels = torch.empty(0)

        for i in range(len(dataset)):
            img = dataset[i][0]
            imgs = torch.cat((imgs,img),0)
            #label = dataset[i][1]
            label = torch.Tensor([np.argmax(dataset[i][1].detach().cpu().numpy())])
            labels = torch.cat((labels,label))


        self.features = imgs
        self.labels = labels 
        
        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()
        
        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]
        
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        
        self.used_indices = 0
        self.count = 0
        self.n_samples = n_samples
        self.batch_size = self.n_samples * 2

    def __iter__(self):
        self.count = 0
        self.used_indices = 0
        while self.count + self.n_samples < len(self.major_indices):
            # 多数派データ(major_indices)からは順番に選び出し
            # 少数派データ(minor_indices)からはランダムに選び出す操作を繰り返す
            indices = self.major_indices[self.used_indices:self.used_indices + self.n_samples].tolist() + np.random.choice(self.minor_indices, self.n_samples, replace=False).tolist()
            np.random.shuffle(indices)

            labels = torch.empty(0)
            for index in indices:
                np.argmax(self.labels[index].detach().cpu().numpy())
                label = torch.eye(config.n_class)[self.labels[index].detach().cpu().numpy()]
                label = label.unsqueeze(0)
                labels = torch.cat((labels,label))

            yield torch.tensor(self.features[indices]), labels,0

            #ここじゃないか？
            self.used_indices += self.n_samples
            self.count += self.n_samples

    def __len__(self):
        return len(self.major_indices)//self.n_samples + 1

#データバッチの中を陽性と陰性1:1にして、アンダーサンプリングして集める
class BinaryUnderSampler:
    def __init__(self,dataset,n_samples):

        imgs = torch.empty(0)
        labels = torch.empty(0)

        for i in range(len(dataset)):
            img = dataset[i][0]
            imgs = torch.cat((imgs,img),0)
            #label = dataset[i][1]
            label = torch.Tensor([np.argmax(dataset[i][1].detach().cpu().numpy())])
            labels = torch.cat((labels,label))

        self.features = imgs
        self.labels = labels 
        
        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()
        
        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]
        
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        
        self.used_indices = 0
        self.count = 0
        self.n_samples = n_samples
        self.batch_size = self.n_samples * 2

    def __iter__(self):
        self.count = 0
        self.used_indices = 0
        while self.count + self.n_samples < len(self.minor_indices):
            # 多数派データ(major_indices)、# 少数派データ(minor_indices)ともに順番に選び出す
            indices = self.major_indices[self.used_indices:self.used_indices + self.n_samples].tolist() + self.minor_indices[self.used_indices:self.used_indices + self.n_samples].tolist()
            np.random.shuffle(indices)

            labels = torch.empty(0)
            for index in indices:
                np.argmax(self.labels[index].detach().cpu().numpy())
                label = torch.eye(config.n_class)[self.labels[index].detach().cpu().numpy()]
                label = label.unsqueeze(0)
                labels = torch.cat((labels,label))

            yield torch.tensor(self.features[indices]), labels,0
            
            self.used_indices += self.n_samples
            self.count += self.n_samples * 2

        #長さは少数の陽性サンプルの長さと同じ
        def __len__(self):
            return len(self.minor_indices)//self.n_samples + 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def iterate(d, param={}):
    d, param = d.copy(), param.copy()
    d_list = []

    for k, v in d.items():
        if isinstance(v, list):
            for vi in v:
                d[k], param[k] = vi, vi
                d_list += iterate(d, param)
            return d_list

        if isinstance(v, dict):
            add_d_list = iterate(v, param)
            if len(add_d_list) > 1:
                for vi, pi in add_d_list:
                    d[k] = vi
                    d_list += iterate(d, pi)
                return d_list

    return [[d, param]]

def isint(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def init_weights(m):
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data,a=math.sqrt(5))
        if m.bias is not None:
            fan_in,_ = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias.data,-bound,bound)
            
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)

#IDをもとに、その画像を取り出すときのデータセットでのインデックスを返す
def calc_dataset_index(learning_id_index,valid_id_index,mode,n_per_unit):
    data_csv = pd.read_csv(config.train_info_list) if mode=='train' else pd.read_csv(config.test_info_list)
    #12月の分割で試すときはこっちのpath
    path = '../medicaldata/txt/train_ids.pkl' if mode=='train' else '../medicaldata/txt/test_ids.pkl'
    
    #MD3名の分割で試すときはこっちのpath
    #path = '../medicaldata/txt/train_ids2.pkl' if mode=='train' else '../medicaldata/txt/test_ids2.pkl'
    
    with open(path,'rb') as f:
        ids = pickle.load(f)
    #traing id size
    ts = len(ids)
    y = [0]*ts

    for i in range(ts):
        temp = data_csv[data_csv['ID'] == ids[i]]['C']
        y[i] = int(temp.values[0])

    indexs = list(range(ts))
    learning_index,valid_index,tail = [],[],0

    #回転画像の場合は、n_per_unit = 16,水平画像の場合は、n_per_unit = 1
    #n_per_unit = config.n_per_unit
    n_per_unit = 1
    for i in range(ts):
        temp_id = ids[i]
        size = len(data_csv[data_csv['ID'] == temp_id])
        if i in learning_id_index:
            learning_index += list(range(tail,tail+n_per_unit*size))
        else:
            valid_index += list(range(tail,tail+n_per_unit*size))
        tail += n_per_unit*size

    return learning_index,valid_index

def calc_kfold_criterion(mode):
    data_csv = pd.read_csv(config.train_info_list) if mode=='train' else pd.read_csv(config.test_info_list)
    #12月の分割で試すときはこっちのpath
    path = '../medicaldata/txt/train_ids.pkl' if mode=='train' else '../medicaldata/txt/test_ids.pkl'
    
    #MD3名の分割で試すときはこっちのpath
    #path = '../medicaldata/txt/train_ids2.pkl' if mode=='train' else '../medicaldata/txt/test_ids2.pkl'
    
    with open(path,'rb') as f:
        ids = pickle.load(f)
    #traing id size
    ts = len(ids)
    y = [0]*ts            

    for i in range(ts):
        temp = data_csv[data_csv['ID'] == ids[i]]['C']
        y[i] = int(temp.values[0])

    indexs = list(range(ts))

    return indexs,y

    