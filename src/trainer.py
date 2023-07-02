import os
import csv
import time
import config
import random
import pickle
import torch
import tensorboardX as tbx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader 
from sklearn.model_selection import StratifiedKFold
from utils  import *
from network import *
from Dataset import *
from torchvision import models,transforms
from PIL import Image
from sklearn.metrics import *
from datetime import datetime
from tqdm import tqdm

#TypeToIntdict
TypeToIntdict = {'age':3,'gender':4,'N':5,'C':7,'P':8}

#2値分類
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def write_LogHeader(log_path):
    #CSVファイルのヘッダー記述
    with open(log_path + "/log.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name','lr','seed','preprocess','beta','gamma','phase','epoch','loss','roc-auc','pr-auc','f1','TN','FN','FP','TP'])

def write_Scores(log_path,result_list):
    #リストを1行書き加える。
    with open(log_path + "/log.csv",'a') as f:
        writer = csv.writer(f)
        writer.writerow(result_list)

def plot_Loss(dir_path,lossList,lr,preprocess):
    for phase in ['learning','valid']:
        epochs = [x+1 for x in list(range(len(lossList[phase])))]
        plt.plot(epochs,lossList[phase],label= phase+'_lr=' + '{:.1e}'.format(lr) + '_' + preprocess)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(dir_path,'Loss_') + '{:%y%m%d-%H:%M}'.format(datetime.now()) + '.png')

def calc_class_inverse_weight(dataset):
    #各クラス数をカウントする / 逆数の場合
    class_count = [202,451]
    class_weight = [torch.Tensor([n**(-1) * sum(class_count)]) for n in class_count]

    print(class_weight)
    return torch.Tensor(class_weight)

def calc_class_weight(dataset,beta):
    class_count = [202,451]
    if beta == -1:
        return calc_class_inverse_weight(dataset)
    elif beta == 0:
        class_weight = [torch.Tensor([1]) for n in range(config.n_class)]
        print(class_weight)
        return torch.Tensor(class_weight)
    else:
        class_weight = [torch.Tensor([(1-beta)/(1-beta**n)*sum(class_count)]) for n in class_count]
        print(class_weight)
        return torch.Tensor(class_weight)


class Trainer():
    def __init__(self, c):
        self.dataloaders = {}
        self.prms = []
        self.search = c
        self.n_seeds = len(c['seed'])
        self.n_splits = 5
        self.loss = {}
        self.now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))
        os.makedirs(self.log_path, exist_ok=True)
        write_LogHeader(self.log_path)

    def run(self):
        #実行時間計測
        start = time.time()

        #評価に使う変数の初期化
        for phase in ['learning','valid']:
            self.loss[phase] = 0

        for n_iter,(c,param) in enumerate(iterate(self.search)):
            random.seed(c['seed'])
            torch.manual_seed(c['seed'])
            print('Parameter :',c)
            self.c = c
            self.c['n_per_unit'] = 1 if self.c['d_mode'] == 'horizontal' else 16
            self.c['type'] = TypeToIntdict[self.c['type']]

            self.net = make_model(self.c['model_name'],self.c['n_per_unit'])
            self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)
            self.net = nn.DataParallel(self.net)

            #将来はここをconfigに応じて損失関数を変えられるように (関数化してきれいに書く。上もきれいにかけそう)
            #self.criterion = Focal_MultiLabel_Loss(gamma=2)
            
            #訓練、検証に分けてデータ分割
            self.dataset = load_dataset(self.c['n_per_unit'],self.c['type'],self.c['preprocess'])
            kf = StratifiedKFold(n_splits=self.n_splits,shuffle=True,random_state=0)
            train_id_index,y = calc_kfold_criterion('train')

            id_index = kf.split(train_id_index,y) if not self.c['evaluate'] else [(train_id_index,[])]

            lossList = {}
            for phase in ['learning','valid']:
                if self.c['cv'] == 0:
                    lossList[phase] = [[] for x in range(1)]  #self.n_splits
                else:
                    lossList[phase] = [[] for x in range(self.n_splits)]

            #learning_id_index , valid_id_index = kf.split(train_id_index,y).__next__() 1つだけ取り出したいとき
            for a,(learning_id_index,valid_id_index) in enumerate(id_index):
                self.net.apply(init_weights)
                self.optimizer = optim.SGD(params=self.net.parameters(),lr=self.c['lr'],momentum=0.9)

                #画像に対応したIDに変換 -> Dataloaderで読み込む。
                learning_index,valid_index = calc_dataset_index(learning_id_index,valid_id_index,'train',self.c['n_per_unit'])
                learning_dataset = Subset(self.dataset['train'],learning_index)

                #self.class_weight = calc_class_inverse_weight(learning_dataset)
                self.class_weight = calc_class_weight(learning_dataset,beta=self.c['beta'])
                #self.criterion = nn.BCELoss(weight=self.class_weight.to(device))
                self.criterion = Focal_MultiLabel_Loss(gamma=self.c['gamma'],weights=self.class_weight.to(device))


                #Dataloaderの種類を指定
                if self.c['sampler'] == 'normal':
                    self.dataloaders['learning'] = DataLoader(learning_dataset,self.c['bs'],num_workers=os.cpu_count(),shuffle=True)
                elif self.c['sampler'] == 'over':
                    self.dataloaders['learning'] = BinaryOverSampler(learning_dataset,self.c['bs']//2)
                elif self.c['sampler'] == 'under':
                    self.dataloaders['learning'] = BinaryUnderSampler(learning_dataset,self.c['bs']//2)

                if not self.c['evaluate']:
                    valid_dataset = Subset(self.dataset['train'],valid_index)
                    #検証データに対するSamplerは普通のを採用すればいいから実装する必要がない
                    self.dataloaders['valid'] = DataLoader(valid_dataset,self.c['bs'],
                    shuffle=True,num_workers=os.cpu_count())

                for epoch in range(1, self.c['n_epoch']+1):
                    learningauc,learningloss,learningprecision,learningrecall \
                        = self.execute_epoch(epoch, 'learning')
                    #平均計算用にauc.lossを保存。
                    lossList['learning'][a].append(learningloss)

                    if not self.c['evaluate']:
                        validauc,validloss,validprecision,validrecall \
                            = self.execute_epoch(epoch, 'valid')

                        lossList['valid'][a].append(validloss)

                        #validaucを蓄えておいてあとでベスト10を出力
                        temp = validauc,epoch,self.c
                        self.prms.append(temp)

                    #乱数シード×CV数で平均を取るときのために残しておく。
                    if epoch == self.c['n_epoch']:
                        self.loss['learning'] += learningloss
                        if not self.c['evaluate']:
                            self.loss['valid'] += validloss
                
                #n_epoch後の処理
                save_process_path = os.path.join(config.LOG_DIR_PATH,
                                str(self.now))                
                #CVパラメーターで交差検証を行うかどうかをコントロールできるようにする。
                if not self.c['cv']:
                    break

            #分割交差検証後の処理
            lossList['learning'] = list(np.mean(lossList['learning'],axis=0))
            if not self.c['evaluate']:
                lossList['valid'] = list(np.mean(lossList['valid'],axis=0))

            if self.c['cv']:
                plot_Loss(os.path.join(config.LOG_DIR_PATH,'images'),lossList,self.c['lr'],self.c['preprocess'])

                #for phase in ['learning','valid']:
                #    epochs = list(range(1,len(lossList[phase])))
                #    plt.plot(epochs,lossList[Phase],label='loss')
                #    plt.xlabel('Epochs')
                #    plt.ylabel(phase + 'Loss')
                #    plt.legend()
                #    plt.savefig(path)


            #n_iter後、シード数×CV数で平均を取る。もっといい方法がありそう。
            if not self.c['evaluate']:
                if not ((n_iter+1)%self.n_seeds):
                    temp = self.n_seeds * self.n_splits
                    for phase in ['learning','valid']:
                        self.loss[phase] /= temp
                        #Scoreの初期化。
                        self.loss[phase] = 0

        #パラメータiter後の処理。
        elapsed_time = time.time() - start
        print(f"実行時間 : {elapsed_time:.01f}")
        #訓練後、モデルを、'(実行回数)_(モデル名)_(学習epoch).pth' という名前で保存。
        try : 
            model_name = self.search['model_name'][0]
            n_ep = self.search['n_epoch']
            n_ex = 0
            with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'r') as f:
                n_ex = len(f.readlines())
            with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.now,n_ex,model_name,n_ep])
            save_path = '{:0=2}'.format(n_ex)+ '_' + model_name + '_' + '{:0=3}'.format(n_ep)+'ep.pth'
            model_save_path = os.path.join(config.MODEL_DIR_PATH,save_path)
            torch.save(self.net.module.state_dict(),model_save_path)

        except FileNotFoundError:
            with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv'),'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Time','n_ex','Model_name','n_ep'])

    #1epochごとの処理
    def execute_epoch(self, epoch, phase):
        preds, labels,total_loss= [], [],0
        if phase == 'learning':
            self.net.train()
        else:
            self.net.eval()

        for inputs_, labels_,_ in tqdm(self.dataloaders[phase]):

            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)

            #Samplerを使うときの処理
            if phase == 'learning' and ((self.c['sampler'] == 'over') or (self.c['sampler'] == 'under')):
                inputs_ = inputs_.unsqueeze(1)
                #labels_ = labels_.unsqueeze(1)


            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'learning'):
                outputs_ = self.net(inputs_).to(device)
                #outputs__ = outputs_.unsqueeze(1)
                loss = self.criterion(outputs_, labels_)
                total_loss += loss.item()

                if phase == 'learning':
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

            preds += [outputs_.detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]
            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        try:
            roc_auc = roc_auc_score(labels, preds)
        except:
            roc_auc = 0


        threshold = 0.5
        preds[preds > threshold] = 1
        preds[preds <= threshold] = 0

        preds = np.argmax(preds,axis=1)
        labels = np.argmax(labels,axis=1)

        total_loss /= len(preds)
        recall = recall_score(labels,preds)
        precision = precision_score(labels,preds)
        precisions, recalls, thresholds = precision_recall_curve(labels, preds)
        try:
            pr_auc = auc(recalls, precisions)
        except:
            pr_auc = 0
        f1 = f1_score(labels,preds)
        confusion_Matrix = confusion_matrix(labels,preds)
        try:
            TN = confusion_Matrix[0][0]
        except:
            TN = 0
        try:
            FN = confusion_Matrix[0][1]
        except:
            FN = 0
        try:
            FP = confusion_Matrix[1][0]
        except:
            FP = 0
        try:
            TP = confusion_Matrix[1][1]
        except:
            TP = 0


        print(
            f'epoch: {epoch} phase: {phase} loss: {total_loss:.3f} roc-auc: {roc_auc:.3f} pr-auc: {pr_auc:.3f} f1:{f1:.3f} TN:{TN} FN:{FN} FP:{FP} TP:{TP}')

        #ここにpr_auc,f1,tn,fn,tp,fpを追加
        result_list = [self.c['model_name'],self.c['lr'],self.c['seed'],self.c['preprocess'],self.c['beta'],self.c['gamma'],phase,epoch,total_loss,roc_auc,pr_auc,f1,TN,FN,FP,TP]

        write_Scores(self.log_path,result_list)
        
        #with open(self.log_path + "/log.csv",'a') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(result_list)


        return auc,total_loss,recall,precision