import os
import sys
import time
import random
import config
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import seaborn as sns
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils  import *
from network import *
from Dataset import *
from tqdm import tqdm
from sklearn.metrics import *
from datetime import datetime
from PIL import Image
from torchvision.utils import make_grid, save_image
import matplotlib
import csv
import copy


c = {
    'model_name': 'Resnet18','seed': 0, 'bs': 32
}

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def write_LogHeader(log_path):
    #CSVファイルのヘッダー記述
    with open(log_path + "/evaluate_log.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Modelno','pr-auc'])

def write_Scores(log_path,result_list):
    #リストを1行書き加える。
    with open(log_path + "/evaluate_log.csv",'a') as f:
        writer = csv.writer(f)
        writer.writerow(result_list)

class Evaluater():
    def __init__(self,c):
        self.dataloaders = {}
        self.c = c
        self.now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                'evaluate')
        

        #コマンドラインの変数を処理する部分。
        args = len(sys.argv)
        with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv')) as f:
            lines = [s.strip() for s in f.readlines()]
        # 引数がない場合は一番最近のモデルを使用
        if args < 2 :
            target_data = lines[-1].split(',')
        elif isint(sys.argv[1]):
            if int(sys.argv[1])<1:
                print('Use the first data')
                target_data = lines[-1].split(',')
            else:
                try:
                    target_data = lines[int(sys.argv[1])].split(',')
                    self.c['n_per_unit'] = 1 if sys.argv[2] == 'horizontal' else 16
                    self.c['type'] = sys.argv[3].split('=')[1]
                except IndexError:
                    print('It does not exit. Use the first data')
                    target_data = lines[-1].split(',')
                    self.c['n_per_unit'] = 1 if sys.argv[2] == 'horizontal' else 16
        else:
            self.c['n_per_unit'] = 1 if sys.argv[1] == 'horizontal' else 16
            target_data = lines[-1].split(',')

        #上の情報からモデル名を作る。
        self.n_ex = '{:0=2}'.format(int(target_data[1]))
        self.c['model_name'] = target_data[2]
        self.c['n_epoch'] = '{:0=3}'.format(int(target_data[3]))
        temp = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep.pth'
        model_path = os.path.join(config.MODEL_DIR_PATH,temp)

        #モデルの作成。
        self.net = make_model(self.c['model_name'],self.c['n_per_unit'])
        self.net.load_state_dict(torch.load(model_path,map_location=device))
        self.criterion = nn.BCELoss()

    def run(self):
        #各々の部位を整数に変換
        self.c['type'] = CataractTypeToInt(self.c['type'])

        #テストデータセットの用意。
        self.dataset = load_dataset(self.c['n_per_unit'],self.c['type'],'Add_BlackRects')
        test_id_index,_ = calc_kfold_criterion('test')
        test_index,_ = calc_dataset_index(test_id_index,[],'test',self.c['n_per_unit'])
        test_dataset = Subset(self.dataset['test'],test_index)
        self.dataloaders['test'] = DataLoader(test_dataset,self.c['bs'],
                    shuffle=False,num_workers=os.cpu_count())
        preds,labels,paths,total_loss,accuracy= [],[],[],0,0
        right,notright = 0,0

        self.net.eval()

        for inputs_, labels_,paths_ in tqdm(self.dataloaders['test']):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)


            torch.set_grad_enabled(True)
            outputs_ = self.net(inputs_)
            loss = self.criterion(outputs_, labels_)

            preds += [outputs_.detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]
            paths += paths_

            total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        preds = sigmoid(preds)[:,1]
        labels = np.argmax(labels,axis=1)

        r_cnt,tmp,cnt = 0,0,0
        total_loss /= len(preds)

        print(labels,preds)

        #作成画像の保存パス
        model_info = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep'

        # Auc値の計算,ROC曲線の描画
        roc_auc = roc_auc_score(labels, preds)
        fig_path = model_info + '_ep_ROC.png'
        save_fig_path = os.path.join(config.LOG_DIR_PATH,'images',fig_path)
        make_ROC(labels,preds,save_fig_path)

        # PR-AUCの計算,PR曲線の描画
        fig_path = model_info + '_ep_PRC.png'
        save_fig_path = os.path.join(config.LOG_DIR_PATH,'images',fig_path)
        make_PRC(labels,preds,save_fig_path)

        #出力をもとに2値分類
        precisions, recalls, thresholds = precision_recall_curve(labels, preds)
        pr_auc = auc(recalls, precisions)

        f1_list = []

        #for threshold in thresholds:
        #    preds_cpy = copy.deepcopy(preds)
        #    preds_cpy[preds_cpy>threshold] = 1
        #    preds_cpy[preds_cpy<=threshold] = 0
        #    f1 = f1_score(preds_cpy,labels)
        #    f1_list.append(f1)

        #print(f1_list)
        


        #混同行列を作り、ヒートマップで可視化。
        fig_path = model_info + '_ep_CM.png'
        save_fig_path = os.path.join(config.LOG_DIR_PATH,'images',fig_path)

        threshold = 0.53
        preds[preds > threshold] = 1
        preds[preds <= threshold] = 0
        cm = confusion_matrix(labels,preds)
        make_ConfusionMatrix(cm,save_fig_path)

        right += (preds == labels).sum()
        notright += len(preds) - (preds == labels).sum()
        accuracy = right / len(test_dataset)
        recall = recall_score(labels,preds)
        precision = precision_score(labels,preds)
        f1 = f1_score(labels,preds)

        print('accuracy :',accuracy)
        print('AUC-ROC :',roc_auc)
        print('AUC-PRC :',pr_auc)
        print('F1 Score', f1)
        print('Precision',precision)
        print('Recall',recall)


        #評価値の棒グラフを作って保存。
        fig,ax = plt.subplots()
        ax.bar(['Acc','AUC-ROC','Recall','Precision','AUC-PRC'],[accuracy,pr_auc,recall,precision,pr_auc],width=0.4,tick_label=['Accuracy','Auc','Recall','Precision','AUC-PRC'],align='center')
        ax.grid(True)
        plt.yticks(np.linspace(0,1,21))
        fig_path = model_info+'ep_graph.png'
        fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))


        # 結果をcsvに記録
        result_list = [self.n_ex,pr_auc]
        write_Scores(self.log_path,result_list)


if __name__ == '__main__':
    evaluater = Evaluater(c)
    evaluater.run()