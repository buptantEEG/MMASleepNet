from re import A
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import cohen_kappa_score
# Leave One Subject Out
from dataset.data_loaders import dataload_test
import model
import torch
import numpy as np
import os
import model
import model_paper
import glob as glb
import time
from xml.dom.minidom import parse
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.metrics import geometric_mean_score
# import xlsxwriter

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class Config():
    def __init__(self): 
        self.ss=False
        self.date = time.strftime('%m%d_%H%M')
        self.batch_size=256
        self.afr_reduced_cnn_size=30
        self.d_model = 96
        self.inplanes=128
        #transformer
        self.nhead=4
        self.num_layers=1
        self.model = model.MMASleepNet_4(self)

class Config2():
    def __init__(self): 
        self.ss=True
        self.date = time.strftime('%m%d_%H%M')
        self.batch_size=256
        self.afr_reduced_cnn_size=30
        self.d_model = 64
        #transformer
        self.nhead=4
        self.num_layers=3
        self.model = model.AMMSleepNet_2(self)
        

def test(dataloader, model,ss):
    model.eval()
    result=[]
    true=[]
    if ss==1 or ss==9:
        with torch.no_grad():
            for X_0,y in dataloader:
                X0 = X_0.type(torch.FloatTensor).to(device)
                pred = model(X0)
                # print(pred.argmax(1).tolist())
                result= result + pred.argmax(1).tolist()
                true=true+y.tolist()
    
    elif ss==4 or ss==11 or ss==12:
        with torch.no_grad():
            for X_0,X_1,X_2,X_3,y in dataloader:
                X0 = X_0.type(torch.FloatTensor).to(device)
                X1 = X_1.type(torch.FloatTensor).to(device)
                X2 = X_2.type(torch.FloatTensor).to(device)
                X3 = X_3.type(torch.FloatTensor).to(device)
                pred = model(X0,X1,X2,X3)
                # print(pred.argmax(1).tolist())
                result= result + pred.argmax(1).tolist()
                true=true+y.tolist()
    elif ss==8 or ss==3:
        with torch.no_grad():
            for X_0,X_1,X_2,y in dataloader:
                X0 = X_0.type(torch.FloatTensor).to(device)
                X1 = X_1.type(torch.FloatTensor).to(device)
                X2 = X_2.type(torch.FloatTensor).to(device)
                pred = model(X0,X1,X2)
                # print(pred.argmax(1).tolist())
                result= result + pred.argmax(1).tolist()
                true=true+y.tolist()
    elif ss==5 or ss==10:
        with torch.no_grad():
            for X_0,X_1,y in dataloader:
                X0 = X_0.type(torch.FloatTensor).to(device)
                X1 = X_1.type(torch.FloatTensor).to(device)
                pred = model(X0,X1).reshape(-1,5)
                y=y.reshape(-1)
                # print(pred.shape)
                # print(pred.argmax(1).tolist())
                result= result + pred.argmax(1).tolist()
                true=true+y.tolist()
    print('pred length:',len(result))
    print("true length:",len(true))
    return np.array(result),np.array(true)

def test_noSS(dataloader, model):
    model.eval()
    result=[]
    with torch.no_grad():
        for X_0,X_1,X_2,_ in dataloader:
            X0 = X_0.type(torch.FloatTensor).to(device)
            X1 = X_1.type(torch.FloatTensor).to(device)
            X2 = X_2.type(torch.FloatTensor).to(device)
            pred = model(X0,X1,X2)
            # print(pred.argmax(1).tolist())
            result= result + pred.argmax(1).tolist()
    print(len(result))
    return np.array(result)


def save_config(config, writer):
    dict = vars(config) # return type dict
    for key, value in dict.items():
        writer.add_text(key, str(value))
        # print(i, key, value)

def load_checkpoint(model, checkpoint_PATH):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        # for name, weights in model_CKPT.items():
        #     # print(name, weights.size())  可以查看模型中的模型名字和权重维度
        #     if len(weights.size()) == 2: #判断需要修改维度的条件
        #         model_CKPT[name] = weights.squeeze(0)  #去掉维度0，把(1,128)转为(128)
        model.load_state_dict(model_CKPT)
    return model

def get_label(filename):
      
        
        label = []
        DOMTree = parse(filename)
        root = DOMTree.documentElement
        sleep_stages = root.getElementsByTagName('SleepStage')
        # print('number of sleep stages: ', len(sleep_stages))

        for i in range(len(sleep_stages)):
            label.append(int(sleep_stages[i].firstChild.data)) # '0' '1' '2' '3' '5' -> 0 1 2 3 5
        # print(label[:100])

        label = np.array(label)
        # print(np.unique(label, return_counts=True))
        # label = np.where(label==5, 4, label) # turn label from 01235 to 01234
        # print(np.unique(label, return_counts=True))
        return label # ndarray

class Config():
    def __init__(self,model_name,dataset):
        if model_name=='MMASleepNet':
            self.scaler = None
            self.batch_size=256
            self.afr_reduced_cnn_size=4
            self.d_model = 96
            #transformer
            self.ch=4
            self.nhead=4
            self.num_layers=1
            self.inplanes=self.ch
            self.SS=3
            # Model
            if dataset=='isruc-sleep-3':
                self.model=model_paper.MMASleepNet_eegeogemg_plus_ISRUC(self).to(device)
            else:
                self.model=model_paper.MMASleepNet_eegeogemg_plus(self).to(device)
        elif model_name=='SalientSleep' :
            self.model=model.SalientSleepNet().to(device)
            self.SS=SS
        elif model_name=='AttnSleep' :
            if dataset=='isruc-sleep-3':
                self.model=model_paper.AttnSleep_isruc().to(device)
            else:
                self.model=model.AttnSleep().to(device)
            self.SS=SS
        elif model_name=='SleepPrintNet' :
            if dataset=='isruc-sleep-3-printnet':
                self.model=model_paper.SleepPrintNet_isruc().to(device)
            elif dataset=='sleepedf-78-printnet':
                self.model=model_paper.SleepPrintNet().to(device)
            elif dataset=='sleepedf-20':
                self.model=model.SleepPrintNet().to(device)
            self.SS=SS

            
        

if __name__ == '__main__':
    import argparse
    import seaborn as sns #导入包
    import pandas as pd
 
    

    from util.utils import *
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-d', '--dataset', type=str,
                      help='name of dataset,could be sleepedf-20,sleepedf-78,isruc-sleep-3')
    args.add_argument('-m', '--modelname', type=str,
                      help='name of model, could be MMASleepNet,AttnSleep,SalientSleep,SleepPrintNet')
    args.add_argument('-s', '--SS', type=str,
                      help='parameter of loading data')

    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = []
    
    args2 = args.parse_args()
    # 1. 选择模型 和数据
    dataset = args2.dataset #dataset = 'sleepedf-20','sleepedf-78','isruc-sleep-3','isruc-sleep-3-printnet'
    modelname = args2.modelname # modelname = 'MMASleepNet', 'AttnSleep', 'SalientSleep', 'SleepPrintNet'
    SS = int(args2.SS) ## MMASleepNet->isruc:SS=8 sleepedf:SS=3 ; 
    # AttnSleep->isruc:SS=10 sleepedf:SS=12 ; 
    

    # 2. 设置数据地址和模型地址
    if dataset=="sleepedf-78-printnet":
        datapath = f'/home/brain/code/SleepStaging/data_npy/{dataset}'
    else:
        datapath = f'/home/brain/code/SleepStagingPaper/data_npy/{dataset}'
    datalist = os.listdir(datapath)
    

    ckptpath = f'/home/brain/code/SleepStagingPaper/ckpt/paper/{modelname}/{dataset}/'
    
    if dataset=="isruc-sleep-3" or dataset=="isruc-sleep-3-printnet":
        k=10
        folds_data,subjects = load_folds_data_isruc(datapath, k)
        subject = [0,1,2,3,4,5,6,7,8,9]
    elif dataset=="sleepedf-78" or  dataset=="sleepedf-78-printnet":
        k=10
        folds_data,subjects = load_folds_data(datapath, k)
        subject = [0,1,2,3,4,5,6,7,8,9]
    elif dataset=="sleepedf-20":
        k=20
        folds_data,subjects = load_folds_data(datapath, k)
        subject = [14,5,4,17,8,7,19,12,0,15,16,9,11,10,3,1,6,18,2,13]

    # 3. 开启循环，从0-k：
    turelabel=[]
    predict=[]
    for i in range(k):
    #### 3.1 加载对应的fold保存的模型参数文件
        if modelname=='MMASleepNet':
            checkpoint_PATH=ckptpath+f'fold_{i}/'
        else:
            m = subject[i]
            checkpoint_PATH=ckptpath+f'fold_{m}/'
        # print(checkpoint_PATH)
        checkpoint_file_PATH=glb.glob(checkpoint_PATH+"*.pth")
        print(checkpoint_file_PATH)

        cfg=Config(modelname,dataset)
        net = cfg.model
        net = load_checkpoint(net,checkpoint_file_PATH[0]).to(device)

    #### 3.2 加载对应的fold的数据
        print("test file:",folds_data[i][1])
        test_loader=dataload_test(subject_files=folds_data[i][1], 
                batch_size=8,SS=SS)

    #### 3.3 model test\
        result,ture = test(test_loader,net,SS)
        temp_acc = accuracy_score(ture, result)
        print("temp_acc:",temp_acc)
    #### 3.4 把预测结果加到整体预测结果中
        # if i==0:
        #     predict=result
        #     turelabel=ture
        # else:
        #### 3.4 把预测结果加到整体预测结果中
        predict=np.concatenate((predict,result),axis=0)
        #### 3.5 把真实结果加到整体真实矩阵中
        turelabel=np.concatenate((turelabel,ture),axis=0)
        print(predict.shape[0])
    # 4. 计算真实结果与测试结果的混淆矩阵
    print("all predict:",len(predict))
    print("all truelabels:",len(turelabel))

    acc = accuracy_score(turelabel, predict)
    f1score = f1_score(turelabel, predict, average='macro')
    print('acc',acc)
    print('f1score',f1score)
    print(classification_report(turelabel, predict,digits=4))
    C=confusion_matrix(turelabel, predict)
    kappa = cohen_kappa_score(turelabel, predict)
    MGm=geometric_mean_score(turelabel, predict)
    print('C',C)
    df=pd.DataFrame(C)
    sns.heatmap(df,fmt='g',annot=True,cmap='Blues')

    print('kappa',kappa)
    print('MGm',MGm)
    
    print("Done!")
    