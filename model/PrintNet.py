from re import T
from turtle import forward
import torch
from torch._C import TensorType
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy

from model.BasicModel import BasicModel

class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Temporal_feature_EEG(nn.Module):
    def __init__(self):
        super(Temporal_feature_EEG, self).__init__()
        drate = 0.5
        self.Flatten = Flatten()
        self.features1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=64, stride=8, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
    
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),            

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
            
        )
        

        self.features2 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=512, stride=64, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
    
            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),            

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        # self.Flatten=nn.Flatten()
 

    def forward(self, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(dim = 0)
        y1 = self.Flatten(self.features1(y))
        y2 = self.Flatten(self.features2(y))
        y_concat = torch.cat((y1, y2), dim=1)
        return y_concat

        

# class Spectral_Spatial(nn.Module):
#     def __init__(self):
#         super(Spectral_Spatial, self).__init__()
#         self.Flatten = Flatten()
#         self.features0 = nn.Sequential(
#             nn.Conv1d(1,32,kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(), 

#         ) 
#         self.features1 = nn.Sequential(
#             nn.Conv1d(32,64,kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(64),
#         )     
#         self.features2 = nn.Sequential(
#             nn.Conv1d(32,64,kernel_size=3, stride=1, bias=False, padding=1),
#             nn.BatchNorm1d(64),
#         )

#         self.feature3 = nn.Sequential(

#             nn.MaxPool1d(kernel_size=4, stride=2,padding=0),
#             nn.LeakyReLU(),
#             self.Flatten,
#             nn.Dropout(0.5),
#         )
#     def forward(self, x):
        
#         x = x.unsqueeze(dim = 1)

#         x_psd = self.features0(x)
#         x_1 = self.features1(x_psd)
#         x_2 = self.features2(x_psd)
#         x_add_temp = torch.add(x_1, x_2)
#         x_add_temp = x_add_temp.squeeze(0)
#             # print('x_1,x_2,x_add_temp',x_1.shape,x_2.shape,x_add_temp.shape)
#         x_ss = self.feature3(x_add_temp)
#         return x_ss
class Spectral_Spatial(nn.Module):
    def __init__(self):
        super(Spectral_Spatial, self).__init__()
        self.Flatten = Flatten()
        self.features0 = nn.Sequential(
            nn.Conv2d(5,32,kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(), 

        ) 
        self.features1 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
        )     
        self.features2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(64),
        )

        self.feature3 = nn.Sequential(

            nn.MaxPool2d(kernel_size=4, stride=2,padding=0),
            nn.LeakyReLU(),
            self.Flatten,
            nn.Dropout(0.5),
        )
    def forward(self, x):
        
        # x = x.unsqueeze(dim = 1)
        # print('x_ss',x.shape)
        x_psd = self.features0(x)
        x_1 = self.features1(x_psd)
        x_2 = self.features2(x_psd)
        x_add_temp = torch.add(x_1, x_2)
        x_add_temp = x_add_temp.squeeze(0)
            # print('x_1,x_2,x_add_temp',x_1.shape,x_2.shape,x_add_temp.shape)
        x_ss = self.feature3(x_add_temp)
        return x_ss
        

class Temporal_feature_multimodel(nn.Module):
    def __init__(self,channels):
        super(Temporal_feature_multimodel, self).__init__()
        drate = 0.5
        self.Flatten = Flatten()
        self.features1 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=64, stride=8, bias=False, padding=24),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
    
            nn.Conv1d(64, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),            

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
            
        )
        

        self.features2 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=512, stride=64, bias=False, padding=24),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(32, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
    
            nn.Conv1d(64, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),            

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
    def forward(self,x):
        
        if len(x.shape) == 2:
            x = x.unsqueeze(dim = 0)
        # print('x.shape',x.shape)
        x1 = self.Flatten(self.features1(x))
        x2 = self.Flatten(self.features2(x))
        x_concat = torch.cat((x1, x2), dim=1)
        return x_concat


class SleepPrintNet(BasicModel):
    def __init__(self):
        super(SleepPrintNet, self).__init__()
        self.EEG_feature = Temporal_feature_EEG()
        self.EOG_feature = Temporal_feature_multimodel(channels=1)
        self.EMG_feature = Temporal_feature_multimodel(channels=1)
        self.SS_feature = Spectral_Spatial()
        self.linear1 = nn.Sequential(nn.Linear(9280,256),nn.ReLU(True))
        self.linear2 = nn.Sequential(nn.Linear(256,5))
        
    
    def forward(self, x1,x2,x3,x4):
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        # x2 = x2.permute(1,0)
        # x3 = x3.permute(1,0)
        # x4 = x4.permute(1,0)
        x2 = x2.unsqueeze(dim = 1)
        x3 = x3.unsqueeze(dim = 1)
        # x4 = x4.unsqueeze(dim = -1)

        x_EEG1=self.EEG_feature(x1)
        
        x_EOG=self.EOG_feature(x2)
        x_EMG=self.EMG_feature(x3)
        x_EEG2=self.SS_feature(x4)
        # print('x_EEG1,x_EOG,x_EMG,x_EEG2',x_EEG1.shape,x_EOG.shape,x_EMG.shape,x_EEG2.shape)
        if x_EEG1.shape[0] != x_EEG2.shape[0]:
            x_EEG2 = x_EEG2.reshape(x_EEG1.shape[0],int(x_EEG2.shape[0]*x_EEG2.shape[1]/x_EEG1.shape[0]))
        x_cat=torch.cat((x_EEG1,x_EOG,x_EMG,x_EEG2),dim=1)#
        # print(x_cat.shape)
        x_final = self.linear1(x_cat)
        x_final = self.linear2(x_final)
        return x_final



