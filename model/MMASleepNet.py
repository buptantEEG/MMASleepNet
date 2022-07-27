## Attention based MultiModal Sleep Staging Network

from re import T
from turtle import forward
import torch
from torch._C import TensorType
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from torch.nn import init

from model.BasicModel import BasicModel

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Temporal_feature_EEG(nn.Module):
    def __init__(self,channels):
        super(Temporal_feature_EEG, self).__init__()
        drate = 0.5
        # self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)
        self.features1 = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=64, stride=8, bias=False, padding=24),
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
            nn.Conv1d(channels, 64, kernel_size=512, stride=64, bias=False, padding=24),
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
        self.dropout = nn.Dropout(drate)
    

    def forward(self, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(dim = 1)
        y1 = self.features1(y)
        y2 = self.features2(y)
        # print(y1.shape)
        y_concat = torch.cat((y1, y2), dim=2)
        y_concat = self.dropout(y_concat)
        # y_concat = self.AFR(y_concat)
        return y_concat

class Temporal_feature_multimodel(nn.Module):
    def __init__(self,channels):
        super(Temporal_feature_multimodel, self).__init__()
        drate = 0.5
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
        self.dropout = nn.Dropout(drate)
    
    def forward(self,x):
        
        if len(x.shape) == 2:
            x = x.unsqueeze(dim = 1)
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)

        return x_concat

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _  = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y=y.view(b, c, 1 ,1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=1):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,stride=(stride,stride))
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        # print("x.shape:",x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print("out shape:",out.shape)
        # [256, 30, 128, 24]
        out = self.se(out)
        # print("out.shape:",out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)

        # print("x_downsample.shape:",residual.shape)
        out += residual
        out = self.relu(out)

        return out

class MMASleepNet(BasicModel):
    def __init__(self,cfg):#,d_model,afr_reduced_cnn_size):
        super(MMASleepNet, self).__init__()
        
        self.afr_reduced_cnn_size=cfg.afr_reduced_cnn_size
        self.d_model = cfg.d_model
        self.inplanes = cfg.inplanes
        self.nhead=cfg.nhead
        self.num_layers=cfg.num_layers
        self.EEG_channels=cfg.EEG_channels

        self.EEG_feature = Temporal_feature_EEG(channels=self.EEG_channels)
        self.EOG_feature = Temporal_feature_multimodel(channels=1)
        self.EMG_feature = Temporal_feature_multimodel(channels=1)

        self.linear1 = nn.Sequential(nn.Linear(self.d_model*self.afr_reduced_cnn_size*16,32),nn.ReLU(True))
        self.linear2 = nn.Sequential(nn.Linear(32,5),nn.Softmax(dim=1))

        self.AFR = self._make_layer(SEBasicBlock, self.afr_reduced_cnn_size, blocks = 1 ,stride=1)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead,batch_first=True,activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
    
    def _make_layer(self, block, planes, blocks, stride=4):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1,x2,x3):
        self.batch_size=x1.shape[0]

        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)

        x_EEG=self.EEG_feature(x1).view(self.batch_size,2,64,24)
        x_EOG=self.EOG_feature(x2).view(self.batch_size,1,64,24)
        x_EMG=self.EMG_feature(x3).view(self.batch_size,1,64,24)

        x_cat=torch.cat((x_EEG,x_EOG,x_EMG),dim=1) #[256,4,64,24]
        # print("x_cat_2d.shape:",x_cat_2d.shape)
        x_afr = self.AFR(x_cat) #[256,4,64,24]
        # print(x_cat.shape)
        x_afr= x_afr.view(self.batch_size,-1,96) 
        # x_cat=x_cat.squeeze()
        # print(x_afr.shape)
        encoded_features=self.transformer_encoder(x_afr.view(self.batch_size,-1,96) )
        
        # print(encoded_features.shape)

        encoded_features=x_afr*encoded_features
        encoded_features=encoded_features.contiguous().view(encoded_features.shape[0], -1)
        # print(encoded_features.shape)
        x_final = self.linear1(encoded_features)
        x_final = self.linear2(x_final)
        # print(x_final.shape)
        return x_final



