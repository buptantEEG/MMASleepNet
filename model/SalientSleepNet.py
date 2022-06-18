#https://github.com/ziyujia/SalientSleepNet

from ast import Lambda
from re import S, U
from tkinter.tix import InputOnly
import torch
from torch._C import TensorType
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from functools import reduce

from model.BasicModel import BasicModel
# from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torchvision import transforms


# class Upsample(nn.Module):
#     def __init__(self,crop_size, upscale_factor):
#         super(Upsample, self).__init__()
#         self.upsample = Compose([Resize(),ToTensor()])
        
#     def forward(self, x):
#         x_up = self.upsample(x)
#         return x_up


class creat_bn_conv(nn.Module):
    def __init__(self,input_size,filter_size,kernel_size,padding,dilation):
        super(creat_bn_conv, self).__init__()
        self.conv = nn.Conv2d(input_size,filter_size,kernel_size=(1,kernel_size),stride=(1,1),padding=padding,dilation=(dilation,dilation))
        self.bn = nn.BatchNorm2d(filter_size)
    def forward(self, x):
        
        x_conv = self.conv(x)
        # print("x_conv shape:",x_conv.shape)
        x_bn = self.bn(x_conv)
        return x_bn

class creat_u_encoder(nn.Module):
    def __init__(self, 
                input,
                final_filter,
                kernel_size,
                pooling_size,
                middle_layer_filter,
                depth,
                padding
                ):
        super(creat_u_encoder, self).__init__()
        self.depth=depth
        self.creat_bn_conv0 = creat_bn_conv(input_size=input,filter_size=final_filter,kernel_size=kernel_size,padding='same',dilation=1)#final_layer
        self.creat_bn_conv1 = nn.ModuleList()
        self.creat_bn_conv1.append(creat_bn_conv(input_size=final_filter,filter_size=middle_layer_filter,kernel_size=kernel_size,padding='same',dilation=1))
        self.creat_bn_conv1.append(creat_bn_conv(input_size=middle_layer_filter,filter_size=middle_layer_filter,kernel_size=kernel_size,padding='same',dilation=1))
        self.creat_bn_conv1.append(creat_bn_conv(input_size=middle_layer_filter,filter_size=middle_layer_filter,kernel_size=kernel_size,padding='same',dilation=1))
        
        self.creat_bn_conv2 = creat_bn_conv(input_size=middle_layer_filter,filter_size=middle_layer_filter,kernel_size=kernel_size,padding='same',dilation=1)
        self.creat_bn_conv3 = nn.ModuleList()
        self.creat_bn_conv3.append(creat_bn_conv(input_size=middle_layer_filter*2,filter_size=final_filter,kernel_size=kernel_size,padding='same',dilation=1))
        self.creat_bn_conv3.append(creat_bn_conv(input_size=middle_layer_filter*2,filter_size=middle_layer_filter,kernel_size=kernel_size,padding='same',dilation=1))
        self.creat_bn_conv3.append(creat_bn_conv(input_size=middle_layer_filter*2,filter_size=middle_layer_filter,kernel_size=kernel_size,padding='same',dilation=1))
    
        self.maxpool = nn.MaxPool2d(kernel_size=(1,pooling_size), padding=0)#padding=(pooling_size//2,0))
        # self.upsample = transforms.Resize()#mode:'bilinear''nearest'

    def forward(self, input):
        from_encoder = []
        conv_bn0 = self.creat_bn_conv0(input)
        # print('conv_bn0.shape',conv_bn0.shape)
        conv_bn = conv_bn0
        for d in range(self.depth-1):
            # print(d)
            conv_bn = self.creat_bn_conv1[d](conv_bn)
            # print(f'conv_bn{d}.shape',conv_bn.shape)
            from_encoder.append(conv_bn)
            # print('conv_bn',conv_bn.shape)
            if d != self.depth-2:
                conv_bn = self.maxpool(conv_bn)
        # print("conv_bn:",conv_bn.shape)
        conv_bn = self.creat_bn_conv2(conv_bn)

        for d in range(self.depth-1,0,-1):
            # print('conv_bn.shape up',conv_bn.shape)
            conv_bn = transforms.Resize([from_encoder[-1].shape[2],from_encoder[-1].shape[3]])(conv_bn)
            # print('conv_bn.shape up',conv_bn.shape)
            x_concat = torch.cat((conv_bn,from_encoder.pop()),dim = 1)
            # print(d)
            conv_bn = self.creat_bn_conv3[d-1](x_concat)
        # print('conv_bn,conv_bn0',conv_bn.shape,conv_bn0.shape)
        x_final = torch.add(conv_bn,conv_bn0)
        return x_final

class create_mse(nn.Module):
    def __init__(self,
                input,
                final_filter,
                kernel_size,
                dilation_rates):
        
        super(create_mse, self).__init__()
        self.dilation_rates = dilation_rates
        self.creat_bn_conv0 = nn.ModuleList()
        self.creat_bn_conv0.append(creat_bn_conv(input_size=input,filter_size=final_filter,kernel_size=kernel_size,padding='same',dilation=self.dilation_rates[0]))
        self.creat_bn_conv0.append(creat_bn_conv(input_size=input,filter_size=final_filter,kernel_size=kernel_size,padding='same',dilation=self.dilation_rates[1]))
        self.creat_bn_conv0.append(creat_bn_conv(input_size=input,filter_size=final_filter,kernel_size=kernel_size,padding='same',dilation=self.dilation_rates[2]))
        self.creat_bn_conv0.append(creat_bn_conv(input_size=input,filter_size=final_filter,kernel_size=kernel_size,padding='same',dilation=self.dilation_rates[3]))#final_layer
        
        self.feature = nn.Sequential(
            nn.Conv2d(final_filter*len(self.dilation_rates),final_filter*2,kernel_size=(1,kernel_size),stride=(1,1),padding='same'),
            nn.Conv2d(final_filter*2,final_filter,kernel_size=(1,kernel_size),stride=(1,1),padding='same'),
            nn.BatchNorm2d(final_filter)
        )
    def forward(self,x):
        # print("mse_x:",x.shape)
        convs = []
        for (i,dr) in enumerate(self.dilation_rates):
            conv_bn = self.creat_bn_conv0[i](x)
            convs.append(conv_bn)
        # print('con_conv_0.shape',convs.shape)
        con_conv = reduce(lambda l, r: torch.cat([l, r],dim=1), convs)
        # print('con_conv.shape',con_conv.shape)
        out = self.feature(con_conv)
        return out

class SingleSalientSleepNet(BasicModel):
    def __init__(self):
        super(SingleSalientSleepNet, self).__init__()
        self.padding = 'same'
        self.sleep_epoch_length = 3000
        self.sequence_length = 20
        self.filters = [16,32,64,128,256]
        self.kernel_size = 5
        self.pooling_sizes = [10, 8, 6, 4]
        self.dilation = [1, 2, 3, 4]
        
        self.u_depths = [4, 4, 4, 4]
        self.u_inner_filter = 8
        self.mse_filters = [32, 24, 46, 8, 5]
        self.relu = nn.ReLU(inplace=True)
        
        #encoder1
        self.creat_bn_conv_u1_EEG= creat_u_encoder(input=1,
                final_filter=self.filters[0],
                kernel_size=self.kernel_size,
                pooling_size=self.pooling_sizes[0],
                middle_layer_filter=self.u_inner_filter,
                depth=self.u_depths[0],
                padding=self.padding)
        # self.creat_bn_conv_u1_EOG= creat_u_encoder(input_size=1,filter_size=self.filters[0],kernel_size=self.kernel_size,padding=self.padding,dilation=self.dilation[0])
        self.u1 = nn.Sequential(
        
            nn.Conv2d(self.filters[0],int(self.filters[0]/2),kernel_size = (1,1),stride=(1,1),padding = (0,0)),
            nn.ReLU()
           
        )
      
        #encoder2
        self.creat_bn_conv_u2 = creat_u_encoder(
                input=int(self.filters[0]/2),
                final_filter=self.filters[1],
                kernel_size=self.kernel_size,
                pooling_size=self.pooling_sizes[1],
                middle_layer_filter=self.u_inner_filter,
                depth=self.u_depths[1],
                padding=self.padding)
        
        self.u2 = nn.Sequential(
            
            nn.Conv2d(self.filters[1],int(self.filters[1]/2),kernel_size = (1,1),stride=(1,1),padding = self.padding),
            nn.ReLU(),
            
        )
        #encoder3
        self.creat_bn_conv_u3 = creat_u_encoder(input=int(self.filters[1]/2),
                final_filter=self.filters[2],
                kernel_size=self.kernel_size,
                pooling_size=self.pooling_sizes[2],
                middle_layer_filter=self.u_inner_filter,
                depth=self.u_depths[2],
                padding=self.padding)
        
        self.u3 = nn.Sequential(
            
            nn.Conv2d(self.filters[2],int(self.filters[2]/2),kernel_size = (1,1),stride=(1,1),padding = self.padding),
            nn.ReLU(),
            
        )
        #encoder4
        self.creat_bn_conv_u4 = creat_u_encoder(input=int(self.filters[2]/2),
                final_filter=self.filters[3],
                kernel_size=self.kernel_size,
                pooling_size=self.pooling_sizes[3],
                middle_layer_filter=self.u_inner_filter,
                depth=self.u_depths[3],
                padding=self.padding)
    
        self.u4 = nn.Sequential(
           
            nn.Conv2d(self.filters[3],int(self.filters[3]/2),kernel_size = (1,1),stride=(1,1),padding = self.padding),
            nn.ReLU(),
            
        )
        #encoder5
        self.creat_bn_conv_u5 = creat_u_encoder(input=int(self.filters[3]/2),
                final_filter=self.filters[4],
                kernel_size=self.kernel_size,
                pooling_size=self.pooling_sizes[3],
                middle_layer_filter=self.u_inner_filter,
                depth=self.u_depths[3],
                padding=self.padding)
        
        self.u5 = nn.Sequential(
            nn.Conv2d(self.filters[4],int(self.filters[4]/2),kernel_size = (1,1),stride=(1,1),padding = self.padding),
            nn.ReLU()
        )
       
        #MES
        self.create_mse1 = create_mse(int(self.filters[0]/2),self.mse_filters[0],self.kernel_size, self.dilation)
        self.create_mse2 = create_mse(int(self.filters[1]/2),self.mse_filters[1],self.kernel_size, self.dilation)
        self.create_mse3 = create_mse(int(self.filters[2]/2),self.mse_filters[2],self.kernel_size, self.dilation)
        self.create_mse4 = create_mse(int(self.filters[3]/2),self.mse_filters[3],self.kernel_size, self.dilation)
        self.create_mse5 = create_mse(int(self.filters[4]/2),self.mse_filters[4],self.kernel_size, self.dilation)
        #decoder4

        self.creat_u_encoder_d4 = creat_u_encoder(13,self.filters[3],self.kernel_size,self.pooling_sizes[3],self.u_inner_filter,depth=self.u_depths[3],padding=self.padding)
        self.d4 = nn.Conv2d(self.filters[3],self.filters[3]//2,kernel_size = (1,1),stride=(1,1),padding = self.padding)
        #decoder3
        self.creat_u_encoder_d3 = creat_u_encoder(110,self.filters[2],self.kernel_size,self.pooling_sizes[2],self.u_inner_filter,depth=self.u_depths[2],padding=self.padding)
        self.d3 = nn.Conv2d(self.filters[2],self.filters[2]//2,kernel_size = (1,1),stride=(1,1),padding = self.padding)

        #decoder2
        self.creat_u_encoder_d2 = creat_u_encoder(56,self.filters[1],self.kernel_size,self.pooling_sizes[1],self.u_inner_filter,depth=self.u_depths[1],padding=self.padding)
        self.d2 = nn.Conv2d(self.filters[1],self.filters[1]//2,kernel_size = (1,1),stride=(1,1),padding = self.padding)
        #decoder1
        self.creat_u_encoder_d1 = creat_u_encoder(48,self.filters[0],self.kernel_size,self.pooling_sizes[0],self.u_inner_filter,depth=self.u_depths[0],padding=self.padding)
        self.d1 = nn.Conv2d(self.filters[0],self.filters[0]//2,kernel_size = (1,1),stride=(1,1),padding = self.padding)

        # self.zero = nn.ZeroPad2d(int((self.sleep_epoch_length - int(self.filters[0]/2)) // 2))
        
        
        
    def forward(self,x):
        #EEG
        x_EEG = x.unsqueeze(dim =1)
        x_EEG = x_EEG.unsqueeze(dim =2)
        # print(x_EEG.shape)
        #encoder
        x_EEG_u1 = self.creat_bn_conv_u1_EEG(x_EEG)
        # print('x_EEG_u1.shape',x_EEG_u1.shape)
        x_EEG_u1 = self.u1(x_EEG_u1)
        x_EEG_u1_pool =nn.MaxPool2d(kernel_size=(1,self.pooling_sizes[0]))(x_EEG_u1)


        # print('x_EEG_u1.shape2',x_EEG_u1.shape)
        x_EEG_u2 = self.creat_bn_conv_u2(x_EEG_u1_pool)
        # print('x_EEG_u2.shape2',x_EEG_u2.shape)
        x_EEG_u2 = self.u2(x_EEG_u2)
        x_EEG_u2_pool =nn.MaxPool2d(kernel_size=(1,self.pooling_sizes[1]))(x_EEG_u2)
        
        x_EEG_u3 = self.creat_bn_conv_u3(x_EEG_u2_pool)
        # print('x_EEG_u3.shape2',x_EEG_u3.shape)
        x_EEG_u3 = self.u3(x_EEG_u3)
        x_EEG_u3_pool =nn.MaxPool2d(kernel_size=(1,self.pooling_sizes[2]))(x_EEG_u3)
        # 
        # print(x_EEG_u3.shape)
  
        x_EEG_u4 = self.creat_bn_conv_u4(x_EEG_u3_pool)
        # print('x_EEG_u4.shape2',x_EEG_u4.shape)
        x_EEG_u4 = self.u4(x_EEG_u4)
        x_EEG_u4_pool =nn.MaxPool2d(kernel_size=(1,self.pooling_sizes[3]))(x_EEG_u4)
        # 
        x_EEG_u5 = self.creat_bn_conv_u5(x_EEG_u4_pool)
        # print('x_EEG_u5.shape2',x_EEG_u5.shape)
        x_EEG_u5 = self.u5(x_EEG_u5)
        # 
        #MSE
        # print(x_EEG_u1.shape, x_EEG_u2.shape,x_EEG_u3.shape,x_EEG_u4.shape,x_EEG_u5.shape)
        x_EEG_u1 = self.create_mse1(x_EEG_u1)
        x_EEG_u2 = self.create_mse2(x_EEG_u2)
        x_EEG_u3 = self.create_mse3(x_EEG_u3)
        x_EEG_u4 = self.create_mse4(x_EEG_u4)
        x_EEG_u5 = self.create_mse5(x_EEG_u5)
        # print('x_EEG_u1.shape, x_EEG_u2.shape,_EEG_u3.shape,x_EEG_u4.shape,x_EEG_u5.shape',x_EEG_u1.shape, x_EEG_u2.shape,x_EEG_u3.shape,x_EEG_u4.shape,x_EEG_u5.shape)
        # #decoder
        # x_EEG_up4 = nn.functional.interpolate(x_EEG_u5,(x_EEG_u4.shape[2],x_EEG_u4.shape[3]))
        x_EEG_up4 = transforms.Resize([x_EEG_u4.shape[2],x_EEG_u4.shape[3]])(x_EEG_u5)
        # print('x_EEG_up4,x_EEG_u4',x_EEG_up4.shape,x_EEG_u4.shape)
        x_EEG_d4 = torch.cat((x_EEG_up4,x_EEG_u4),dim=1)
        # print('x_EEG_d4',x_EEG_d4.shape)
        x_EEG_d4 = self.creat_u_encoder_d4(x_EEG_d4)
        # print('x_EEG_d41',x_EEG_d4.shape)
        x_EEG_d4 = self.d4(x_EEG_d4)
        # print('x_EEG_d42',x_EEG_d4.shape)

        x_EEG_up3 = transforms.Resize([x_EEG_u3.shape[2],x_EEG_u3.shape[3]])(x_EEG_d4)
        # print('x_EEG_up3,x_EEG_u3',x_EEG_up3.shape,x_EEG_u3.shape)
        x_EEG_d3 = torch.cat((x_EEG_up3,x_EEG_u3),dim=1)
        # print('x_EEG_d3',x_EEG_d3.shape)
        x_EEG_d3 = self.creat_u_encoder_d3(x_EEG_d3)
        x_EEG_d3 = self.d3(x_EEG_d3)

        x_EEG_up2 = transforms.Resize([x_EEG_u2.shape[2],x_EEG_u2.shape[3]])(x_EEG_d3)
        x_EEG_d2 = torch.cat((x_EEG_up2,x_EEG_u2),dim=1)
        # print('x_EEG_d2',x_EEG_d2.shape)
        x_EEG_d2 = self.creat_u_encoder_d2(x_EEG_d2)
        x_EEG_d2 = self.d2(x_EEG_d2)

        x_EEG_up1 = transforms.Resize([x_EEG_u1.shape[2],x_EEG_u1.shape[3]])(x_EEG_d2)
        x_EEG_d1 = torch.cat((x_EEG_up1,x_EEG_u1),dim=1)
        # print('x_EEG_d1',x_EEG_d1.shape)
        x_EEG_d1 = self.creat_u_encoder_d1(x_EEG_d1)
        # x_EEG_d1 = self.d1(x_EEG_d1)
        # print('x_EEG_d1.shape',x_EEG_d1.shape)
        pad2d= ((self.sleep_epoch_length*self.sequence_length-x_EEG_d1.shape[3])//2,(self.sleep_epoch_length*self.sequence_length-x_EEG_d1.shape[3])//2,0,0)
        x_EEG_d1=F.pad(x_EEG_d1,pad2d)
        
        return x_EEG_d1
        

class SalientSleepNet(BasicModel):
    def __init__(self):
        super(SalientSleepNet, self).__init__()
        self.filters = [16,32,64,128,256]
        self.sleep_epoch_length = 3000
        self.sequence_length = 20
        self.kernel_size = 5
        self.padding = 'same'
        self.branch1=SingleSalientSleepNet()
        self.branch2=SingleSalientSleepNet()
        self.linear = nn.Sequential(
            nn.Linear(self.filters[0],self.filters[0]//4),
            nn.ReLU(),
            nn.Linear(self.filters[0]//4,self.filters[0]),
            nn.Sigmoid()
        )
        self.feature = nn.Sequential(
            nn.Conv2d(self.filters[0],self.filters[0],kernel_size=(1,1),stride = (1,1)),
            nn.Tanh(),

            nn.AdaptiveAvgPool2d((self.sequence_length,1)),
            nn.Conv2d(16,5,kernel_size=(self.kernel_size,1),stride = (1,1),padding = self.padding),
            nn.Softmax(dim=1)
        )
    
    def forward(self,EEG,EOG):
        # print('EEG',EEG.shape,'EOG',EOG.shape)
        EEG_feature=self.branch1(EEG[:,1,:])
        EOG_feature=self.branch2(EOG)
        # print('EEG_feature',EEG_feature.shape)
        # EEG_feature = EEG_feature.to_tensors
        # EOG_feature = EOG_feature.to_tensor
        # print(type(EEG_feature),type(EOG_feature))
        x_mul = torch.multiply(EEG_feature,EOG_feature)
        x_merge = torch.add(EEG_feature,EOG_feature)
        # print('x_merge.shape',x_merge.shape)
        x_merge = torch.add(x_merge,x_mul)
        # print('x_merge.shape',x_merge.shape)

        #attention
        # x_se = nn.Aver
        x_se = x_merge.float()
        # print('x_se.shape',x_se.shape)
        x_se = x_se.mean(dim = -1)
        # print('x_se.shape',x_se.shape)
        x_se = x_se.mean(dim = -1)
        # print('x_se.shape',x_se.shape)
        
        
        

        #excitation
        x_se = self.linear(x_se)
        # print('x_se.shape',x_se.shape)
        x_se = x_se.reshape(x_se.shape[0],self.filters[0],1,1)
        # print('x_se.shape',x_se.shape)
        #re-weight
        x = x_merge*x_se.expand_as(x_merge)
        # print(x.shape)
        # x = torch.multiply(x_merge,x_se)
        x_reshape = x.reshape(x.shape[0],x.shape[1],self.sequence_length,self.sleep_epoch_length)
        x_reshape = self.feature(x_reshape).squeeze()
        # print('x_reshape0.shape',x_reshape.shape)
        x_reshape = x_reshape.permute(0,2,1)
        # print('x_reshape.shape',x_reshape.shape)
        return x_reshape

        


