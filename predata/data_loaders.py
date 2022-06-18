import torch
from torch.utils.data import Dataset
import os
import numpy as np
from dataset.data_psd import PSD
from dataset.data_representation import EEG_Spectral_spatial_representation
class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset,SS,seq_len=None,chs=None):
        super(LoadDataset_from_numpy, self).__init__()
        self.chs=chs
        self.SS=SS
        # load files
        X_train = np.load(np_dataset[0])["x"]
        if self.SS==11 or self.SS==12:
            ss_train = np.load(np_dataset[0])["ss"]
        # print(X_train.shape)
        if self.SS==8 or self.SS==9 or self.SS == 10 or self.SS == 11: # isruc          
            y_train = np.argmax(np.load(np_dataset[0])["y"],axis=1)
        else:
            y_train = np.load(np_dataset[0])["y"]
        # print(y_train.shape)

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            if self.SS==11 or self.SS==12:
                ss_train = np.vstack((ss_train, np.load(np_file)["ss"]))
            if self.SS==8 or self.SS==9 or self.SS == 10 or self.SS == 11 :
                y_train = np.append(y_train, np.argmax(np.load(np_file)["y"],axis=1))
            else:
                y_train = np.append(y_train, np.load(np_file)["y"])
            # print(y_train.shape)

        
        self.x_data = torch.from_numpy(X_train)
        if self.SS==11 or self.SS == 12:
            self.ss_data = torch.from_numpy(ss_train)
        self.y_data = torch.from_numpy(y_train).long()
        # print('self.x_data.shape',self.x_data.shape)
        # print('self.y_data.shape',self.y_data.shape)
        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        
        if len(self.x_data.shape) == 3 :
            if self.x_data.shape[1] != 1 and self.SS!=8 and self.SS!=9 and self.SS!=10 and self.SS!=11 :
                self.x_data = self.x_data.permute(0, 2, 1)
                # print('self.x_data.shape1',self.x_data.shape)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        # print('self.x_data2.shape',self.x_data.shape)

        if self.SS == 8:
            self.len = self.x_data.shape[0]
            self.X = [[],[],[]]
            self.X[2] = self.x_data[:,7,:]
            self.X[0] = self.x_data[:,0:6,:]
            self.X[1] = self.x_data[:,6,:]
            print('self.X.shape:',self.X[0].shape)
            # print('self.X.shape:',self.X.shape)
        elif self.SS == 9:#attn_isruc  EEG(F3-A2/C3-A2/O1-A2/F4-A1/C4-A1/O2-A1)
            # print('self.x_data.shape2',self.x_data.shape)
            self.len = self.x_data.shape[0]
            self.X = []
            self.X = self.x_data[:,0,:]
            # self.X = self.X.unsqueeze(dim =1)
            print('self.X.shape:',self.X.shape)


        elif self.SS == 6:
            
            self.seq_len=seq_len
            self.x_data_reshape = []
            self.y_data_reshape = []
            self.x_data_reshape = torch.Tensor(self.x_data_reshape)
            self.y_data_reshape = torch.Tensor(self.y_data_reshape)
            # self.x_data_reshape = self.x_data_reshape[:,np.newaxis,np.newaxis]
            # print('self.x_data_reshape.shape',self.x_data_reshape.shape) 
            # print(self.x_data.shape[0]//20)    

            self.x_data = self.x_data[0:(self.x_data.shape[0]//self.seq_len)*self.seq_len,:,:]
            self.y_data = self.y_data[0:(self.y_data.shape[0]//self.seq_len)*self.seq_len]
            # print(self.x_data[62])
            self.x_data = self.x_data.reshape(-1,self.seq_len,4,3000)
            # self.x_data = self.x_data.permute(0,2,1,3)
            # self.x_data = self.x_data.reshape(-1,4,60000)
            
            # print('x_data.shape:',self.x_data.shape)
            # print('self.y_data',self.y_data[120:180])
            self.y_data = self.y_data.reshape(-1,self.seq_len)
            self.len = self.x_data.shape[0]
            # if self.len % 8 == 1:
            #     del self.x_data[-1,:,:]
            # print('x_data.shape:',self.x_data.shape)
            if chs==3:
                self.X = [[],[]]
                self.X[0] = self.x_data[:,:,0:2,:]
                self.X[1] = self.x_data[:,:,2,:]
            elif chs==4:
                self.X = [[],[],[]]
                self.X[0] = self.x_data[:,:,0:2,:]
                self.X[1] = self.x_data[:,:,2,:]
                self.X[2] = self.x_data[:,:,3,:]
            # print(self.X[0].shape)
            # print(self.X[1].shape)

        elif self.SS == 10:
            self.x_data_reshape = []
            self.y_data_reshape = []
            self.x_data_reshape = torch.Tensor(self.x_data_reshape)
            self.y_data_reshape = torch.Tensor(self.y_data_reshape)
            print('self.x_data.shape',self.x_data.shape)
            # self.x_data_reshape = self.x_data_reshape[:,np.newaxis,np.newaxis]
            # print('self.x_data_reshape.shape',self.x_data_reshape.shape) 
            # print(self.x_data.shape[0]//20)    

            self.x_data = self.x_data[0:(self.x_data.shape[0]//20)*20,:,:]
            self.y_data = self.y_data[0:(self.y_data.shape[0]//20)*20]
            # print(self.x_data[62])
            self.x_data = self.x_data.reshape(-1,20,8,3000)
            self.x_data = self.x_data.permute(0,2,1,3)
            self.x_data = self.x_data.reshape(-1,8,60000)
            
            # print('x_data.shape:',self.x_data.shape)
            # print('self.y_data',self.y_data[120:180])
            self.y_data = self.y_data.reshape(-1,20)


            self.len = self.x_data.shape[0]
            # if self.len % 8 == 1:
            #     del self.x_data[-1,:,:]
            # print('x_data.shape:',self.x_data.shape)
            print('self.x_data.shape',self.x_data.shape)
            self.X = [[],[]]
            self.X[0] = self.x_data[:,0:5,:]
            self.X[1] = self.x_data[:,6,:]

        elif self.SS == 5 :
            self.x_data_reshape = []
            self.y_data_reshape = []
            self.x_data_reshape = torch.Tensor(self.x_data_reshape)
            self.y_data_reshape = torch.Tensor(self.y_data_reshape)
            print('self.x_data.shape',self.x_data.shape)
            # self.x_data_reshape = self.x_data_reshape[:,np.newaxis,np.newaxis]
            # print('self.x_data_reshape.shape',self.x_data_reshape.shape) 
            # print(self.x_data.shape[0]//20)    

            self.x_data = self.x_data[0:(self.x_data.shape[0]//20)*20,:,:]
            self.y_data = self.y_data[0:(self.y_data.shape[0]//20)*20]
            # print(self.x_data[62])
            self.x_data = self.x_data.reshape(-1,20,4,3000)
            self.x_data = self.x_data.permute(0,2,1,3)
            self.x_data = self.x_data.reshape(-1,4,60000)
            
            # print('x_data.shape:',self.x_data.shape)
            # print('self.y_data',self.y_data[120:180])
            self.y_data = self.y_data.reshape(-1,20)


            self.len = self.x_data.shape[0]
            # if self.len % 8 == 1:
            #     del self.x_data[-1,:,:]
            print('x_data.shape:',self.x_data.shape)
            self.X = [[],[]]
            self.X[0] = self.x_data[:,0:2,:]
            self.X[1] = self.x_data[:,2,:]
  
        # PSD_EEG + 2*EEG(FPz-Cz/Pz-Oz) + EOG + EMG
        elif self.SS==4:
            self.len = self.x_data.shape[0]
            self.X = [[],[],[],[]]
            self.X[3] = np.concatenate((PSD(np.array(self.x_data[:,0,:])),PSD(np.array(self.x_data[:,1,:]))),axis=1)
            self.X[2] = self.x_data[:,3,:]
            self.X[3] = torch.from_numpy(self.X[3])
            self.X[0] = self.x_data[:,0:2,:]
            self.X[1] = self.x_data[:,2,:]
            # print(self.X[3].shape)

        elif self.SS==11 : #

            self.len = self.x_data.shape[0]
            self.X = [[],[],[],[]]
            self.X[3] = self.ss_data
            self.X[2] = self.x_data[:,7,:]
            # self.X[3] = torch.from_numpy(self.X[3])
            self.X[0] = self.x_data[:,0:6,:]
            self.X[1] = self.x_data[:,6,:]
        elif self.SS==12 : #

            self.len = self.x_data.shape[0]
            self.X = [[],[],[],[]]
            self.X[3] = self.ss_data
            self.X[2] = self.x_data[:,3,:]
            # self.X[3] = torch.from_numpy(self.X[3])
            self.X[0] = self.x_data[:,0:2,:]
            self.X[1] = self.x_data[:,2,:]
        # 2*EEG(FPz-Cz/Pz-Oz)+EOG+EMG
        elif self.SS == 3:
            self.len = self.x_data.shape[0]
            self.X = [[],[],[]]
            self.X[2] = self.x_data[:,3,:]
            self.X[0] = self.x_data[:,0:2,:]
            self.X[1] = self.x_data[:,2,:]
        
        # 2*EEG(FPz-Cz/Pz-Oz)+EOG
        elif self.SS == 2:
            self.len = self.x_data.shape[0]
            self.X = [[],[]]
            self.X[0] = self.x_data[:,0:2,:]
            self.X[1] = self.x_data[:,2,:]

        # EEG(FPz-Cz/Pz-Oz)
        elif self.SS == 1:
            self.len = self.x_data.shape[0]
            self.X = []
            self.X = self.x_data[:,0,:]

    
    def __getitem__(self, index):
        if self.SS==4 or self.SS==11 or self.SS==12:
            return self.X[0][index], self.X[1][index],self.X[2][index],self.X[3][index],self.y_data[index]
        elif self.SS == 3 or self.SS==8 :
            return self.X[0][index], self.X[1][index],self.X[2][index],self.y_data[index]
        elif self.SS == 2 or self.SS == 5 or self.SS == 10:
            return self.X[0][index], self.X[1][index],self.y_data[index]
        elif self.SS == 6:
            if self.chs==3:
                return self.X[0][index], self.X[1][index],self.y_data[index]
            if self.chs==4:
                return self.X[0][index], self.X[1][index], self.X[2][index],self.y_data[index]
        elif self.SS == 1 or self.SS == 9:
            return self.X[index], self.y_data[index]
    def __len__(self):
        return self.len


def data_generator_np(training_files, subject_files, batch_size,SS,chs=None,seq_len = None):

    
    train_dataset = LoadDataset_from_numpy(training_files,SS,seq_len,chs)
    test_dataset = LoadDataset_from_numpy(subject_files,SS,seq_len,chs)

    # to calculate the ratio for the CAL
    # print(train_dataset.X[0].shape)
    # print(test_dataset.X[0].shape)
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts


def dataload_test( subject_files, batch_size,SS):

    test_dataset = LoadDataset_from_numpy(subject_files,SS)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return test_loader


