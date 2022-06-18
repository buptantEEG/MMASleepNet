import argparse
import collections
import sys 
sys.path.append("..") 
sys.path.append("...") 
# Leave One Subject Out
from predata.SleepDataset import SleepDataset
from predata.data_loaders import LoadDataset_from_numpy,data_generator_np
from util import writer_func
from util.train_test import test_3ch, test_4ch, test_mul, train, test, train_3ch, train_4ch, train_mul
from util.func import standard_scaler, normalizer, min_max_scaler, max_abs_scaler

import model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import ConcatDataset

from torch.utils.tensorboard import SummaryWriter
import os
from util.utils import *



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(0)
print("Using {} device".format(device))


# fix random seeds for reproducibility
SEED = 24
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)



class Config():
    def __init__(self,  
                # channels
                fold_id,
                subject,
                SS,
                subjects_num
                ):

        self.scaler = None
        self.batch_size=256
        self.afr_reduced_cnn_size=3
        self.d_model = 72
        self.subjects_num=subjects_num
        #transformer
        self.ch=SS+1
        self.nhead=4
        self.num_layers=1
        self.inplanes=3
        self.subject=subject
        self.SS=SS
        self.fold_id = fold_id
        self.lr = 0.0001
        self.epochs = 250
        # Model
        self.model=model.MMASleepNet_eegeog_plus(self).to(device)
        # self.models = [model.MAttnSleep().to(device) for i in range(len(self.channels))]
        # self.model = model.EEGNet(C=8, T=30*128)
        
        # self.optimizer = torch.optim.SGD
        # self.optimizer = torch.optim.Adam([{"params":model.parameters()} for model in self.models], lr=self.lr, weight_decay=0.01, amsgrad=True)
        
        self.loss_fn=nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 1.80, 1.0, 1.25, 1.20]).to(device=device))
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10], gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 10, last_epoch=-1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01, amsgrad=True)
        # self.Dataset = SleepAdult(data_path=self.data_path, downsample=128)
        # self.comment = f'_subject{subject}' # comment in the save dir name
        self.output_dir = f'ckpt/MMASleep-Ablation2/sleepedf-{self.subjects_num}/fold{self.fold_id}_subject{self.subject}/{self.model.init_time}'

def main(fold_id,chs,subjects_num):
    print(f"------------------------------------FOLD_{fold_id}------------------------------------")
    cfg = Config(fold_id=fold_id,subject=subjects[fold_id],SS=chs-1,subjects_num=subjects_num)
    # batch_size = config["data_loader"]["args"]["batch_size"]

    # logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    net = cfg.model
    # loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 1.80, 1.0, 1.25, 1.20]).to(device=device))
    loss_fn = cfg.loss_fn 
    optimizer=cfg.optimizer
    # model.apply(weights_init_normal)
    print("train_files:",folds_data[fold_id][0])
    print("test_files:",folds_data[fold_id][1])
    train_loader, test_loader, counts = data_generator_np(folds_data[fold_id][0],
                                                                   folds_data[fold_id][1], cfg.batch_size,SS=cfg.SS)


    writer = SummaryWriter(log_dir=cfg.output_dir)
    writer_func.save_config(writer, cfg)

    start_time = time.time()
    best_acc = 0
    for t in range(cfg.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if chs==3:
            train_loss, train_acc = train_3ch(train_loader, net, loss_fn, optimizer)
            test_loss,test_acc = test_3ch(test_loader, net, loss_fn)
        elif chs==4:
            train_loss, train_acc = train_4ch(train_loader, net, loss_fn, optimizer)
            test_loss,test_acc = test_4ch(test_loader, net, loss_fn)
        if cfg.scheduler is not None:
            cfg.scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            # for i,net in enumerate(net):
            net.save(cfg.output_dir)
        writer_func.save_scalar(writer, {'Loss/train':train_loss,'Accuracy/train':train_acc,'Loss/test':test_loss, 'Accuracy/test':test_acc}, t)
    end_time = time.time()
    print(f'best_acc {best_acc}, time {(end_time-start_time)/60}min')

    writer_func.save_text(writer, {'best_acc':str(best_acc)})
    writer_func.save_text(writer, {'time':str((end_time-start_time)/60)})
    writer.close()

    print("Done!")
    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-f', '--fold_id', type=str,
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files')
    args.add_argument('-ch', '--channel_number', type=str,
                      help='the number of channels')


    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)
    chs = int(args2.channel_number)
    # config = ConfigParser.from_args(args, fold_id, options)
    if "isruc" in args2.np_data_dir:
        folds_data,subjects = load_folds_data_shhs(args2.np_data_dir, 10)
        subjects_num=10
    elif  "78" in  args2.np_data_dir:
        folds_data,subjects = load_folds_data(args2.np_data_dir, 10)
        subjects_num=78
    elif  "20" in  args2.np_data_dir:
        folds_data,subjects = load_folds_data(args2.np_data_dir, 20)
        subjects_num=20
    # print(len(folds_data[0][0]))
    # print(len(folds_data[0][1]))
    
    main(fold_id,chs,subjects_num)