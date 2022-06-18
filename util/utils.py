import os
import sys
import time
import math
import numpy as np

from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def preprocessing(x):
    # x [bs, seq_len, ch, 30*100]
    # return [bs, seq_len, ch, 29, 129]
    bslist = []
    for b in range(x.shape[0]):
        seqlist = []
        for s in range(x.shape[1]):
            chlist = []
            for c in range(x.shape[2]):
                tmp = stft(x[b,s,c,:])
                tmp = torch.from_numpy(tmp)
                tmp = torch.unsqueeze(tmp, 0)
                chlist.append(tmp)
            chout = torch.cat(chlist, 0) # [ch, 29, 129]
            chout = torch.unsqueeze(chout, 0) # [1, ch, 29, 129]
            seqlist.append(chout)
        seqout = torch.cat(seqlist, 0)
        seqout = seqout.unsqueeze(0)
        bslist.append(seqout)
    out = torch.cat(bslist, 0) # out [bs, seq_len, 29, 129]
    out = out.type(torch.float)
    return out

import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
import math
from glob import glob


def load_folds_data(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    print(files)
    if "78" in np_data_path:
        r_p_path = r"util/sleep-edf-78.npy"
    elif "20" in np_data_path:
        r_p_path = r"util/sleep-edf-78.npy"
    else:
        r_p_path = r"util/isruc-sleep-3.npy"
    
    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
        print(r_permute)
        # r_permute = np.array(range(20))
        # print(r_permute.shape)
    else:
        print ("============== ERROR =================")


    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs,dtype=object)
    print("files_pairs",files_pairs.shape)
    files_pairs = files_pairs[r_permute]
    # print("files_pairs",files_pairs)

    train_files = np.array_split(files_pairs, n_folds)
    print("train_files",train_files[0])

    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        # print("files_pairs2",files_pairs2)
        # print("----------------------------------------------------------------------")
        # print("subject_files",subject_files)
        # print(set(files_pairs2) - set(subject_files))
        # training_files = [i for i in files_pairs2 if i not in subject_files]
        training_files = list(set(files_pairs2) - set(subject_files))
        # print("training_files:",training_files)
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data,r_permute

def load_folds_data_isruc(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"util/r_permute_isruc.npy"
    
    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
        print(r_permute)
        # r_permute = np.array(range(20))
        # print(r_permute.shape)
    else:
        print ("============== ERROR =================")


    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[0:-3]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs,dtype=object)
    print("files_pairs",files_pairs.shape)
    files_pairs = files_pairs[r_permute]
    # print("files_pairs",files_pairs)

    train_files = np.array_split(files_pairs, n_folds)
    print("train_files",train_files[0])

    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        # print("files_pairs2",files_pairs2)
        # print("----------------------------------------------------------------------")
        # print("subject_files",subject_files)
        # print(set(files_pairs2) - set(subject_files))
        # training_files = [i for i in files_pairs2 if i not in subject_files]
        training_files = list(set(files_pairs2) - set(subject_files))
        # print("training_files:",training_files)
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data,r_permute


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
