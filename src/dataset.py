#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 下午8:52
# @Author  : PeiP Liu
# @FileName: dataset.py
# @Software: PyCharm
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class Multimodal_Dataset(Dataset):
    def __init__(self, dataset_path, data='mosei', split_type='train', align_sig=False):
        super(Multimodal_Dataset, self).__init__()

        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if align_sig else data+'_data_noalign.pkl')

        dataset = pickle.load(open(dataset_path, 'rb'))

        # these are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)

        self.audio[self.audio == -np.inf] = 0

        self.audio = torch.tensor(self.audio).cpu().detach()

        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None


        self.data = data

        self.n_modalities = 3

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        # the seq_len, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        # the batch_size
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]

        META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])

        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)

        return X, Y, META