#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 上午10:48
# @Author  : PeiP Liu
# @FileName: LoadData.py
# @Software: PyCharm

import os
import logging
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('MSA')

class MSADataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        Data_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        Data_MAP[args.datasetName]() # 初始化处理数据集，数据集的名称要求在main函数中

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f: # 把数据集中的数据信息load出来,以下示例是由sims数据的unaligned_39.pkl进行展示
            data = pickle.load(f) #　
        if self.args.use_bert: # 本文的实验中，默认使用bert
            self.text = data[self.mode]['text_bert'].astype(np.float32) # 该数据的维度是(1368, 3, 39),其中是1368是指数据数量，3-0是word-index,3-1是mask,3-2是seq_index即全0(区别于1)
        else:
            self.text = data[self.mode]['text'].astype(np.float32) # (1368, 39, 768)，可以认为是提取好的特征

        self.vision = data[self.mode]['vision'].astype(np.float32) # (1368, 55, 709),最大长度为55，特征维度为709
        self.audio = data[self.mode]['audio'].astype(np.float32) # (1368, 400, 33),按照最大长度400进行特征获取
        self.rawText = data[self.mode]['raw_text'] # list
        self.ids = data[self.mode]['id'] # list
        self.labels = {
            "M": data[self.mode][self.args.train_mode+'_labels'].astype(np.float32) # (1368,)
        }

        if self.args.datasetName == 'sims':
            for m in 'TAV':
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m] # (1368,)

        logger.info('{} samples: {}'.format(self.mode, self.labels['M'].shape))

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths'] # 每条数据的真实长度
            self.vision_lengths = data[self.mode]['vision_lengths'] # 每条数据的真实长度

        self.audio[self.audio == -np.inf] = 0 # 将原始特征为-np.inf的位置转换为0

        if self.args.need_normalized: # 该标准化方法简单粗暴，就是平均求值。一般不需要标准化
            self.__normalize()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __normalize(self):
        # 该函数在实际使用中，其实没必要
        # (num_examples, max_len, feature_dim) ——> (max_len, num_samples, feature_dim)
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.transpose(self.vision, (1, 0, 2))

        # for visual and audio modality, we average across time
        # i.e., we transfer the original data from (max_len, num_samples, feature_dim) to (1, num_samples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # 将所有为nan的值替换为0
        self.vision[self.vision!=self.vision] = 0 # Nan自己不等于自己，因而可以判断该值是否为nan(即!=为True时).
        self.audio[self.audio!=self.audio] = 0

        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.transpose(self.vision, (1, 0, 2))

    def get_seq_len(self):
    # 返回各模态信息的长度
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1]) # 此处的audio及vision信息保证不经过normalize
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])  # 此处的audio及vision信息保证不经过normalize

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index], # 原始文本信息
            'text': torch.tensor(self.text[index]), # text的特征信息。对于use_bert,是(3, 39);对于非use_bert,是(39, 768)
            'audio': torch.tensor(self.audio[index]), # audio的特征信息,(400, 33)
            'vision': torch.tensor(self.vision[index]), # vision的特征信息,(55, 709)
            'index': index, # 第几条数据
            'id': self.ids[index], # 数据的唯一标识id
            'labels': {k: torch.tensor(v[index].reshape(-1)) for k, v in self.labels.items()} # reshape(-1)的操作实际没有必要.'labels'的value是一个字典，{'M':X, 'A':X, 'T':X, 'V':X}
        }

        if not self.args.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index] # 获取每个样本数据的真实长度
            sample['vision_lengths'] = self.vision_lengths[index]

        return sample

def MSADataLoader(args):
    datasets = {
        'train': MSADataset(args, 'train'),
        'valid': MSADataset(args, 'valid'),
        'test': MSADataset(args, 'test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()

    data_Loader = {
        dsl: DataLoader( # 默认需要执行__getitem__()
            datasets[dsl], # dsl是train/valid/test中的一个,此处得到的结果是MSADataset对象
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        ) for dsl in datasets.keys()
    }

    return data_Loader
