#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 下午4:43
# @Author  : PeiP Liu
# @FileName: alignNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from src.ctc import CTCModule

class AlignNet(nn.Module):
    def __init__(self, args):
        super(AlignNet, self).__init__()
        inp_dim_t, inp_dim_a, inp_dim_v = args.feature_dim
        seq_len_t, seq_len_a, seq_len_v = args.seq_lens

        self.aligned_length = seq_len_t

        self.aligned_mode = args.model_aligned_mode

        self.aligned_way = {
            'avg_pool': self.__avg_pool,
            'ctc': self.__ctc,
            'convnet': self.__convnet
        }

        if self.aligned_mode == 'convnet':
            self.Conv1d_T = nn.Conv1d(seq_len_t, self.aligned_length, kernel_size=1, bias=False) # input_dim, output_dim
            self.Conv1d_A = nn.Conv1d(seq_len_a, self.aligned_length, kernel_size=1, bias=False)
            self.Conv1d_V = nn.Conv1d(seq_len_v, self.aligned_length, kernel_size=1, bias=False)

        elif self.aligned_mode == 'ctc':
            self.ctc_t = CTCModule(inp_dim_t, self.aligned_length)
            self.ctc_a = CTCModule(inp_dim_a, self.aligned_length)
            self.ctc_v = CTCModule(inp_dim_v, self.aligned_length)

    def __avg_pool(self, text, audio, vision):
        def align(minf):
            mlength = minf.size(1)
            if mlength == self.aligned_length: # 如果原始模态信息的长度已经是对齐的，那么就不需要进行对齐操作
                return minf
            elif mlength // self.aligned_length == mlength / self.aligned_length:
                pad_len = 0
                pool_size = mlength // self.aligned_length # 能整除的情况下，那么pool的尺寸就是商
            else:
                pad_len = self.aligned_length - mlength % self.aligned_length
                pool_size = mlength // self.aligned_length + 1

            pad_inf = minf[:, -1, :].unsqueeze(1).expand([minf.size(0), pad_len, minf.size(-1)]) # (num_samples, feature_dim)———> (num_samples, 1, feature_dim),补充的信息是最后一个特征向量
            minf = torch.cat([minf, pad_inf], dim=1).view(minf.size(0), pool_size, self.aligned_length, -1)
            minf = minf.mean(dim=1)
            return minf

        text = align(text)
        audio = align(audio)
        vision = align(vision)

        return text, audio, vision

    def __ctc(self, text, audio, vision):
        text = self.ctc_t(text) if text.size(1) != self.aligned_length else text
        audio = self.ctc_a(audio) if audio.size(1) != self.aligned_length else audio
        vision = self.ctc_v(vision) if vision.size(1) != self.aligned_length else vision
        return text, audio, vision

    def __convnet(self, text, audio, vision):
        text = self.Conv1d_T(text) if text.size(1) != self.aligned_length else text
        audio = self.Conv1d_A(audio) if audio.size(1) != self.aligned_length else audio
        vision = self.Conv1d_V(vision) if vision.size(1) != self.aligned_length else vision
        return text, audio, vision

    def forward(self, text, audio, vision):
        # already aligned
        if text.size(1) == audio.size(1) == vision.size(1):
            return text, audio, vision
        
        return self.aligned_way[self.aligned_mode](text, audio, vision)
