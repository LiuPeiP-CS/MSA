#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 下午10:40
# @Author  : PeiP Liu
# @FileName: mainModel.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal, xavier_uniform, orthogonal

from models.alignNet import AlignNet
from models.mainModel import CLMSA

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.need_model_aligned = args.need_model_aligned
        # simulating word_aligned network (for seq_len_text == seq_len_Audio == seq_len_Vision)
        if self.need_model_aligned:
            self.alignNet = AlignNet(args)
            args.seq_lens = self.alignNet.aligned_length # 在对齐的情况下，以文本长度作为所有的默认长度

        self.model = CLMSA(args)

    def forward(self, text, audio, vision, multi_labels, text_labels=None, audio_labels=None, vision_labels=None):
        if self.need_model_aligned:
            text, audio, vision = self.alignNet(text, audio[0], vision[0]) # forward()接受来自train.do_train()中model的参数，其中audio和vision包含了数据长度
        if text_labels: # 此处的选项是sims数据集
            return self.model(text, audio, vision, multi_labels, text_labels, audio_labels, vision_labels)
        else:
            return self.model(text, audio, vision, multi_labels)

