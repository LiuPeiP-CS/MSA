#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 下午8:08
# @Author  : PeiP Liu
# @FileName: ctc.py
# @Software: PyCharm

import torch
from torch import nn

class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        """
        this module is performing alignment from A modality to B modality
        :param in_dim: dimension for the input modality A
        :param out_seq_len: sequence length for the output modality B
        """
        super(CTCModule, self).__init__()

        # using LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True)

        self.out_seq_len = out_seq_len

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        :param x: the input with a shape of [batch_size, in_seq_len, in_dim]
        :return:
        """
        pred_out_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_out_position_inclu_blank = self.softmax(pred_out_position_inclu_blank) #[batch_size, in_seq_len, out_seq_len+1]

        prob_pred_out_position = prob_pred_out_position_inclu_blank[:, :, 1:] # [batch_size, in_seq_len, out_seq_len]
        prob_pred_out_position = prob_pred_out_position.transpose(1, 2) # [batch_size, out_seq_len, in_seq_len]
        pseudo_aligned_out = torch.bmm(prob_pred_out_position, x) # [batch_size, out_seq_len, in_dim]

        return pseudo_aligned_out, (pred_out_position_inclu_blank)