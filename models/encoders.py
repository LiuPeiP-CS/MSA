#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/28 下午10:26
# @Author  : PeiP Liu
# @FileName: encoders.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.transformer.transformer import TransformerEncoder


class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2, bidirectional=False):
        super(LSTM_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, output_size)

    def forward(self, minf, lengths):
        packed_sequence = pack_padded_sequence(minf, lengths, batch_first=True, enforce_sorted=False) # enforce_sorted表示原始无序长度，系统进行排序;如果lengths的长度小于minf数据中的实际长度，那么会截取到lengths长度的内容
        packed_states, packed_final_state = self.lstm(packed_sequence)
        # 针对pack_padded_sequence和pad_packed_sequence的功能测试
        # import torch
        # import torch.nn as nn
        # from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        # input=torch.randn(5,7,10)
        # packed_sequence = pack_padded_sequence(input, [3,5,6,7,7], batch_first=True, enforce_sorted=False)
        # rnn = nn.LSTM(10, 20, 2, batch_first=True)
        # packed_h1, (fh1, fh2) = rnn(packed_sequence)
        # fh1.data.shape 输出：torch.Size([2, 5, 20]) 2表示双向，5表示batch, 20是维度
        # pad_h1, _ = pad_packed_sequence(packed_h1)
        # pad_h1.data.shape 输出：torch.Size([7, 5, 20]),产生了batch_size和seq_len的位置变换，因此需要permute(1,0,2)或者transpose(0,1)进行位置变换
        # 以上实验表明，在使用pad_X和pack_X时，即便rnn(batch_first=True),仍然会产生(seq_len, batch, dim)的输出

        padded_states, _ = pad_packed_sequence(packed_states)
        padded_states = padded_states.permute(1,0,2) # (batch_size, seq_len, output_size)

        h = self.dropout(packed_final_state[0].squeeze())
        y = self.linear_1(h) # (batch_size, output_size)
        return y, padded_states


class MLP(nn.Module):
    def __init__(self, d_hidden, bias=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_hidden, d_hidden, bias=bias)
        self.activation = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(d_hidden, d_hidden)

    # x: [bs, l, k, d] k=modalityKinds mask: [bs, l]
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return F.normalize(x, dim=1)


class Trans_Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Trans_Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states.mean(1) # 或者hidden_states.sum(1),在seq_len维度上进行pooling
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Trans_Encoder(nn.Module):
    def __init__(self, args, hidden_size, output_size, num_layers, dropout=0.2, relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0):
        super(Trans_Encoder, self).__init__()

        self.transformer = TransformerEncoder(hidden_size, args.transformer_head, num_layers, attention_dropout=dropout,
                                              relu_dropout=relu_dropout, res_dropout=res_dropout, embed_dropout=embed_dropout)

        # 将transformer的输出添加线性变换，从hid_dim到output_size
        # self.mlp = MLP(hidden_size, output_size)
        self.project_m = nn.Sequential()
        self.project_m.add_module('project_m', nn.Linear(in_features=hidden_size, out_features=output_size))
        self.project_m.add_module('project_m_activation', nn.ReLU())
        self.project_m.add_module('project_n_layer_norm', nn.LayerNorm(output_size))

        self.pooler = Trans_Pooler(output_size)

    def forward(self, x_q, x_k, x_v):
        # 传入数据形式是(batch_size, seq_len, dim),但transformer的需求数据形式是(seq_len, batch_size, dim).
        # 因此所有涉及到transformer编码器的内容，都在此处进行维度shape的变换
        x_q = x_q.permute(1,0,2)
        x_k = x_k.permute(1,0,2)
        x_v = x_v.permute(1,0,2)

        trans_result = self.transformer(x_q, x_k, x_v).transpose(0,1) # 转换成为(batch_size, seq_len, dim)的形式
        return trans_result

    def unimodal_feature(self, x, lengths=None):
        # x的输入尺寸为(batch_size, seq_len, dim)

        # 调用self.forward()函数，实现函数复用
        if lengths:
            packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        else:
            packed_sequence = x
        orig_last_layer_states = self.forward(packed_sequence, packed_sequence, packed_sequence) # (batch_size, lengths, output_size)
        tranfer_last_layer_states =  self.project_m(orig_last_layer_states)  # 经过了transformer和MLP的变换，得到的最终的结果，形式为(batch_size, seq_len, dim)
        # 对最后一层的所有状态输出进行池化，以及全序列的原始transformer_dim的输出(用于传给跨模态交互)
        return self.pooler(tranfer_last_layer_states), orig_last_layer_states # (batch, output_size), (batch, seq_len, transformer_dim)

    def multimodal_feature(self, x_q, x_k, x_v):
        last_layer_states = self.forward(x_q, x_k, x_v)  # (batch_size, lengths, output_size)
        # 对最后一层的所有状态输出进行池化
        return self.pooler(last_layer_states) # 获取跨模态attention的注意力向量表示
