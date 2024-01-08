#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/29 下午11:47
# @Author  : PeiP Liu
# @FileName: multihead_attention.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import sys

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout, bias=True, add_bias_kv=False, add_zero_attention=False):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'

        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim)) # 此处的3，表示的是q/k/v三块
        self.register_parameter('in_proj_bias', None)

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attention = add_zero_attention

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0) # 如果偏置非空，就置为0
            nn.init.constant_(self.out_proj.bias, 0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight) # kwargs实际上是个字典。当使用get进行键值获取时，能找到该key，就返回相应的value；如果找不到key,就返回默认值
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start: end, :]
        if bias is not None:
            bias = bias[start: end]
        return F.linear(input, weight, bias) # 实际上是input*weight^T,即得到大小为(start: end)的维度

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1) # 此处的chunk(3)，指的是分成q/k/v三个部分

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1) # 获取到[embed_dim:3*embed_dim]大小的特征，并同理于分割chunk(2)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2*self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def forward(self, query, key, value, attention_mask=None):
        """
        原始输入的数据中，shape=(seq_len, batch_size, feature_dim)
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr() # dara_ptr获取tensor张量的指针地址
        kv_same = key.data_ptr() == value.data_ptr()

        seq_len, batch_size, embed_dim = query.size() # 输入数据的维度非batch_size第一

        assert self.embed_dim == embed_dim # 初始化模型时的特征维度和输入数据的特征维度保证一致
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # the general attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q *= self.scaling # 进行根号下的归一计算

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, batch_size, 1)]) # 经过扩充后的bias_k大小为(1, batch_size, embed_dim)
            v = torch.cat([v, self.bias_v.repeat(1, batch_size, 1)])
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(attention_mask.size(0), 1)], dim=1) # 增补bias的attention mask

        q = q.contiguous().view(seq_len, batch_size*self.num_heads, self.head_dim).transpose(0,1) # 在维度角度对特征进行切分，并转换为batch_first

        if k is not None:
            k = k.contiguous().view(-1, batch_size*self.num_heads, self.head_dim).transpose(0,1)
        if v is not None:
            v = v.contiguous().view(-1, batch_size*self.num_heads, self.head_dim).transpose(0,1) # 对k/v进行特征维度的多头处理

        #######################################
        # 此处经过转换处理后，q/k/v的shape=(batch_size*num_heads, seq_len, feature_dim)

        kv_seq_len = k.size(1)

        if self.add_zero_attention:
            kv_seq_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1) # 构建了(batch_size, 1, embed_dim)的new_zeros向量，并和原向量进行串接
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(attention_mask.size(0), 1)], dim=1)

        attention_weight = torch.bmm(q, k.transpose(1, 2)) # shape=(batch_size*num_heads, q_seq_len, kv_seq_len)

        assert list(attention_weight.size() == [batch_size*self.num_heads, seq_len, kv_seq_len])

        if attention_mask is not None:
            try:
                attention_weight += attention_mask.unsqueeze(0) # attention_mask增添维度，成为(batch_size, 1, kv_seq_len)以便于和attention_weight相加
            except:
                print(attention_weight.shape)
                print(attention_mask.unsqueeze(0).shape)
                assert False

        attention_weight = F.softmax(attention_weight.float(), dim=-1).type_as(attention_weight)
        attention_weight = F.dropout(attention_weight, p=self.attention_dropout, training=self.training)

        attention = torch.bmm(attention_weight, v)
        assert list(attention.size()) == [batch_size*self.num_heads, seq_len, self.head_dim] # 此处的head_dim应该是k/v经过divide后的每个头的维度

        attention = attention.transpose(0, 1).contiguous().view(seq_len, batch_size, embed_dim) # 实现了seq_len处于第一位置
        attention = self.out_proj(attention)

        return attention # 获取的attention matrix

