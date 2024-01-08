#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/29 下午7:51
# @Author  : PeiP Liu
# @FileName: transformer.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.position_embedding import SinuPositionEmbedding
from models.transformer.multihead_attention import MultiheadAttention
import math

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, num_heads, layers, attention_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0, attention_mask=True):
        super(TransformerEncoder, self).__init__()
        self.dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.feature_dim = feature_dim
        self.feature_scale = math.sqrt(feature_dim)
        self.feature_positions = SinuPositionEmbedding(feature_dim)

        self.attention_mask = attention_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(feature_dim, num_heads, attention_dropout, relu_dropout, res_dropout, attention_mask)
            self.layers.append(new_layer)

        self.layer_norm = LayerNorm(self.feature_dim)


    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        :param x_in/x_in_k/x_in_v: shape为(seq_len, batch_size, feature_dim)
        :return: the last encoder layer's output with shape of (seq_len, batch_size, embed_dim)
        """
        x = self.feature_scale * x_in
        if self.feature_positions is not None:
            x += self.feature_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1) # adding the positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.feature_scale * x_in_k
            x_v = self.feature_scale * x_in_v
            if self.feature_positions is not None:
                x_k += self.feature_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_v += self.feature_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)

        return self.layer_norm(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, feature_dim, num_heads=4, attention_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attention_mask=False):
        super(TransformerEncoderLayer, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.self_attention = MultiheadAttention(
            self.feature_dim, self.num_heads, attention_dropout
        )

        self.attention_mask = attention_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.feature_dim, 4*self.feature_dim) # The 'Add & Norm' part in the paper
        self.fc2 = Linear(4*self.feature_dim, self.feature_dim)

        self.layer_norms = nn.ModuleList([LayerNorm(self.feature_dim) for _ in range(2)])

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after # ^运算，相同为0，不同为1
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True) # 第一次执行LayerNorm
        mask = buffered_feature_mask(x, x_k) if self.attention_mask else None # 处理attention_mask,此处mask的获取方式存在疑问
        if x_k is None and x_v is None:
            x = self.self_attention(query=x, key=x, value=x, attention_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x = self.self_attention(query=x, key=x_k, value=x_v, attention_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

def fill_with_neg_inf(tensor_ones):
    return tensor_ones.float().fill_(float('-inf')).type_as(tensor_ones)

def buffered_feature_mask(tensor1, tensor2=None):
    dim1 = dim2 = tensor1.size(0) # 此处的0表示sequence length,即非batch_size=first
    if tensor2 is not None:
        dim2 = tensor2.size(0)

    feature_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1)) # triu()函数的第二个参数以(左上角－右下角=0)为边界,详见https://blog.csdn.net/chengyq116/article/details/106877146
    if tensor1.is_cuda:
        feature_mask = feature_mask.to(tensor1.device)
    return feature_mask[:dim1, :dim2]

def Linear(input_dim, output_dim, bias=True):
    m = nn.Linear(input_dim, output_dim, bias)
    nn.init.xavier_uniform_(m.weight) # 对其权值进行均值初始化
    if bias:
        nn.init.constant_(m.bias, 0) # 使偏置为常数0
    return m # 返回一个函数类型的神经网络层

def LayerNorm(feature_dim):
    m = nn.LayerNorm(feature_dim)
    return m
