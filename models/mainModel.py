#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 下午10:44
# @Author  : PeiP Liu
# @FileName: mainModel.py
# @Software: PyCharm
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.BertTextEncoder import BertTextEncoder
from models.encoders import LSTM_Encoder, Trans_Encoder, MLP
from models.transformer.multihead_attention import MultiheadAttention
from src.contrastive import SCL_SupConLoss, SSCL_PairConLoss

class CLMSA(nn.Module):
    def __init__(self, args):
        super(CLMSA, self).__init__()
        self.aligned = args.need_data_aligned # 默认该值为false
        self.lstm_encoder = args.lstm_encoder # 是否选择lstm编码器
        output_dim = args.num_classes if args.train_mode == 'classification' else 1
        text_input_dim, audio_input_dim, vision_input_dim = args.feature_dims

        self.text_bert_encoder = BertTextEncoder(args.bert_root, language=args.language, use_finetune = args.use_finetune)
        self.align_t_transformer_dim = nn.Linear(text_input_dim, args.transformer_dim)
        if args.lstm_encoder:
            self.audio_lstm_encoder = LSTM_Encoder(audio_input_dim, args.a_lstm_hidden_size, args.audio_out, args.a_lstm_layers, args.a_lstm_dropout)
            self.vision_lstm_encoder = LSTM_Encoder(vision_input_dim, args.v_lstm_hidden_size, args.video_out, args.v_lstm_layers, args.v_lstm_dropout)
            self.align_a_transformer_dim = nn.Linear(args.audio_out, args.transformer_dim)
            self.align_v_transformer_dim = nn.Linear(args.video_out, args.transformer_dim) # 将特征转换成为transformer维度，供下一级进行跨模态的transformer学习
        else:
            # 在数据预处理阶段，添加conv1d层操作
            self.conv_a = nn.Conv1d(in_channels=audio_input_dim, out_channels=args.transformer_dim, kernel_size=args.conv1d_kernel_size_a, stride=1, padding=1) # 此处的设计保证了原始特征维度的转换
            self.conv_v = nn.Conv1d(in_channels=vision_input_dim, out_channels=args.transformer_dim, kernel_size=args.conv1d_kernel_size_v, stride=1, padding=1)

            self.audio_trans_encoder = Trans_Encoder(args, args.transformer_dim, args.audio_out, args.unimodal_trans_levels,
                                                     dropout=args.atten_dropout_a, relu_dropout=args.relu_dropout,
                                                     res_dropout=args.res_dropout, embed_dropout=args.embed_dropout)
            self.vision_trans_encoder = Trans_Encoder(args, args.transformer_dim, args.video_out, args.unimodal_trans_levels,
                                                      dropout=args.atten_dropout_v, relu_dropout=args.relu_dropout,
                                                      res_dropout=args.res_dropout, embed_dropout=args.embed_dropout)

        # 以下是单模态的输出后拟合变换
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, output_dim)

        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, output_dim)

        self.post_vision_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_vision_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_vision_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_vision_layer_3 = nn.Linear(args.post_video_dim, output_dim)

        # 在监督聚类前进行适当的操作，面向的维度是x_dim(即单模态的输出维度)，映射器projection
        self.mlp_t = MLP(text_input_dim)
        self.mlp_a = MLP(args.audio_out)
        self.mlp_v = MLP(args.video_out)

        self.scl_t = SCL_SupConLoss()
        self.scl_a = SCL_SupConLoss()
        self.scl_v = SCL_SupConLoss()

        #**************************************************************************************************************#
        #* 以上都属于单模态信息的编码和转换(无论什么编码方式，最后的输出都是面向单模态的维度)，接下来的操作是跨模态
        #**************************************************************************************************************#
        self.args = args
        self.modal_based = args.modal_based # 是最先还是最后0/1
        self.aligned_dim = args.transformer_dim # 将transformer输出的特征维度，作为我们特征传递的主要维度
        self.cross_transformer_levels = args.multimodal_trans_levels
        self.atten_dropout = args.atten_dropout
        # 设置三种模态的相同维度
        # 设置相应的模态setting
        self.transformer_ij = self.get_cross_transformer()
        self.transformer_ji = self.get_cross_transformer()
        self.transformer_ki = self.get_cross_transformer()
        self.transformer_ik = self.get_cross_transformer()
        self.transformer_jk = self.get_cross_transformer()
        self.transformer_kj = self.get_cross_transformer()

        self.transformer_ijk = self.get_cross_transformer()
        self.transformer_jik = self.get_cross_transformer()
        self.transformer_kij = self.get_cross_transformer()
        self.transformer_ikj = self.get_cross_transformer()
        self.transformer_jki = self.get_cross_transformer()
        self.transformer_kji = self.get_cross_transformer()

        # 在进行对比之前，需要适当的projection
        self.ijk_mlp = MLP(args.transformer_dim)
        self.jik_mlp = MLP(args.transformer_dim)
        self.kij_mlp = MLP(args.transformer_dim)
        self.ikj_mlp = MLP(args.transformer_dim)
        self.jki_mlp = MLP(args.transformer_dim)
        self.kji_mlp = MLP(args.transformer_dim)

        self.self_cl_i = SSCL_PairConLoss()
        self.self_cl_j = SSCL_PairConLoss()
        self.self_cl_k = SSCL_PairConLoss()

        #**************************************************************************************************************#
        #* 以上都属于跨模态信息的编码和转换(无论什么编码方式，最后的输出都是(batch_size, transformer_dim)的维度)，接下来的操作是融合模态结果
        #**************************************************************************************************************#
        # self-attention之后，进行线性变换
        self.fusion_encoder = Trans_Encoder(self.args, 6*self.aligned_dim, self.aligned_dim, 2) # input_dim is 6*self.aligned_dim, and output's dim is self.aligned_dim

        # fusion layer
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(self.aligned_dim, self.aligned_dim) # 编码后的特征维度转换
        self.post_fusion_layer_2 = nn.Linear(self.aligned_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, output_dim) # 适用于拟合任务

        self.fusion_mlp = MLP(self.aligned_dim)
        self.scl_fusion = SCL_SupConLoss()

    def get_cross_transformer(self):
        return Trans_Encoder(self.args, self.aligned_dim, self.aligned_dim, self.cross_transformer_levels, self.atten_dropout)

    def unimodal_encoding(self, text, audio, vision):
        """
        :param text: (batch_size, 3, 39),需经过bert模型后得到实际的feature
        :param audio: (batch_audio, batch_lengths)，其中batch_audio的数据表示为(batch_size, audio_seq_len, audio_dim), batch_lengths的尺寸是(batch_size, ),是每条audio数据的长度，如果不对齐就是原始的变长，如果对齐就是统一的固定长度
        :param vision: 同理于audio数据的信息
        :return:
        """
        audio_input_embed, audio_lengths = audio
        vision_input_embed, vision_lengths = vision

        mask_len = torch.sum(text[:, 1, :], dim=1, keepdim=True) # text[:, 1, :]把所有数据的第一维度获取到，得到大小为(batch_size, seq_len)的mask，经过sum(dim=1)可以得到序列的真实长度,keepdim＝True可以保持原始尺度，即(batch_size,1)
        text_lengths = mask_len.squeeze().int().detach().cpu() # 大小为batch的列表，每个元素代表了序列的真实长度

        text_embed = self.text_bert_encoder(text)
        text_cls_embed = text_embed[:, 0, :] # 所有句子的[CLS]的向量表示，shape=[batch_size, 768]
        text_tokens_embed = text_embed[:, 1: -1, :] # 由bert编码的所有token的表示语义, shape=(batch, seq_len, 768)
        text_tokens_embed = self.align_t_transformer_dim(text_tokens_embed)

        if self.aligned: # 获取相同长度的三种模态特征,默认aligned为False
            if self.lstm_encoder:
                audio_cls_embed, audio_tokens_embed = self.audio_lstm_encoder(audio_input_embed, text_lengths) # (batch_size, args.audio_out)
                vision_cls_embed, vision_tokens_embed = self.vision_lstm_encoder(vision_input_embed, text_lengths) # (batch_size, args.video_out)
                audio_tokens_embed = self.align_a_transformer_dim(audio_tokens_embed)
                vision_tokens_embed = self.align_v_transformer_dim(vision_tokens_embed)
            else:
                conved_audio = self.conv_a(audio_input_embed.transpose(1,2)).transpose(1,2) # conv的输入输出形式是(batch_size, dim, seq_len), 输出后转换为正常的(batch_size, seq_len, dim)
                audio_cls_embed, audio_tokens_embed = self.audio_trans_encoder.unimodal_feature(conved_audio, text_lengths) # (batch_size, args.audio_out)

                conved_vision = self.conv_v(vision_input_embed.transpose(1,2)).transpose(1,2)
                vision_cls_embed, vision_tokens_embed  = self.vision_trans_encoder.unimodal_feature(conved_vision, text_lengths) # (batch_size, args.video_out)
        else: # 获取不同长度的三种模态特征，各模态有自己的长度
            if self.lstm_encoder:
                audio_cls_embed, audio_tokens_embed = self.audio_lstm_encoder(audio_input_embed, audio_lengths)
                vision_cls_embed, vision_tokens_embed  = self.vision_lstm_encoder(vision_input_embed, vision_lengths) # 保持原始长度的情况下进行lstm编码
                audio_tokens_embed = self.align_a_transformer_dim(audio_tokens_embed)
                vision_tokens_embed = self.align_v_transformer_dim(vision_tokens_embed) # (batch, seq_len, train_dim)
            else:
                conved_audio = self.conv_a(audio_input_embed.transpose(1,2)).transpose(1,2) # conv的输入输出形式是(batch_size, dim, seq_len), 输出后转换为正常的(batch_size, seq_len, dim)
                audio_cls_embed, audio_tokens_embed = self.audio_trans_encoder.unimodal_feature(conved_audio) # (batch_size, args.audio_out)

                conved_vision = self.conv_v(vision_input_embed.transpose(1,2)).transpose(1,2)
                vision_cls_embed, vision_tokens_embed  = self.vision_trans_encoder.unimodal_feature(conved_vision) # (batch_size, args.video_out)

        #*******************************************************************************************************************#
        # 以上部分完成了模型的初始编码
        return (text_cls_embed, audio_cls_embed, vision_cls_embed), (text_tokens_embed, audio_tokens_embed, vision_tokens_embed)
        # 前一个的输出尺度为(batch_size, out_dim); 后一个的输出维度为(batch_size, seq_len ,trans_dim)

    def unimodal_CL(self, unimodal_embed, text_labels, audio_labels, vision_labels):
        """
        # *******************************************************************************************************************#
        # 接下来实现子任务的对比学习
        :param text: 来自原始数据的text_bert的数据, (batch, 3, 39)
        :param audio: 来自原始数据的信息 (batch, seq_len, dim)
        :param vision: 来自原始数据的信息 (batch, seq_len, dim)
        :param text_labels: 原始数据的单模态标签
        :param audio_labels: 原始数据的单模态标签
        :param vision_labels: 原始数据的单模态标签
        :return:　对比学习的损失和预测分类的损失，分别计算
        """
        # unimodal_embed, _ = self.unimodal_encoding(text, audio, vision)
        text, audio, vision = unimodal_embed # the output from encoder

        # 面向单模态的文本信息
        # 执行文本信息的单模态分类,获取分类损失
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)

        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t) # 执行分类或者拟合任务,the classifier, and the same as below
        # 执行文本信息的监督对比，获取同类/不同类的聚类损失
        text_p = self.mlp_t(text_h)
        # text_p = self.mlp_t(text)
        scl_loss_t = self.scl_t(text_p.unsqueeze(1), text_labels) # the projection and contrastive learning


        # 面向单模态的音频信息
        # 执行音频信息的单模态分类
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)

        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)
        # 执行音频信息的监督对比，获取同类/不同类的聚类损失
        audio_p = self.mlp_a(audio_h)
        # audio_p = self.mlp_a(audio)
        scl_loss_a = self.scl_a(audio_p.unsqueeze(1), audio_labels)


        # 面向单模态的视觉信息
        # 执行视觉信息的单模态分类
        vision_h = self.post_vision_dropout(vision)
        vision_h = F.relu(self.post_vision_layer_1(vision_h), inplace=False) # 中间层特征

        x_v = F.relu(self.post_vision_layer_2(vision_h), inplace=False)
        output_vision = self.post_vision_layer_3(x_v) # 预测结果
        # 执行视觉信息的监督对比，获取同类/不同类的聚类损失
        vision_p = self.mlp_v(vision_h)
        # vision_p = self.mlp_v(vision)
        scl_loss_v = self.scl_v(vision_p.unsqueeze(1), vision_labels)

        return (output_text, output_audio, output_vision), (text_h, audio_h, vision_h), scl_loss_t + scl_loss_a + scl_loss_v
        # 前三个是单模态的预测输出，后三个是单模态的监督对比损失

    def multimodal_CL_first(self, unimodal_embed):
        # _, unimodal_embed = self.unimodal_encoding(text, audio, vision)
        # 定义i, j, k = text, audio, vision
        text, audio, vision = unimodal_embed # 此处的unimodal_embed的shape是(batch_size, seq_len, transformer_dim)
        # ijk <——> ikj
        # multi_ijk = self.multimodal_encoding(text, audio, vision)
        jk = self.transformer_jk(audio, vision, vision)
        ijk = self.transformer_ijk.multimodal_feature(text, jk, jk) # the pooling output from encoder
        ijk_p = self.ijk_mlp(ijk) # the output from projection
        # multi_ikj = self.multimodal_encoding(text, vision, audio)
        kj = self.transformer_kj(vision, audio, audio)
        ikj = self.transformer_ikj.multimodal_feature(text, kj, kj) # the pooling output from encoder
        ikj_p = self.ikj_mlp(ikj) # the output from projection
        sscl_i = self.self_cl_i(ijk_p, ikj_p)

        # jik <——> jki
        # multi_jik = self.multimodal_encoding(audio, text, vision)
        ik = self.transformer_ik(text, vision, vision)
        jik = self.transformer_jik.multimodal_feature(audio, ik, ik) # the pooling output from encoder
        jik_p = self.jik_mlp(jik) # the output from projection
        # multi_jki = self.multimodal_encoding(audio, vision, text)
        ki = self.transformer_ki(vision, text, text)
        jki = self.transformer_jki.multimodal_feature(audio, ki, ki) # the pooling output from encoder
        jki_p = self.jki_mlp(jki) # the output from projection
        sscl_j = self.self_cl_j(jik_p, jki_p)

        # kij <——> kji
        # multi_kij = self.multimodal_encoding(vision, text, audio)
        ij = self.transformer_ij(text, audio, audio)
        kij = self.transformer_kij.multimodal_feature(vision, ij, ij) # the pooling output from encoder
        kij_p = self.kij_mlp(kij) # the output from projection
        # multi_kji = self.multimodal_encoding(vision, audio, text)
        ji = self.transformer_ji(audio, text, text)
        kji = self.transformer_kji.multimodal_feature(vision, ji, ji) # the pooling output from encoder
        kji_p = self.kji_mlp(kji) # the output from projection
        sscl_k = self.self_cl_k(kij_p, kji_p)

        return torch.cat([ijk, ikj, jik, jki, kij, kji], dim=1), sscl_i + sscl_j + sscl_k
        # 可能此处仍然获取ijk的(batch, seq_len, dim)形式，然后进行self-attention (transformer)


    def multimodal_CL_last(self, unimodal_embed):
        # _, unimodal_embed = self.unimodal_encoding(text, audio, vision)
        # 定义i, j, k = text, audio, vision
        text, audio, vision = unimodal_embed  # 此处的unimodal_embed的shape是(batch_size, seq_len, transformer_dim)
        # ijk <——> jik
        # multi_ijk = self.multimodal_encoding(text, audio, vision)
        jk = self.transformer_jk(audio, vision, vision)
        ijk = self.transformer_ijk.multimodal_feature(text, jk, jk) # the pooling output from encoder
        ijk_p = self.ijk_mlp(ijk)  # the output from projection
        # multi_jik = self.multimodal_encoding(audio, text, vision)
        ik = self.transformer_ik(text, vision, vision)
        jik = self.transformer_jik.multimodal_feature(audio, ik, ik) # the pooling output from encoder
        jik_p = self.jik_mlp(jik)  # the output from projection
        sscl_k = self.self_cl_k(ijk_p, jik_p)

        # kji <——> jki
        # multi_kji = self.multimodal_encoding(vision, audio, text)
        ji = self.transformer_ji(audio, text, text)
        kji = self.transformer_kji.multimodal_feature(vision, ji, ji) # the pooling output from encoder
        kji_p = self.kji_mlp(kji)  # the output from projection
        # multi_jki = self.multimodal_encoding(audio, vision, text)
        ki = self.transformer_ki(vision, text, text)
        jki = self.transformer_jki.multimodal_feature(audio, ki, ki) # the pooling output from encoder
        jki_p = self.jki_mlp(jki)  # the output from projection
        sscl_i = self.self_cl_i(kji_p, jki_p)

        # kij <——> ikj
        # multi_kij = self.multimodal_encoding(vision, text, audio)
        ij = self.transformer_ij(text, audio, audio)
        kij = self.transformer_kij.multimodal_feature(vision, ij, ij) # the pooling output from encoder
        kij_p = self.kij_mlp(kij)  # the output from projection
        # multi_ikj = self.multimodal_encoding(text, vision, audio)
        kj = self.transformer_kj(vision, audio, audio)
        ikj = self.transformer_ikj.multimodal_feature(text, kj, kj) # the pooling output from encoder
        ikj_p = self.ikj_mlp(ikj)  # the output from projection
        sscl_j = self.self_cl_j(kij_p, ikj_p)

        return torch.cat([ijk, ikj, jik, jki, kij, kji], dim=-1), sscl_i + sscl_j + sscl_k

    def multimodal_CL(self, tokens_embed, multi_labels):
        if self.modal_based == 0:
            multi_features, sscl_loss = self.multimodal_CL_first(tokens_embed)
        else:
            multi_features, sscl_loss = self.multimodal_CL_last(tokens_embed)

        # 在此处添加对multi_features的self-attention，然后执行scl

        if len(multi_features.shape) == 2:
            fusion_features, _ = self.fusion_encoder.unimodal_feature(multi_features.unsqueeze(1)) # 编码器输出的特征,(batch_size, self.aligned_dim)
        else:
            fusion_features, _ = self.fusion_encoder.unimodal_feature(multi_features)
        fusion_h = self.post_fusion_dropout(fusion_features)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)

        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f) # the predicted result

        fusion_p = self.fusion_mlp(fusion_h)
        scl_loss_f = self.scl_fusion(fusion_p.unsqueeze(1), multi_labels)

        return output_fusion, fusion_h, sscl_loss+scl_loss_f

    def forward(self, text, audio, vision, multi_labels, text_labels=None, audio_labels=None, vision_labels=None):
        cls_embed, tokens_embed = self.unimodal_encoding(text, audio, vision)
        # 至于下面的损失函数计算，可以设计不同的方法和融合方式
        if text_labels: # 面向SIMS数据集
            uni_tav_preds, uni_tav_features, uni_scl_loss = self.unimodal_CL(cls_embed, text_labels, audio_labels, vision_labels)
            multi_f_pred, multi_f_feature, multi_scl_loss = self.multimodal_CL(tokens_embed, multi_labels)
            features = {
                'Feature_T': uni_tav_features[0],
                'Feature_A': uni_tav_features[1],
                'Feature_V': uni_tav_features[2],
                'Feature_M': multi_f_feature
            }
            return uni_tav_preds, uni_scl_loss, (multi_scl_loss, multi_f_pred), features
        else: # # 面向其他两个数据集
            multi_f_pred, multi_f_feature, multi_scl_loss = self.multimodal_CL(tokens_embed, multi_labels)
            features = {
                'Feature_M': multi_f_feature
            }
            return multi_scl_loss, multi_f_pred, features

        # 在执行特征可视化的过程中，我们一样要返回特征向量的表示，具体的字典信息为(除了当前的输出，还应包括的特征信息):
        # features = {
        #     'Feature_t': [],
        #     'Feature_a': [],
        #     'Feature_v': [],
        #     'Feature_f': []
        # }

