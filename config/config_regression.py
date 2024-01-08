#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 上午11:12
# @Author  : PeiP Liu
# @FileName: config_regression.py
# @Software: PyCharm

# 本文件可以认为是该项目及模型中所有信息的参数集合

import os
import argparse
from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # setting the hyper-parameters for the model
        model_hyper_parameters = self.__init_modelParams() # 面向模型运行过程中的参数设置
        # load the common params for all the datasets
        model_commonArgs = model_hyper_parameters['commonParams'] # 无论使用什么数据集，在模型运行时的通用参数

        dataset_hyper_parameters = self.__init_datasetParams() # 面向数据集本身的参数设置
        datasetName = str.lower(args.datasetName)
        datasetArgs = dataset_hyper_parameters[datasetName] # 针对某个数据集的参数设置

        # 保证模型运行时需要aligned，并且数据集有aligned的条件
        if model_commonArgs['need_data_aligned'] and 'aligned' in datasetArgs:
            datasetArgs = datasetArgs['aligned']
        else:
            datasetArgs = datasetArgs['unaligned']

        # 以字典的形式将数据集的所有参数配置进行集成
        self.args = Storage(
            dict(
                vars(args), # 返回原始字典的类对象。参数只能是字典类，不能是字典类对象。
                # 使用 ** 可以解包字典，解包完后再使用dict或者{}就可以合并,　详情可见https://www.zhihu.com/tardis/zm/art/130204496?source_id=1005
                **model_commonArgs,
                **model_hyper_parameters['datasetParams'][datasetName],
                **datasetArgs
            )
        )


    def __init_modelParams(self):
        dict4model = {
            'commonParams': {
                'need_data_aligned': False,
                'need_model_aligned': False, # 模型对齐是指针对T/A/V的不同序列长度，进行对齐处理。对齐的主要操作，在alignNet
                'model_aligned_mode': 'avg_pool', # 模型对齐的方式，可选avg_pool/ctc/convnet
                'need_normalized': False, # 正常情况下即False
                'use_bert': True,
                'bert_root':'/data/liupeipei/paper/ICASSP/pretrained_bert',
                'lstm_encoder':True, # 是否选择lstm作为编码器
                'save_labels': False,
                'modal_based': 0, # 0表示第一个模态为基准，1表示最后一个模态为基准进行
                'early_stop': 8,
                'update_epochs': 4 # 优化器的更新周期，最好设置为1
            },
            # for dataset setting
            'datasetParams': {
                'mosi':{
                    'batch_size': 32,

                    'atten_dropout_a': 0.2,
                    'atten_dropout_v': 0.0,
                    'relu_dropout': 0.0,
                    'embed_dropout': 0.2,
                    'res_dropout': 0.0,
                    'transformer_head': 4,
                    'transformer_dim': 30,
                    'learning_rate': 0.002,
                    'conv1d_kernel_size_t': 5,
                    'conv1d_kernel_size_a': 5,
                    'conv1d_kernel_size_v': 5,
                    'unimodal_trans_levels': 4,
                    'multimodal_trans_levels': 2,
                    'atten_dropout': 0.3,

                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,

                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.01,
                    # feature subnets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,

                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout': 0.1,

                    #post feature
                    'post_fusion_dim': 128,
                    'post_text_dim': 64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,

                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.0,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,

                    # res ???
                    'H': 3.0
                },

                'mosei': {
                    'batch_size': 32,

                    'atten_dropout_a': 0.0,
                    'atten_dropout_v': 0.0,
                    'relu_dropout': 0.0,
                    'embed_dropout': 0.0,
                    'res_dropout': 0.0,
                    'transformer_head': 4,
                    'transformer_dim': 30,
                    'learning_rate': 0.0005,
                    'conv1d_kernel_size_t': 5,
                    'conv1d_kernel_size_a': 1,
                    'conv1d_kernel_size_v': 3,
                    'unimodal_trans_levels': 4,
                    'multimodal_trans_levels': 2,
                    'atten_dropout': 0.4,

                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,

                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.0,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.01,
                    # feature subnets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,

                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout': 0.1,


                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim': 32,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,

                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.1,

                    # res ???
                    'H': 3.0
                },

                'sims': {
                    'batch_size': 32,

                    'atten_dropout_a': 0.1,
                    'atten_dropout_v': 0.0,
                    'relu_dropout': 0.0,
                    'embed_dropout': 0.1,
                    'res_dropout': 0.2,
                    'transformer_head': 4,
                    'transformer_dim': 30,
                    'learning_rate': 0.002,
                    'conv1d_kernel_size_t': 5,
                    'conv1d_kernel_size_a': 1,
                    'conv1d_kernel_size_v': 1,
                    'unimodal_trans_levels': 4,
                    'multimodal_trans_levels': 2,
                    'atten_dropout': 0.2,

                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 5e-3,
                    'learning_rate_other': 1e-3,

                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    # feature subnets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,

                    'text_out': 768,
                    'audio_out': 16,
                    'video_out': 32,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout': 0.1,

                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim': 64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,

                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,

                    # res ???
                    'H': 1.0
                }
            }
        }

        return dict4model


    def __init_datasetParams(self):
        root_dataset_dir = '/data/liupeipei/paper/ICASSP/SelfMM_data'

        dict4data = {
            'mosi': {
                'aligned':{
                    'datapath': os.path.join(root_dataset_dir, 'MOSI/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50), # 这是三种模态已经对齐的特征信息，长度为50
                    # for (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284, # 训练数据的样本数量
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },

                'unaligned':{
                    'datapath': os.path.join(root_dataset_dir, 'MOSI/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),  # 这是三种模态已经对齐的特征信息，长度为50
                    # for (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,  # 训练数据的样本数量
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },

            'mosei':{
                'aligned': {
                    'datapath': os.path.join(root_dataset_dir, 'MOSEI/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),  # 这是三种模态已经对齐的特征信息，长度为50
                    # for (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,  # 训练数据的样本数量
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },

                'unaligned': {
                    'datapath': os.path.join(root_dataset_dir, 'MOSEI/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),  # 这是三种模态已经对齐的特征信息，长度为50
                    # for (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,  # 训练数据的样本数量
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },

            'sims':{
                'unaligned':{
                    'datapath': os.path.join(root_dataset_dir, 'SIMS/unaligned_50.pkl'),
                    'seq_lens': (39, 400, 55),  # 这是三种模态已经对齐的特征信息，长度为50
                    # for (text, audio, video)
                    'feature_dims': (768, 33, 709),
                    'train_samples': 1368,  # 训练数据的样本数量
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss'
                }
            }
        }

        return dict4data

    def get_config(self):
        return self.args





