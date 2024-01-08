#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/31 上午9:58
# @Author  : PeiP Liu
# @FileName: train1.py
# @Software: PyCharm
import  os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict2str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')
class ModelTrain():
    def __init__(self, args):
        # assert args.train_mode == 'regression'
        self.train_mode = args.train_mode
        self.criterion = nn.MSELoss() if self.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.args = args
        # self.args.tasks = 'MTAV'
        self.loss_path = self.args.datapath.replace('.pkl', '.loss')
        self.metrics = MetricsTop(args.train_mode).getMetrics(args.datasetName) # 返回的是一个函数名

        # self.feature_map = {
        #     "fusion": torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
        #     'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
        #     'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
        #     'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device)
        # }

        # self.dim_map = {
        #     'fusion': torch.tensor(args.post_fusion_dim).float(),
        #     'text': torch.tensor(args.post_text_dim).float(),
        #     'audio': torch.tensor(args.post_audio_dim).float(),
        #     'vision': torch.tensor(args.post_video_dim).float(),
        # }

        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision',
        }

    def do_train(self, model, dataloader, return_results=False): # 此处的model是AMIO
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.model.text_bert_encoder.named_parameters()) + list(model.model.align_t_transformer_dim.named_parameters())
        if self.args.lstm_encoder:
            audio_params = list(model.model.audio_lstm_encoder.named_parameters()) + list(model.model.align_a_transformer_dim.named_parameters())
            video_params = list(model.model.vision_lstm_encoder.named_parameters()) + list(model.model.align_v_transformer_dim.named_parameters())
            model_params_other = [p for n, p in list(model.model.named_parameters()) if
                                      'text_bert_encoder' not in n and 'align_t_transformer_dim' not in n and
                                      'audio_lstm_encoder' not in n and 'align_a_transformer_dim' not in n and
                                      'vision_lstm_encoder' not in n and 'align_v_transformer_dim' not in n]
        else:
            audio_params = list(
                model.model.conv_a.named_parameters()) + list(model.model.audio_trans_encoder.named_parameters())
            video_params = list(
                model.model.conv_v.named_parameters()) + list(model.model.vision_trans_encoder.named_parameters())
            model_params_other = [p for n, p in list(model.model.named_parameters()) if
                                  'text_bert_encoder' not in n and 'align_t_transformer_dim' not in n and
                                  'conv_a' not in n and 'audio_trans_encoder' not in n and
                                  'conv_v' not in n and 'vision_trans_encoder' not in n]


        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0,
             'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio,
             'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video,
             'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
             'lr': self.args.learning_rate_other},
        ]

        optimizer = optim.Adam(optimizer_grouped_parameters)
        saved_labels = {}

        logger.info("Init labels ...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                indexes = batch_data['index'].view(-1)
                self.label_map['fusion'][indexes] = batch_data['labels']['M'].view(-1).to(self.args.device)
                if self.args.datasetName.upper() == 'SIMS':
                    self.label_map['text'][indexes] = batch_data['labels']['T'].view(-1).to(self.args.device)
                    self.label_map['audio'][indexes] = batch_data['labels']['A'].view(-1).to(self.args.device)
                    self.label_map['vision'][indexes] = batch_data['labels']['V'].view(-1).to(self.args.device)

        # initialize results
        logger.info('Starting training ...')
        epochs, best_epoch = 0, 0

        if return_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }

        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0 # 选择以更大值还是更小值进行选择更优

        losses = []
        # loop until early stop
        while True:
            epochs +=1
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}

            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td: # batch_data来自于MSADataset.__getitem__
                    if left_epochs == self.args.update_epochs: # 首次执行初始化
                        optimizer.zero_grad()
                    left_epochs -= 1 # 每次执行减少epoch数值

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)

                    batch_y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
                    batch_y_true = {'M': [], 'T': [], 'A': [], 'V': []}

                    if not self.args.need_data_aligned: # 不需要数据对齐
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    loss = 0.0
                    # forward，对于sims和其他两个数据集进行不同的loss计算
                    if self.args.datasetName.upper() == 'SIMS':
                        # 在此处需要获取y_true标签
                        for m in self.args.tasks:
                            batch_y_true[m] = self.label_map[self.name_map[m]][indexes]
                        single_pred, single_closs, fusion_result, _ = model(text, (audio, audio_lengths),
                                                                         (vision, vision_lengths), batch_y_true['M'],
                                                                         batch_y_true['T'], batch_y_true['A'],
                                                                         batch_y_true['V'])
                        batch_y_pred['M'], batch_y_pred['T'], batch_y_pred['A'], batch_y_pred['V'] = fusion_result[1], \
                                                                                                     single_pred[0], \
                                                                                                     single_pred[1], \
                                                                                                     single_pred[2]
                        for m in self.args.tasks:
                            y_pred[m].append(batch_y_pred[m].cpu())
                            y_true[m].append(batch_y_true[m].cpu())

                        loss = loss + single_closs + fusion_result[0]

                        for m in self.args.tasks:
                            loss += self.weighted_loss(batch_y_pred[m], batch_y_true[m], indexes=indexes, mode=self.name_map[m])

                    else:
                        batch_y_true['M'] = self.label_map[self.name_map['M']][indexes]
                        sscl_loss, output_fusion, _ = model(text, (audio, audio_lengths),
                                                                     (vision, vision_lengths), batch_y_true['M'])
                        batch_y_pred['M'] = output_fusion

                        y_pred['M'].append(batch_y_pred['M'].cpu())
                        y_true['M'].append(batch_y_true['M'].cpu())

                        loss = loss + sscl_loss
                        loss += self.weighted_loss(batch_y_pred['M'], batch_y_true['M'], indexes=indexes)
                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()
            train_loss = train_loss/len(dataloader['train'])
            losses.append(train_loss)
            with open(self.loss_path, 'w+') as loss_rwt:
                loss_rwt.write(losses)
            loss_rwt.close()

            # 在每个epoch之中进行validation和test
            # 此处的logger输出忽略
            train_results = {}
            for m in self.args.tasks: # MTAV or M
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results['Metrics_'+m] = self.metrics(pred, true)
                logger.info("%s: >> "%(m) + dict2str(train_results[m]))
                # if self.args.datasetName.upper() != 'SIMS':
                #     break # 不是sims的数据，仅经历一轮游，实现M的测试
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode='VAL')
            cur_valid = val_results[self.args.KeyEval] # 以pred和true之间的loss作为判定标准，其他loss信息不算做选择范围

            # save the best model，实际上我们选择的是使用更小的loss，即cur_valid越小越好
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # save the labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k,v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epochs] = tmp_save
            # epoch results
            if return_results:
                train_results['Loss'] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode='TEST')
                epoch_results['test'].append(test_results)
            if epochs - best_epoch >= self.args.early_stop:
                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir, '{}-{}-labels.pkl'.format(self.args.modelName, self.args.datasetName)), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)

                return epoch_results if return_results else None

    def do_test(self, model, dataloader, mode='VAL', return_sample_results=False): # 验证集选择模型时，只能选择一个指标作为最优模型的选择评判。因此，这里只选择M作为评价标准。
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        if return_sample_results:
            ids, features, all_labels, sample_results = [], {},{},{}
            for task_sig in self.args.tasks:
                features['Feature_'+task_sig] = []
                all_labels[task_sig] = []
                sample_results[task_sig] = []

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:

                    batch_y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
                    batch_y_true = {'M': [], 'T': [], 'A': [], 'V': []}

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)

                    if not self.args.need_data_aligned:  # 不需要数据对齐
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # 获取ground_truth标签信息
                    batch_y_true['M'] = batch_data['labels']['M'].view(-1).to(self.args.device)
                    if self.args.datasetName.upper() == 'SIMS':
                        batch_y_true['T'] = batch_data['labels']['T'].view(-1).to(self.args.device)
                        batch_y_true['A'] = batch_data['labels']['A'].view(-1).to(self.args.device)
                        batch_y_true['V'] = batch_data['labels']['V'].view(-1).to(self.args.device)

                    loss = 0.0
                    # forward，对于sims和其他两个数据集进行不同的loss计算
                    if self.args.datasetName.upper() == 'SIMS':
                        # 在此处需要获取y_true标签
                        single_pred, single_closs, fusion_result, learned_features = model(text, (audio, audio_lengths),
                                                                         (vision, vision_lengths), batch_y_true['M'],
                                                                         batch_y_true['T'], batch_y_true['A'],
                                                                         batch_y_true['V'])
                        batch_y_pred['M'], batch_y_pred['T'], batch_y_pred['A'], batch_y_pred['V'] = fusion_result[1], \
                                                                                                     single_pred[0], \
                                                                                                     single_pred[1], \
                                                                                                     single_pred[2]
                        for m in self.args.tasks:
                            y_pred[m].append(batch_y_pred[m].cpu())
                            y_true[m].append(batch_y_true[m].cpu())

                        # loss = loss + single_closs + fusion_result[0]
                        #
                        for m in self.args.tasks:
                            loss += self.weighted_loss(batch_y_pred[m], batch_y_true[m], indexes=indexes,
                                                       mode=self.name_map[m])

                    else:
                        sscl_loss, output_fusion, learned_features = model(text, (audio, audio_lengths),
                                                         (vision, vision_lengths), batch_y_true['M'])
                        batch_y_pred['M'] = output_fusion

                        y_pred['M'].append(batch_y_pred['M'].cpu())
                        y_true['M'].append(batch_y_true['M'].cpu())

                        # loss = loss + sscl_loss
                        loss += self.weighted_loss(batch_y_pred['M'], batch_y_true['M'], indexes=indexes)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in self.args.tasks:
                            features['Feature_'+item].append(learned_features['Feature_'+item].cpu().detach().numpy())
                            all_labels[item].extend(batch_y_true[item].cpu().detach().tolist())
                            sample_results[item].extend(batch_y_pred[item].cpu().detach().numpy().squeeze())

                    # loss = self.weighted_loss(batch_y_pred['M'], batch_y_true['M'])
                    eval_loss += loss.item()
        eval_loss = eval_loss / len(dataloader)
        logger.info(mode + '-(%s)' % self.args.modelName + " >> loss: %.4f " % eval_loss)
        # pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        # eval_results = self.metrics(pred, true)
        # 在此处添加，如果是sims数据集，那么需要对TAV进行性能测试
        # if self.args.datasetName.upper() == 'SIMS': # 执行单模态的性能评估，评测了TAV形式下的单模态的指标
        #     for uni_modal in 'TAV':
        #         uni_pred, uni_true = torch.cat(y_pred[uni_modal]), torch.cat(y_true[uni_modal])
        #         uni_eval_results = self.metrics(uni_pred, uni_true)
        #         for metric_key in uni_eval_results.keys():
        #             eval_results[uni_modal+"_"+metric_key] = uni_eval_results[metric_key]
        eval_results = {}
        for uni_modal in self.args.tasks:
            uni_pred, uni_true = torch.cat(y_pred[uni_modal]), torch.cat(y_true[uni_modal])
            uni_eval_results = self.metrics(uni_pred, uni_true)
            eval_results['Metrics_'+uni_modal] = uni_eval_results
            logger.info("%s: >> "%(uni_modal) + dict2str(eval_results))
        eval_results['Loss'] = eval_loss
        if return_sample_results:
            eval_results['Ids'] = ids
            eval_results['PredResults'] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0) # 包含了单模态的特征向量
            eval_results['Features'] = features
            eval_results['TrueLabels'] = all_labels
        return eval_results


    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        # 考虑训练模式，然后选择不同的损失函数，包括classification和regression
        if self.train_mode == 'regression':
            y_pred = y_pred.view(-1)
            y_true = y_true.view(-1)
            if mode == 'fusion':
                weighted = torch.ones_like(y_pred)
            else:
                weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
            loss = torch.mean(weighted * torch.abs(y_pred-y_true))
        # elif self.train_mode == 'regression': # MSE
        #     y_true = y_true.view(-1, 1) # y_pred的形式是(batch, 1)
        #     loss = self.criterion(y_pred, y_true)
        else: # classification
            y_true = y_true.view(-1).long() # y_pred的形式是(batch, num_classes)
            loss = self.criterion(y_pred, y_true)
        return loss
