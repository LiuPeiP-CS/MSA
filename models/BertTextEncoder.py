#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/29 下午1:41
# @Author  : PeiP Liu
# @FileName: BertTextEncoder.py
# @Software: PyCharm
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, BertConfig

class BertTextEncoder(nn.Module):
    def __init__(self, bert_root, language='en', use_finetune=False):
        super(BertTextEncoder, self).__init__()
        assert language in ['en', 'cn']

        if language == 'en':
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_root, 'bert_en'), do_lower_case=True)
            self.model = BertModel.from_pretrained(os.path.join(bert_root, 'bert_en'))
        elif language == 'cn':
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_root, 'bert_cn'))
            self.model = BertModel.from_pretrained(os.path.join(bert_root, 'bert_cn'))

    def forward(self, text):
        """
        :param text: shape=(num_samples, 3, seq_len), 3-0: word-id in bert, 3-1: attention mask, 3-2: token_seq_type(0 or 1)
        :return: the hidden representation from bert encoder
        """
        input_ids, attention_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()

        last_hidden_states = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids = segment_ids
                                        )[0] # bert模型的输出有四种信息，分别是last_hidden_states, pooler_output, hidden_states, attentions
        return last_hidden_states # 表示bert模型获取的文本表示向量，其中向量的首尾已经添加了[cls]和[sep]

