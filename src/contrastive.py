#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/22 上午8:18
# @Author  : PeiP Liu
# @FileName: contrastive.py
# @Software: PyCharm

import torch
import torch.nn as nn
import numpy as np

"""
The codes are from the paper "Supporting Clustering with Contrastive Learning" for self-supervised contrastive learning,
and "Supervised Contrastive Learning" for supervised contrastive learning.
We add the notes for easy understanding.
"""

# for supervised contrastive learning
class SCL_SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SCL_SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """
        :param features: the shape of features is (batch, 1, dim)
        :param labels: ground truth with shape [batch_size]
        :param mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        :return: a loss scalar
        """
        device = features.device
        batch_size, contrast_count, _ = features.shape
        features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device) # the shape is (batch_size, batch_size), while mask[i][j]=1 while they are from the same class including i=j
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (batch_size*contrast_count, dim)

        anchor_feature = contrast_feature # any sample can be the anchor
        anchor_count = contrast_count # facing the anchor_count to compute similarity

        all_pairs = torch.div(torch.matmul(anchor_feature, anchor_feature.T), self.temperature) # (batch_size, batch_size), torch.div(a,b) = a/b
        logits_max, _ = torch.max(all_pairs, dim=1, keepdim=True) # return the max value and its position in the corresponding dim
        logits = all_pairs -logits_max # we can regard it as standardization

        # tile mask, i.e., an anchor with contrast_count pos/neg samples
        mask = mask.repeat(anchor_count, contrast_count) # (anchor_count, contrast_count)*(batch_size, batch_size), each (batch_size, batch_size) is the former mask
        logits_mask = torch.scatter(torch.ones_like(mask), dim=1,
                                    index=torch.arange(batch_size*anchor_count).view(-1, 1).to(device),
                                    value=0) # the diagonal is False, others are 1

        # the above logits_mask is the same with the following two rows
        # logits_mask = ~torch.eye(batch_size*anchor_count, dtype=torch.bool).to(device)
        # logits_mask = torch.gt(logits_mask, 0).int()

        mask = mask * logits_mask # setting the mask[i][i]=0, mask[i][j]=1 while i and j are from the same class and i≠j
        exp_logits = torch.exp(logits) * logits_mask # based on the row (anchor), the similarity between anchor and all other samples

        log_prob4each_pos_pair = torch.log(torch.exp(logits) / (exp_logits.sum(1, keepdim=True))) * mask # each similarity in the row divided by the sum of row, but we only choose the same class by mask
        mean_log_prob4each_pos_pair = log_prob4each_pos_pair.sum(1) / mask.sum(1) # mean log_prob for each pos pair. .sum(1) is the sum for same class , 1/mask.sum(1) is for 1/P(i)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob4each_pos_pair
        loss = loss.view(anchor_count, batch_size).mean() # average the loss of all samples

        return loss

# for self-supervised contrastive learning
class SSCL_PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(SSCL_PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-8
        print('We are initializing the self-supervised contrastive learning\n')

    def forward(self, feature1, feature2):
        """
        :param feature1: feature1 has the same shape with the feature2, and the element with the same index is the optimal object. (batch_size, dim)
        :param feature2:
        :return: the mean loss in this batch
        """
        device = feature1.device
        batch_size = feature1.shape[0]
        assert feature1.shape == feature2.shape, "There is the wrong shape between the two input tensors in SSCL!"
        features = torch.cat([feature1, feature2], dim=0) # shape = (2*batch_size, dim)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device) # a matrix with the diagonal being True, others being False
        mask = mask.repeat(2, 2) # mask.shape = (2*batch_size, 2*batch_size)
        mask = ~mask # diagonal of each matrix in the (2,2) is False, others is True

        pos = torch.exp(torch.sum(feature1*feature2, dim=-1) / self.temperature) # torch.sum(X) is used for the pos-pair similarity computation
        pos = torch.cat([pos, pos], dim=0) #shape=(2*batch_size,), the first pos is used for anchor, and the second is used for its pos

        all_pairs = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature) # torch.mm(X) is for the all element-pair similarity, no matter the pos or neg
        neg = all_pairs.masked_select(mask).view(2*batch_size, -1) # the return from masked_select is a vector with one dim, i.e., (X, )
        neg = neg.sum(-1) # the sum includes all the anchor-neg sims

        loss_pos = (-torch.log(pos / (pos + neg))).mean()

        return loss_pos # 是一个张量，可以用于反传

