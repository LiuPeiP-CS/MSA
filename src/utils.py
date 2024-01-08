#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 下午10:27
# @Author  : PeiP Liu
# @FileName: utils.py
# @Software: PyCharm

import torch
import os
from src.dataset import Multimodal_Dataset

def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + "_{}_{}.dt".format(split, alignment)
    if not os.path.exists(data_path)
        print('Creating new {} data'.format(split))
        data = Multimodal_Dataset(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)

    else:
        print('Found cached {} data'.format(split))
        data = torch.load(data_path)
    return data

def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else "aligned_model"
    elif not args.aligned:
        name = name if len(name) > 0 else 'noaligned_model'

    return name + '_' + args.model

def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, 'pre_trained_models/{}.pt'.format(name))

def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load('pre_trained_models/{}.pt'.format(name))