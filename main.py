#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 下午2:22
# @Author  : PeiP Liu
# @FileName: main.py
# @Software: PyCharm

import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train

parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# for the MSA task, single modal or multi-modal
parser.add_argument('--model', type=str, default='ContrMSA', help='name of the model to use multi-view Transformer for MSA')

# Tasks
parser.add_argument('--vonly', action='store_true', help='use of the cross-modal fusion into vision (default is False)')
parser.add_argument('--aonly', action='store_true', help='use of the cross-modal fusion into audio (default is False)')
parser.add_argument('--lonly', action='store_true', help='use of the cross-modal fusion into language (default is False)')
parser.add_argument('--aligned', action='store_true', help='consider the aligned experiments or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei', help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='~/data', help='path for storing the data features')


# Dropout
parser.add_argument('--atten_dropout', type=float, default=0.1, help='attention_dropout')
parser.add_argument('--atten_dropout4a', type=float, default=0.0, help='attention_dropout for audio')
parser.add_argument('--atten_dropout4v', type=float, default=0.0, help='attention_dropout for vision')
parser.add_argument('--relu_dropout', type=float, default=0.1, help='relu_dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25, help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1, help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0, help='output layer dropout')

# Neural Network Architecture
parser.add_argument('--nlevels', type=int, default=5, help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5, help='number of heads for the Transformer network (default : 5)')
parser.add_argument('--atten_mask', action='store_false', help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N', help='batch_size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value (default : 0.8)')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20, help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1, help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30, help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--no_cuda', action='store_true', help='do not use cuda')
parser.add_argument('--name', type=str, default='ContrMSA', help='name of the trial (default: ContrMSA)')

args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.aonly = args.vonly = True
elif valid_partial_mode != 1:
    raise ValueError('You can only choose one of l/a/v.')

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei': 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print('WARNING: You have a CUDA device, so you should probably not run with -- no_cuda')
    else:
        torch.cuda_manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True


# Load the dataset (aligned or non-aligned)
print('Start loading the data ...')
train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')


train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

print('Finish loading the data ...')
if not args.aligned:
    print('Note: You are running in unaligned mode !')

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1loss')

if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
