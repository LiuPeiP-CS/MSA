#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 下午3:03
# @Author  : PeiP Liu
# @FileName: run.py
# @Software: PyCharm

import os
import gc
import time
import random
import torch
import pynvml
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import csv

from data.LoadData import MSADataLoader
from config.config_regression import ConfigRegression
from models.AMIO import AMIO
from src.train import ModelTrain

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    # 此处的参数args是集合了输入和config中的参数

    if not os.path.exists(args.model_save_dir): # 创建模型存储文件夹
        os.makedirs(args.model_save_dir)

    args.model_save_path = os.path.join(args.model_save_dir, '{}-{}-{}-{}.pth'.format(args.modelName, args.datasetName, args.trainMode, args.modelmode))

    # if len(args.gpu_ids) == 0 and torch.cuda.is_available():
    #     # load the free-most gpu,即已使用最小的gpu
    #     pynvml.nvmlInit() # 对gpu管理系统进行初始化
    #     dst_gpu_id, min_mem_used = 0, 1e16 # 设置允许的最小显卡内存
    #     for gpu_id in [0, 1, 2, 3]:
    #         handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id) # 获取gpu的内存信息
    #         meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #         mem_used = meminfo.used # 此块gpu的内存使用信息
    #
    #         if mem_used < min_mem_used:
    #             min_mem_used = mem_used
    #             dst_gpu_id = gpu_id
    #
    #     print('Find the gpu: {}, use memory: {}!'.format(dst_gpu_id, min_mem_used))
    #     logger.info('Find the gpu: {} with the memory: {} left !'.format(dst_gpu_id, min_mem_used))
    #
    #     args.gpu_ids.append(dst_gpu_id)

    # 在cuda可用的情况下使用gpu
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available() #　即使用gpu, 要求为true
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))

    device = torch.device('cuda: %d' % int(args.gpu_ids[0]) if using_cuda else 'cpu') # 设置gpu的使用信息
    args.device = device

    # data
    dataloader = MSADataLoader(args) # 尚未进行具体的模型化和数据集化的参数设置args
    model = AMIO(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
        return answer

    logger.info('The model has {} trainable parameters'.format(count_parameters(model)))

    run_model = ModelTrain(args)
    run_model.do_train(model, dataloader) # training over and saving the trained model

    # load the pre-trained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)

    if args.tune_mode:
        # using the valid dataset to debug hyper parameters
        results = run_model.do_test(model, dataloader['valid'], mode='VALID')
    else:
        results = run_model.do_test(model, dataloader['test'], mode='TEST')

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


"""
def run_tune(args, tune_times=50):
    args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = [] # used for saving params
    save_file_path = os.path.join(args.res_save_dir, '{}-{}-{}-{}.csv'.format(args.modelName, args.datasetName, args.trainMode, args.modelmode))
    if not os.path.exists(os.path.dirname(save_file_path)): # 如果之前没有该文件，则创建该文件
        os.makedirs(os.path.dirname(save_file_path))

    for i in range(tune_times):
        # load free-most gpu
        # pynvml.nvmlInit()
        # cancel random seed
        set_seed(int(time.time()))
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        logger.info("#"*40 + '%s-(%d/%d)'%(args.modelName, i+1, tune_times) + '#'*40)
        for k, v in args.items():
            if k in args.d_paras:
                logger.info(k+':' + str(v))
        logger.info('#'*90)
        logger.info('Start running %s ...'%(args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            logger.info('These paras have been used !')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        results = []
        for j, seed in enumerate([1111]):
            args.cur_time = j+1
            set_seed(seed)
            results.append(run(args))
        # save results to csv
        logger.info('Start saving results ...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns=[k for k in args.d_paras] + [k for k in results[0].keys()])

        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values)*100/len(values), 2))

        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        logger.info('Results are saved to %s...' % (save_file_path))
"""


def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals') # only construct a string address
    init_args = args
    model_results = []
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        if args.train_mode == 'regression':
            config = ConfigRegression(args)
        args = config.get_config()
        set_seed(seed)
        args.seed = seed
        logger.info('Start running %s ...' % (args.modelName))
        logger.info(args)
        # running
        args.cur_time = i + 1
        test_results = run(args) # ********************************* #
        # restore results
        model_results.append(test_results)
    criterions = list(model_results[0].keys()) # 利用第一个结果的所有key来说明结果的评价head name
    # load other results
    save_path = os.path.join(args.res_save_dir, '{}-{}.csv'.format(args.datasetName, args.train_mode))
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=['Model']+criterions) # 模型名称＋评价指标，作为文件的头

    # save results
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results] # 获取所有数据的c指标，并形成列表
        mean = round(np.mean(values)*100, 2) # 获取该指标下训练结果的均值以及方差
        std = round(np.std(values)*100, 2)
        res.append((mean, std)) # 每个指标下的均值和方差
    df.loc[len(df)] = res # 将res赋值到固定行上
    df.to_csv(save_path, index=None) # 将df对象存储到固定位置上
    logger.info('Results are added to %s ...'%(save_path))

def set_log(args):
    """
    # 构建日志产生器
    :param args:
    :return:
    """
    log_file_path = 'logs/{}-{}-{}-{}'.format(args.modelName, args.datasetName, args.train_mode, args.modelmode)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)

    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H: %M: %S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    # add the StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)

    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=False, help='tune parameter?')
    parser.add_argument('--train_mode', type=str, default='regression', help='regression or classification')
    parser.add_argument('--modelName', type=str, default='MSA', help='support MSA')
    parser.add_argument('--tasks', type=str, default='M', help='MTAV/M') # M for MOSEI and MOSI, MTAV for SIMS
    parser.add_argument('--datasetName', type=str, default='sims', help='mosi/mosei/sims')
    parser.add_argument('--modelmode', type=str, default='all', help='the ablation and specific modality, including A/T/V/AT/AV/TV/DSC')
    parser.add_argument('--num_workers', type=int, default=0, help='number workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models', help='path to save resulting models')
    parser.add_argument('--res_save_dir', type=str, default='results/results', help='path to save results')
    parser.add_argument('--gpu_ids', type=list, default=[0], help='indicate the gpus will be used. If none, the most-free gpu will be used')

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args() # 此处的args参数中配置了train_mode, modelName, datasetName, model_save_dir, res_save_dir, gpu_ids等
    logger = set_log(args)
    for data_name in ['sims', 'mosi', 'mosei']:
        args.datasetName = data_name
        args.seeds = [1111, 1112, 1113, 1114, 1115]
        if args.is_tune:
            run_tune(args, tune_times=50)
        else:
            run_normal(args)

