#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/31 上午10:14
# @Author  : PeiP Liu
# @FileName: metricsTop.py
# @Software: PyCharm
import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == 'regression':
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
                'SIMS': self.__eval_sims_regression
            }
        else:
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_classification,
                'MOSEI': self.__eval_mosei_classification,
                'SIMS': self.__eval_sims_classification
            }

    def __eval_mosi_classification(self, y_pred, y_true):
        """
        {
            'Negative': 0,
            'Neutral': 1,
            'Positive': 2
        }
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        # three classes
        y_pred_3 = np.argmax(y_pred, axis=1)
        Multi_acc_3 = accuracy_score(y_pred_3, y_true)
        F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')

        # two classes
        y_pred = np.array([[v[0], v[2]] for v in y_pred])
        # with 0 (<=0 or >0)
        y_pred_2 = np.argmax(y_pred, axis=1)
        y_true_2 = []
        for v in y_true:
            y_true_2.append(0 if v <= 1 else 1)
        y_true_2 = np.array(y_true_2)
        Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
        # without 0 (<0 or >0)
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1]) # 记下所有非1的index
        y_pred_2 = y_pred[non_zeros]
        y_pred_2 = np.argmax(y_pred_2, axis=1)
        y_true_2 = y_true[non_zeros]
        Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Non0_F1_score = f1_score(y_true_2, y_true_2, average='weighted')

        eval_results = {
            'Has0_acc_2': round(Has0_acc_2, 4),
            'Has0_F1_score': round(Has0_F1_score, 4),
            'Non0_acc_2': round(Non0_acc_2, 4),
            'Non0_F1_score': round(Non0_F1_score, 4),
            'Acc_3': round(Multi_acc_3, 4),
            'F1_score_3': round(F1_score_3, 4)
        }

        return eval_results

    def __eval_mosei_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __eval_sims_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosi_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.) # 所有的数值中，大于3的赋值3，小于-3的赋值-3，中间数值保持不变
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(test_preds - test_truth))
        corr = np.corrcoef(test_preds, test_truth)[0][1]

        multi_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        multi_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        multi_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0) # 进行2分类的设置，即只有True和False
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_preds = (test_preds >= 0)
        binary_truth = (test_truth >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        eval_resutls = {
            'Has0_acc_2': round(acc2, 4),
            'Has0_F1_score': round(f_score, 4),
            'Non0_acc_2': round(non_zeros_acc2, 4),
            'Non0_F1_score': round(non_zeros_f1_score, 4),
            'Multi_acc_5': round(multi_a5, 4),
            'Mutti_acc_7': round(multi_a7, 4),
            'MAE': round(mae, 4),
            'Corr': round(corr, 4)
        }

        return eval_resutls

    def __eval_mosei_regression(self, y_pred, y_true):
        return self.__eval_mosi_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.) # 将数据的大小限制在(a_min, a_max)之间

        # two classes {[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            # logical_and(a,b)对a和b进行逻辑与计算，其实本条计算就是为了取值在区间(ms_2[i], ms_2[i+1]]的元素，并赋值为i(即两个类别)
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes {[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(2):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i

        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth))
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        multi_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        multi_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        multi_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            'Multi_acc_2': multi_a2,
            'Multi_acc_3': multi_a3,
            'Multi_acc_5': multi_a5,
            'F1_score': f_score,
            'MAE': mae,
            'Corr': corr, # correlation coefficient
        }

        return eval_results

    def getMetrics(self, datasetName):
        return self.metrics_dict[datasetName.upper()] # 返回的是一个函数，没有参数