# -*- coding: UTF-8 -*-
import numpy as np
import math


def cal_hr(predict, ground_truth):
    """
    calculate hit rate
    :param predict: predict, (batch_size, k)
    :param ground_truth: ground truth, (batch_size, )
    :return: hr
    """
    hr = 0
    for i, gt in enumerate(ground_truth):
        if gt in predict[i]:
            hr += 1
    return hr


def cal_mrr(predict, ground_truth):
    """
    calculate mrr
    :param predict: predict, (batch_size, k)
    :param ground_truth: ground truth, (batch_size, )
    :return: mrr score
    """
    mrr = 0
    for i, gt in enumerate(ground_truth):
        if gt in predict[i]:
            rank = np.where(predict[i] == gt)[0][0]
            mrr += 1 / (rank + 1)
    return mrr


def cal_ndcg(predict, ground_truth):
    """
    calculate ndcg
    :param predict: predict, (batch_size, k)
    :param ground_truth: ground truth, (batch_size, )
    :return: ndcg score
    """
    ndcg = 0
    for i, gt in enumerate(ground_truth):
        if gt in predict[i]:
            rank = np.where(predict[i] == gt)[0][0]
            ndcg += 1 / math.log(rank + 2, 2)
    return ndcg