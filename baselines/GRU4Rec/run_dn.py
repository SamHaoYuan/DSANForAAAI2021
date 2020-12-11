# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import gru4rec
import evaluation

PATH_TO_TRAIN = 'datasets/diginetica/train_session_3.csv'
PATH_TO_TEST = 'datasets/diginetica/test_session_3.csv'

if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep=',', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep=',', dtype={'ItemId':np.int64})
    
    #State-of-the-art results on RSC15 from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
    #BPR-max, no embedding (R@20 = 0.7197, M@20 = 0.3157)
    # gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad', 
    # n_epochs=5, batch_size=64, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2, momentum=0.3, n_sample=2048, 
    # sample_alpha=0, bpreg=1, constrained_embedding=False)
    # gru.fit(data)
    # res = evaluation.evaluate_gpu(gru, valid, cut_off=20)
    # print('Recall@20: {}'.format(res[0]))
    # print('MRR@20: {}'.format(res[1]))
    # print('NDCG@20: {}'.format(res[2]))

    # res = evaluation.evaluate_gpu(gru, valid, cut_off=10)
    # print('Recall@10: {}'.format(res[0]))
    # print('MRR@10: {}'.format(res[1]))
    # print('NDCG@10: {}'.format(res[2]))

    # res = evaluation.evaluate_gpu(gru, valid, cut_off=5)
    # print('Recall@5: {}'.format(res[0]))
    # print('MRR@5: {}'.format(res[1]))
    # print('NDCG@5: {}'.format(res[2]))

    #BPR-max, constrained embedding (R@20 = 0.7261, M@20 = 0.3124)
    # gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad', n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2, momentum=0.1, n_sample=2048, sample_alpha=0, bpreg=0.5, constrained_embedding=True)
    # gru.fit(data)
    # res = evaluation.evaluate_gpu(gru, valid)
    # print('Recall@20: {}'.format(res[0]))
    # print('MRR@20: {}'.format(res[1]))

    #Cross-entropy (R@20 = 0.7180, M@20 = 0.3087)
    # gru = gru4rec.GRU4Rec(loss='cross-entropy', final_act='softmax', hidden_act='tanh', layers=[100], adapt='adagrad', n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0.3, learning_rate=0.1, momentum=0.7, n_sample=2048, sample_alpha=0, bpreg=0, constrained_embedding=False)
    # gru.fit(data)
    # res = evaluation.evaluate_gpu(gru, valid)
    # print('Recall@20: {}'.format(res[0]))
    # print('MRR@20: {}'.format(res[1]))
    
    #OUTDATED!!!
    #Reproducing results from the original paperr"Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    #print('Training GRU4Rec with 100 hidden units')    
    gru = gru4rec.GRU4Rec(loss='top1-max', final_act='tanh', hidden_act='tanh', n_epochs=10, layers=[100], batch_size=64, dropout_p_hidden=0.5, learning_rate=0.1, momentum=0, time_sort=False)
    gru.fit(data)

    res = evaluation.evaluate_gpu(gru, valid, cut_off=20)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print('NDCG@20: {}'.format(res[2]))

    res = evaluation.evaluate_gpu(gru, valid, cut_off=10)
    print('Recall@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))
    print('NDCG@10: {}'.format(res[2]))

    res = evaluation.evaluate_gpu(gru, valid, cut_off=5)
    print('Recall@5: {}'.format(res[0]))
    print('MRR@5: {}'.format(res[1]))
    print('NDCG@5: {}'.format(res[2]))
