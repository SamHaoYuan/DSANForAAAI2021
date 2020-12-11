import math
import random
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy
import os
import datetime
import pickle

seed = 1
random.seed(seed)
np.random.seed(seed)


class SessionData(object):
    def __init__(self, session_index, session_id, items_indexes):
        self.session_index = session_index
        self.session_id = session_id
        self.item_list = items_indexes

    def generate_train_datas(self, session_length, padding_idx=0, predict_length=1):
        sessions = []
        if len(self.item_list) < 2:
            self.item_list.append[self.item_list[0]]
        if predict_length == 1:
            for i in range(1, len(self.item_list) - 1):
                if i < session_length:
                    train_data = [0 for _ in range(session_length - i - 1)]
                    train_data.extend(self.item_list[:i + 1])
                    train_data.append(self.item_list[i + 1])
                else:
                    train_data = self.item_list[i + 1 - session_length:i + 1]
                    train_data.append(self.item_list[i + 1])
                sessions.append(train_data)
        else:

            pass
        return self.session_index, sessions

    def __str__(self):
        info = " session index = {}\n session id = {} \n the length of item list= {} \n the fisrt item index in item list is {}".format(
            self.session_index, self.session_id, len(self.item_list), self.item_list[0])
        return info


class SessionDataSet(object):
    def __init__(self, train_file, test_file, padding_idx=0):
        super(SessionDataSet, self).__init__()
        self.index_count = 0
        self.session_count = 0
        self.train_count = 0
        self.test_count = 0
        self.max_session_length = 0
        self.padding_idx = padding_idx
        self.item2index = dict()
        self.index2item = dict()
        self.session2index = dict()
        self.index2session = dict()
        self.item_total_num = dict()
        self.train_item_total_num = dict()
        self.item2index["<pad>"] = padding_idx
        self.index2item[padding_idx] = "<pad>"
        self.train_data = self.load_data(train_file)
        print("training set is loaded, # index: ", len(self.item2index.keys()))
        self.train_count = self.session_count
        print("train_session_num", self.train_count)
        self.test_data = self.load_data(test_file)
        print("testing set is loaded, # index: ", len(self.index2item.keys()))
        print("# item", self.index_count)
        self.test_count = self.session_count - self.train_count
        print("# test session:", self.test_count)
        self.all_training_data = []
        self.all_testing_data = []
        self.all_meta_training_data = []
        self.all_meta_testing_data = []
        self.train_session_length = 0
        self.test_session_length = 0

    def load_data(self, file_path, is_train=True):
        data = pickle.load(open(file_path, 'rb'))
        print(len(data))
        session_ids = data[0]
        session_data = data[1]
        session_label = data[2]

        result_data = []
        lenth = len(session_ids)
        print("# session", lenth)

        last_session_id = session_ids[0]

        session_item_indexes = []

        for item_id in session_data[0]:
            if item_id not in self.item2index.keys():
                self.index_count += 1
                self.item2index[item_id] = self.index_count
                self.index2item[self.index_count] = item_id
                self.item_total_num[self.index_count] = 0
                if is_train:
                    self.train_item_total_num[self.index_count] = 0
            session_item_indexes.append(self.item2index[item_id])
            self.item_total_num[self.item2index[item_id]] += 1
            if is_train:
                self.train_item_total_num[self.item2index[item_id]] += 1
        target_item = session_label[0]
        if target_item not in self.item2index.keys():
            self.index_count += 1
            self.item2index[target_item] = self.index_count
            self.index2item[self.index_count] = target_item
            self.item_total_num[self.index_count] = 0
            if is_train:
                self.train_item_total_num[self.index_count] = 0
        session_item_indexes.append(self.item2index[target_item])
        self.item_total_num[self.item2index[target_item]] += 1
        if is_train:
            self.train_item_total_num[self.item2index[item_id]] += 1
        for session_id, items, target_item in zip(session_ids, session_data, session_label):
            if session_id != last_session_id:
                self.session_count += 1
                self.session2index[last_session_id] = self.session_count
                self.index2session[self.session_count] = last_session_id
                last_session_id = session_id
                if len(session_item_indexes) > self.max_session_length:
                    self.max_session_length = len(session_item_indexes)
                new_session = SessionData(self.session_count, last_session_id, session_item_indexes)
                result_data.append(new_session)
                session_item_indexes = []
                for item_id in items:
                    if item_id not in self.item2index.keys():
                        self.index_count += 1
                        self.item2index[item_id] = self.index_count
                        self.index2item[self.index_count] = item_id
                        self.item_total_num[self.index_count] = 0
                        if is_train:
                            self.train_item_total_num[self.index_count] = 0
                    session_item_indexes.append(self.item2index[item_id])
                    self.item_total_num[self.item2index[item_id]] += 1
                    if is_train:
                        self.train_item_total_num[self.item2index[item_id]] += 1
                if target_item not in self.item2index.keys():
                    self.index_count += 1
                    self.item2index[target_item] = self.index_count
                    self.index2item[self.index_count] = target_item
                    self.item_total_num[self.index_count] = 0
                    if is_train:
                        self.train_item_total_num[self.index_count] = 0
                session_item_indexes.append(self.item2index[target_item])
                self.item_total_num[self.item2index[target_item]] += 1
                if is_train:
                    self.train_item_total_num[self.item2index[item_id]] += 1
            else:
                continue

        self.session_count += 1
        self.session2index[last_session_id] = self.session_count
        new_session = SessionData(self.session_count, last_session_id, session_item_indexes)
        result_data.append(new_session)
        print("loaded")
        print(new_session)

        return result_data

    def get_batch(self, batch_size, session_length=10, predict_length=1, all_data=None, phase="train", neg_num=1,
                  sampling_mathod="random"):

        if phase == "train":
            if all_data is None:
                all_data = self.get_all_training_data(session_length)
            indexes = np.random.permutation(all_data.shape[0])
            all_data = all_data[indexes]
        else:
            if all_data is None:
                all_data = self.get_all_testing_data(session_length)

        sindex = 0
        eindex = batch_size
        while eindex < all_data.shape[0]:
            batch = all_data[sindex: eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            if phase == "train":
                batch = self.divid_and_extend_negative_samples(batch, session_length=session_length,
                                                               predict_length=predict_length, neg_num=neg_num,
                                                               method=sampling_mathod)
            else:
                batch = [batch[:, :session_length], batch[:, session_length:]]
            yield batch

        if eindex >= all_data.shape[0]:
            batch = all_data[sindex:]
            if phase == "train":
                batch = self.divid_and_extend_negative_samples(batch, session_length=session_length,
                                                               predict_length=predict_length, neg_num=neg_num,
                                                               method=sampling_mathod)
            else:
                batch = [batch[:, :session_length], batch[:, session_length:]]
            yield batch

    def divid_and_extend_negative_samples(self, batch_data, session_length, predict_length=1, neg_num=1,
                                          method="random"):
        neg_items = []
        if method == "random":
            for session_and_target in batch_data:
                neg_item = []
                for i in range(neg_num):
                    rand_item = random.randint(1, self.index_count)
                    while rand_item in session_and_target or rand_item in neg_item:
                        rand_item = random.randint(1, self.index_count)
                    neg_item.append(rand_item)
                neg_items.append(neg_item)
        else:
            total_list = set()
            for session in batch_data:
                for i in session:
                    total_list.add(i)
            total_list = list(total_list)
            total_list = sorted(total_list, key=lambda item: self.item_total_num[item], reverse=True)
            for i, session in enumerate(batch_data):
                np.random.choice(total_list)
        session_items = batch_data[:, :session_length]
        target_item = batch_data[:, session_length:]
        neg_items = np.array(neg_items)
        return [session_items, target_item, neg_items]

    def get_all_training_data(self, session_length, predict_length=1):
        if len(self.all_training_data) != 0 and self.train_session_length == session_length:
            return self.all_training_data
        print("Start building the all training dataset")
        all_sessions = []
        for session_data in self.train_data:
            session_index, sessions = session_data.generate_train_datas(session_length, padding_idx=self.padding_idx)
            if sessions is not None:
                all_sessions.extend(sessions)
        all_sessions = np.array(all_sessions)
        self.all_training_data = all_sessions
        self.train_session_length = session_length
        return all_sessions

    def get_all_testing_data(self, session_length, predict_length=1):
        if len(self.all_testing_data) != 0 and self.test_session_length == session_length:
            return self.all_testing_data
        all_sessions = []
        for session_data in self.test_data:
            session_index, sessions = session_data.generate_train_datas(session_length, padding_idx=self.padding_idx)
            if sessions is not None:
                all_sessions.extend(sessions)
        all_sessions = np.array(all_sessions)
        self.all_testing_data = all_sessions
        self.test_session_length = session_length
        return all_sessions

    def get_train_most_popular_items(self, k):
        train_item_total_num = np.array(list(self.train_item_total_num.values()))

        top_k_index = np.argsort(train_item_total_num)[::-1][:k]
        return np.array(list(self.train_item_total_num.keys()))[top_k_index]

    def get_current_most_popular_items(self, batch_session, k):
        item_total_num = np.array(list(self.item_total_num.values()))
        for i in range(batch_session.shape[0]):
            temp_list = []
            for j, item in enumerate(batch_session[i]):
                if item in temp_list:
                    batch_session[i][j] = 0
                else:
                    temp_list.append(item)

        session_item_num = item_total_num[batch_session - 1]
        sorted_session_item = np.argsort(session_item_num)[:, ::-1]
        session_result = []
        if batch_session.shape[1] >= k:
            for i in range(batch_session.shape[0]):
                session_result.append(batch_session[i][sorted_session_item[i][:k]])
        else:
            pad_zero = np.zeros((batch_session.shape[0], k - batch_session.shape[1]), dtype=np.int)
            for i in range(batch_session.shape[0]):
                data = batch_session[i][sorted_session_item[i]]
                session_result.append(np.concatenate((np.array(data), pad_zero[i]), -1))
        return np.array(session_result)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def bpr_loss(r):
    return torch.sum(-torch.log(torch.sigmoid(r)))


def get_hit_num(pred, y_truth):
    """
        pred: numpy type(batch_size,k)
        y_truth: list type (batch_size,groudtruth_num)
    """

    hit_num = 0
    for i in range(len(y_truth)):
        for value in y_truth[i]:
            hit_num += np.sum(pred[i] == value)
    return hit_num


def get_rr(pred, y_truth):
    rr = 0.
    for i in range(len(y_truth)):
        for value in y_truth[i]:
            hit_indexes = np.where(pred[i] == value)[0]
            for hit_index in hit_indexes:
                rr += 1 / (hit_index + 1)
    return rr


def get_dcg(pred, y_truth):
    y_pred_score = np.zeros_like(pred)

    for i in range(len(y_truth)):

        for j, y_pred in enumerate(pred[i]):
            if y_pred == y_truth[i][0]:
                y_pred_score[i][j] = 1
    gain = 2 ** y_pred_score - 1
    discounts = np.tile(np.log2(np.arange(pred.shape[1]) + 2), (len(y_truth), 1))
    dcg = np.sum(gain / discounts, axis=1)
    return dcg


def get_ndcg(pred, y_truth):
    dcg = get_dcg(pred, y_truth)
    idcg = get_dcg(np.concatenate((y_truth, np.zeros_like(pred)[:, :-1] - 1), axis=1), y_truth)
    ndcg = np.sum(dcg / idcg)

    return ndcg


def dcg_score(y_pre, y_true, k):
    y_pre_score = np.zeros(k)
    if len(y_pre) > k:
        y_pre = y_pre[:k]
    for i in range(len(y_pre)):
        pre_tag = y_pre[i]
        if pre_tag in y_true:
            y_pre_score[i] = 1
    gain = 2 ** y_pre_score - 1
    discounts = np.log2(np.arange(k) + 2)
    return np.sum(gain / discounts)


def ndcg_score(y_pre, y_true, k=5):
    dcg = dcg_score(y_pre, y_true, k)
    idcg = dcg_score(y_true, y_true, k)
    return dcg / idcg


def pop():
    predict_nums = [1, 5, 10, 20]
    session_length = 20
    pop_k = dataset.get_train_most_popular_items(predict_nums[-1])
    start_test_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start testing", start_test_time)
    rrs = [0 for _ in range(len(predict_nums))]
    hit_nums = [0 for _ in range(len(predict_nums))]
    ndcgs = [0 for _ in range(len(predict_nums))]
    test_num = 0
    for i, batch_data in enumerate(dataset.get_batch(batch_size, session_length, phase="test")):
        target_items = np.array(batch_data[1])
        test_num += len(target_items)
        y_pred = np.tile(pop_k, (batch_size, 1))
        for j, predict_num in enumerate(predict_nums):
            hit_nums[j] += get_hit_num(y_pred[:, :predict_num], target_items)
            rrs[j] += get_rr(y_pred[:, :predict_num], target_items)
            ndcgs[j] += get_ndcg(y_pred[:, :predict_num], target_items)
    end_test_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hrs = [hit_num / test_num for hit_num in hit_nums]
    mrrs = [rr / test_num for rr in rrs]
    mndcgs = [ndcg / test_num for ndcg in ndcgs]
    print("testing over [%s] " % end_test_time)
    for k, predict_num in enumerate(predict_nums):
        print("\tHR@%d=%.5f  MRR@%d=%.5f  NDCG@%d=%.5f" % (
            predict_num, hrs[k], predict_num, mrrs[k], predict_num, mndcgs[k]))


def spop():
    predict_nums = [1, 5, 10, 20]
    session_length = 20
    pop_k = dataset.get_train_most_popular_items(predict_nums[-1])
    start_test_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start testing", start_test_time)
    rrs = [0 for _ in range(len(predict_nums))]
    hit_nums = [0 for _ in range(len(predict_nums))]
    ndcgs = [0 for _ in range(len(predict_nums))]
    test_num = 0
    for i, batch_data in enumerate(dataset.get_batch(batch_size, session_length, phase="test")):
        target_items = np.array(batch_data[1])
        test_num += len(target_items)
        y_pred = dataset.get_current_most_popular_items(batch_data[0], predict_nums[-1])
        for j, predict_num in enumerate(predict_nums):
            hit_nums[j] += get_hit_num(y_pred[:, :predict_num], target_items)
            rrs[j] += get_rr(y_pred[:, :predict_num], target_items)
            ndcgs[j] += get_ndcg(y_pred[:, :predict_num], target_items)
    end_test_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hrs = [hit_num / test_num for hit_num in hit_nums]
    mrrs = [rr / test_num for rr in rrs]
    mndcgs = [ndcg / test_num for ndcg in ndcgs]
    print("testing over [%s] " % end_test_time)
    for k, predict_num in enumerate(predict_nums):
        print("\tHR@%d=%.5f  MRR@%d=%.5f  NDCG@%d=%.5f" % (
            predict_num, hrs[k], predict_num, mrrs[k], predict_num, mndcgs[k]))


# dataset = SessionDataSet(train_file="../data/diginetica/train.txt", test_file="data/diginetica/test.txt")
dataset = SessionDataSet(train_file="../data/retailrocket/train.txt", test_file="data/retailrocket/test.txt")

session_length = 145
batch_size = 1
plot_num = 500
epochs = 30

model = pop()
# model = spop()
