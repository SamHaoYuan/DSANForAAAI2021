import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import math
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy
import os
import datetime
import pickle


class SessionData(object):
    def __init__(self, session_index, session_id, items_indexes):
        self.session_index = session_index
        self.session_id = session_id
        self.item_list = items_indexes

    def generate_seq_datas(self, session_length, padding_idx=0, predict_length=1):
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

    def load_data(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
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
            session_item_indexes.append(self.item2index[item_id])
            self.item_total_num[self.item2index[item_id]] += 1
        target_item = session_label[0]
        if target_item not in self.item2index.keys():
            self.index_count += 1
            self.item2index[target_item] = self.index_count
            self.index2item[self.index_count] = target_item
            self.item_total_num[self.index_count] = 0
        session_item_indexes.append(self.item2index[target_item])
        self.item_total_num[self.item2index[target_item]] += 1

        for session_id, items, target_item in zip(session_ids, session_data, session_label):
            if session_id != last_session_id:

                self.session_count += 1
                self.session2index[last_session_id] = self.session_count
                self.index2session[self.session_count] = last_session_id
                if len(session_item_indexes) > self.max_session_length:
                    self.max_session_length = len(session_item_indexes)
                new_session = SessionData(self.session_count, last_session_id, session_item_indexes)
                result_data.append(new_session)
                last_session_id = session_id
                session_item_indexes = []
                for item_id in items:
                    if item_id not in self.item2index.keys():
                        self.index_count += 1
                        self.item2index[item_id] = self.index_count
                        self.index2item[self.index_count] = item_id
                        self.item_total_num[self.index_count] = 0
                    session_item_indexes.append(self.item2index[item_id])
                    self.item_total_num[self.item2index[item_id]] += 1
                if target_item not in self.item2index.keys():
                    self.index_count += 1
                    self.item2index[target_item] = self.index_count
                    self.index2item[self.index_count] = target_item
                    self.item_total_num[self.index_count] = 0
                session_item_indexes.append(self.item2index[target_item])
                self.item_total_num[self.item2index[target_item]] += 1
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
            session_index, sessions = session_data.generate_seq_datas(session_length, padding_idx=self.padding_idx)
            if sessions is not None:
                all_sessions.extend(sessions)
        all_sessions = np.array(all_sessions)
        self.all_training_data = all_sessions
        self.train_session_length = session_length
        print("The total number of training samples is:", all_sessions.shape)
        return all_sessions

    def get_all_testing_data(self, session_length, predict_length=1):
        if len(self.all_testing_data) != 0 and self.test_session_length == session_length:
            return self.all_testing_data
        all_sessions = []
        for session_data in self.test_data:
            session_index, sessions = session_data.generate_seq_datas(session_length, padding_idx=self.padding_idx)
            if sessions is not None:
                all_sessions.extend(sessions)
        all_sessions = np.array(all_sessions)
        self.all_testing_data = all_sessions
        self.test_session_length = session_length
        print("The total number of testing samples is:", all_sessions.shape)
        return all_sessions

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def bpr_loss(r):
    return torch.sum(-torch.log(torch.sigmoid(r)))


def get_hit_num(pred, y_truth):
    """
    get hit num, HR@k the k is determined by the size of pred
    :param pred: numpy (batch_size,k)
    :param y_truth: list[batch_size][ground_truth_num]
    :return: hit num
    """

    hit_num = 0
    for i in range(len(y_truth)):
        for value in y_truth[i]:
            hit_num += np.sum(pred[i] == value)
    return hit_num


def get_rr(pred, y_truth):
    """
    get MRR value, MRR@k the k is determined by the size of pred
    :param pred: numpy (batch_size,k)
    :param y_truth: list[batch_size][ground_truth_num]
    :return: the value of MRR
    """
    rr = 0.
    for i in range(len(y_truth)):
        for value in y_truth[i]:
            hit_indexes = np.where(pred[i] == value)[0]
            for hit_index in hit_indexes:
                rr += 1 / (hit_index + 1)
    return rr


def get_dcg(pred, y_truth):
    """
    get DCG value, the k is determined by the size of pred
    :param pred: numpy (batch_size,k)
    :param y_truth: list[batch_size][ground_truth_num]
    :return: the value of DCG
    """
    y_pred_score = np.zeros_like(pred)

    for i in range(len(y_truth)):
        for j, y_pred in enumerate(pred[i]):
            if y_pred == y_truth[i][0]:
                y_pred_score[i][j] = 1
    gain = 2 ** y_pred_score - 1  # DCG Numerator
    discounts = np.tile(np.log2(np.arange(pred.shape[1]) + 2), (len(y_truth), 1))  # DCG dominator
    dcg = np.sum(gain / discounts, axis=1)
    return dcg


def get_ndcg(pred, y_truth):
    """
    get NDCG value,NDCG@k the k is determined by the size of pred
    :param pred: numpy (batch_size,k)
    :param y_truth: list[batch_size][ground_truth_num]
    :return: the value of NDCG
    """
    dcg = get_dcg(pred, y_truth)
    idcg = get_dcg(np.concatenate((y_truth, np.zeros_like(pred)[:, :-1] - 1), axis=1), y_truth)
    ndcg = np.sum(dcg / idcg)

    return ndcg


def dcg_score(y_pre, y_true, k=20):
    """
    get DCG@k
    :param y_pre: numpy (batch_size,x)
    :param y_true: y_truth: list[batch_size][ground_truth_num]
    :param k: k
    :return: DCG@k
    """
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


def ndcg_score(y_pre, y_true, k=20):
    """
    get NDCG@k
    :param y_pre: numpy (batch_size,x)
    :param y_true: y_truth: list[batch_size][ground_truth_num]
    :param k: k
    :return: NDCG@k
    """
    dcg = dcg_score(y_pre, y_true, k)
    idcg = dcg_score(y_true, y_true, k)
    return dcg / idcg


class Attention(torch.nn.Module):
    def __init__(self, method="specific", hidden_size=64):
        super(Attention, self).__init__()
        self.config = list()
        self.method = method

        self.hidden_size = hidden_size

        if self.method == "dot":

            self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)

        elif self.method == "general":
            self.attention = torch.nn.Linear(self.hidden_size, self.hidden_size)

        elif self.method == "concat":
            self.attention = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))

        elif self.method == "specific":
            self.W0 = torch.nn.Linear(self.hidden_size, 1, bias=False)
            torch.nn.init.normal_(self.W0.weight, 0, 0.05)
            self.W1 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            torch.nn.init.normal_(self.W1.weight, 0, 0.05)
            self.W2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            torch.nn.init.normal_(self.W2.weight, 0, 0.05)
            self.W3 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            torch.nn.init.normal_(self.W3.weight, 0, 0.05)
            self.b = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_output, weights=None):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output, weights=None):
        energy = self.attention(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output, weights=None):
        energy = self.attention(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def specific_score(self, session, x_t, m_s, mask=None, weights=None):
        if weights is None:
            if mask is None:
                W1Xi = self.W1(session)
                W2Xt = self.W2(x_t).unsqueeze(1).repeat((1, session.shape[1], 1))
                W3Ms = self.W3(m_s).unsqueeze(1).repeat((1, session.shape[1], 1))
                energy = self.W0(torch.sigmoid(W1Xi + W2Xt + W3Ms + self.b)).repeat((1, 1, session.shape[2]))
            else:
                W1Xi = self.W1(session) * mask
                W2Xt = self.W2(x_t).unsqueeze(1).repeat((1, session.shape[1], 1)) * mask
                W3Ms = self.W3(m_s).unsqueeze(1).repeat((1, session.shape[1], 1)) * mask
                energy = self.W0(torch.sigmoid(W1Xi + W2Xt + W3Ms + self.b)).repeat((1, 1, session.shape[2])) * mask
        else:
            key = 1
            if mask is None:
                W1Xi = torch.matmul(session, weights[key].t())
                key += 1
                W2Xt = (torch.matmul(x_t, weights[key].t())).unsqueeze(1).repeat((1, session.shape[1], 1))
                key += 1
                W3Ms = (torch.matmul(x_t, weights[key].t())).unsqueeze(1).repeat((1, session.shape[1], 1))
                energy = torch.matmul(torch.sigmoid(W1Xi + W2Xt + W3Ms + weights[key + 1]), weights[0].t()).repeat(
                    (1, 1, session.shape[2]))
            else:
                W1Xi = torch.matmul(session, weights[key].t()) * mask
                key += 1
                W2Xt = (torch.matmul(x_t, weights[key].t())).unsqueeze(1).repeat((1, session.shape[1], 1)) * mask
                key += 1
                W3Ms = (torch.matmul(x_t, weights[key].t())).unsqueeze(1).repeat((1, session.shape[1], 1)) * mask
                energy = torch.matmul(torch.sigmoid(W1Xi + W2Xt + W3Ms + weights[key + 1]), weights[0].t()).repeat(
                    (1, 1, session.shape[2])) * mask
        return torch.sum(energy * session, dim=1)

    def forward(self, hidden, encoder_outputs=None, x_t=None, mask=None):

        if self.method == "general":
            attention_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attention_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attention_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == "specific":
            session = hidden
            m_s = encoder_outputs
            return self.specific_score(session, x_t, m_s, mask)

        attention_energies = attention_energies.t()

        return F.softmax(attention_energies, dim=1).unsqueeze(1)


class STAMP(torch.nn.Module):
    def __init__(self, hidden_size=64, item_num=0, padding_idx=0, dropout=0.5, activate="relu"):
        """
        STAMP
        :param hidden_size: embedding size
        :param item_num: the number of item
        :param padding_idx: padding idx
        :param dropout: dropout
        :param activate: activate function name
        """
        super(STAMP, self).__init__()
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size

        if activate == "sigmoid":
            self.activate = torch.sigmoid
        elif activate == "tanh":
            self.activate = torch.tanh
        else:
            self.activate = torch.relu

        self.dropout = torch.nn.Dropout(dropout)

        self.item_embedding = torch.nn.Embedding(item_num, hidden_size, padding_idx=self.padding_idx, max_norm=1.5)
        torch.nn.init.normal_(self.item_embedding.weight, 0, 0.002)
        torch.nn.init.constant_(self.item_embedding.weight[0], 0)

        self.attention = Attention(method="specific", hidden_size=hidden_size)

        self.left_mlp1 = torch.nn.Linear(hidden_size, hidden_size)
        torch.nn.init.normal_(self.left_mlp1.weight, 0, 0.05)
        torch.nn.init.constant_(self.left_mlp1.bias, 0)

        self.right_mlp1 = torch.nn.Linear(hidden_size, hidden_size)
        torch.nn.init.normal_(self.left_mlp1.weight, 0, 0.05)
        torch.nn.init.constant_(self.right_mlp1.bias, 0)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=110)

    def forward(self, session):
        mask = (session != self.padding_idx).float()

        length = torch.sum(mask, 1).unsqueeze(1).repeat((1, self.hidden_size))

        mask = mask.unsqueeze(2).repeat((1, 1, self.hidden_size))
        session_item_vecs = self.item_embedding(session) * mask
        mean_session = torch.sum(session_item_vecs, dim=1) / length

        compute_output = self.attention(session_item_vecs, mean_session, session_item_vecs[:, -1])
        left_output = self.dropout(self.activate(self.left_mlp1(compute_output)))
        right_output = self.dropout(self.activate(self.right_mlp1(session_item_vecs[:, -1])))

        result = torch.matmul(left_output * right_output, self.item_embedding.weight[1:].t())

        return result

    def predict_top_k(self, session, k=20):

        mask = (session != 0).float()

        length = torch.sum(mask, 1).unsqueeze(1).repeat((1, self.hidden_size))

        mask = mask.unsqueeze(2).repeat((1, 1, self.hidden_size))
        session_item_vecs = self.item_embedding(session) * mask
        mean_session = torch.sum(session_item_vecs, dim=1) / length
        compute_output = self.attention(session_item_vecs, mean_session, session_item_vecs[:, -1])
        left_output = self.activate(self.left_mlp1(compute_output))

        right_output = self.activate(self.right_mlp1(session_item_vecs[:, -1]))

        result = torch.matmul(left_output * right_output, self.item_embedding.weight[1:].t())

        result = torch.topk(result, k, dim=1)[1]
        return result


def train(args):
    hidden_size = args["hidden_size"] if "hidden_size" in args.keys() else 100
    dropout = args["dropout"] if "dropout" in args.keys() else 0.5
    lr = args["lr"] if "lr" in args.keys() else 3e-3
    session_length = args["session_length"] if "session_length" in args.keys() else 20

    model = STAMP(hidden_size=hidden_size, item_num=dataset.index_count + 1, padding_idx=0, dropout=dropout,
                  activate="tanh").to(device)

    opti = torch.optim.Adam(model.parameters(), lr=lr)

    best_model_hr = 0.0
    best_model_mrr = 0.0
    best_model_ndcg = 0.0
    best_r1m = 0.0
    best_model = None

    predict_nums = [1, 5, 10, 20]
    for epoch in range(epochs):
        loss_sum = 0
        batch_losses = []
        epoch_losses = []
        for i, batch_data in enumerate(dataset.get_batch(batch_size, session_length, phase="train")):
            sessions = torch.tensor(batch_data[0]).to(device)
            target_items = torch.tensor(batch_data[1]).squeeze().to(device) - 1
            result_pos = model(sessions)
            loss = loss_function(result_pos, target_items)
            opti.zero_grad()
            loss.backward()
            opti.step()
            batch_losses.append(loss.cpu().detach().numpy())
            epoch_losses.append(loss.cpu().detach().numpy())
            if i % plot_num == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("[%s] [%d/%d] %d mean_batch_loss : %.6f" % (
                    time, epoch + 1, epochs, i, np.mean(batch_losses)))
                loss_sum = loss_sum + np.array(batch_losses).sum()
                batch_losses = []
        with torch.no_grad():
            start_test_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("Start predicting", start_test_time)
            rrs = [0 for _ in range(len(predict_nums))]
            hit_nums = [0 for _ in range(len(predict_nums))]
            ndcgs = [0 for _ in range(len(predict_nums))]
            for i, batch_data in enumerate(dataset.get_batch(batch_size, session_length, phase="test")):
                sessions = torch.tensor(batch_data[0]).to(device)
                mask = sessions != 0
                a = sessions[mask].tolist()
                target_items = np.array(batch_data[1]) - 1
                y_pred = model.predict_top_k(sessions, 20).cpu().numpy()
                for j, predict_num in enumerate(predict_nums):
                    hit_nums[j] += get_hit_num(y_pred[:, :predict_num], target_items)
                    rrs[j] += get_rr(y_pred[:, :predict_num], target_items)
                    ndcgs[j] += get_ndcg(y_pred[:, :predict_num], target_items)

            end_test_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            hrs = [hit_num / len(dataset.all_testing_data) for hit_num in hit_nums]
            mrrs = [rr / len(dataset.all_testing_data) for rr in rrs]
            mndcgs = [ndcg / len(dataset.all_testing_data) for ndcg in ndcgs]
            if hrs[-1] + mrrs[-1] > best_r1m:
                print("change best")
                best_model = deepcopy(model)
                best_model_hr = hrs[-1]
                best_model_mrr = mrrs[-1]
                best_model_ndcg = mndcgs[-1]
                best_r1m = hrs[-1] + mrrs[-1]
                no_improvement_epoch = 0
            else:
                no_improvement_epoch += 1
            print("testing finish [%s] " % end_test_time)
            for k, predict_num in enumerate(predict_nums):
                print("\tHR@%d=%.5f  MRR@%d=%.5f  NDCG@%d=%.5f" % (
                    predict_num, hrs[k], predict_num, mrrs[k], predict_num, mndcgs[k]))
        if no_improvement_epoch >= patience:
            print("early stopping")
            break
        print("epoch: %d, total loss: %0.6f" % (epoch, loss_sum))
    return best_model, best_model_hr, best_model_mrr, best_model_ndcg


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    session_length = [20]
    batch_size = 512
    plot_num = 500
    epochs = 50
    loss_function = torch.nn.CrossEntropyLoss()

    # dataset = SessionDataSet(train_file="../data/diginetica/train.txt", test_file="../data/diginetica/test.txt")
    dataset = SessionDataSet(train_file="../data/retailrocket/train.txt", test_file="../data/retailrocket/test.txt")

    # original: hidden_size:100, lr:5e-3
    hidden_sizes = [100]
    dropouts = [0]
    lrs = [5e-3]
    session_lengths = [289]
    patience = 100
    best_params = ""
    best_all_model = 0.0
    best_all_hr = 0.0
    best_all_mrr = 0.0
    best_all_ndcg = 0.0
    best_all_r1m = 0.0
    for session_length in session_lengths:
        for hidden_size in hidden_sizes:
            for dropout in dropouts:
                for lr in lrs:
                    args = {}
                    print(
                        "current model hyper-parameters: session_length=%d, hidden_size=%d, lr=%.4f, dropout=%.2f\n" % (
                            session_length, hidden_size, lr, dropout))
                    args["session_length"] = session_length
                    args["hidden_size"] = hidden_size
                    args["dropout"] = dropout
                    args["patience"] = patience
                    args["lr"] = lr

                    best_model, best_model_hr, best_model_mrr, best_model_ndcg = train(args)

                    if best_model_hr + best_model_mrr > best_all_r1m:
                        print("best model change")
                        best_all_r1m = best_model_hr + best_model_mrr
                        best_all_hr = best_model_hr
                        best_all_mrr = best_model_mrr
                        best_all_ndcg = best_model_ndcg
                        best_all_model = best_model
                        best_params = "session_length=%d, hidden_size=%d, lr=%.4f, dropout=%.2f\n" % (
                            session_length, hidden_size, lr, dropout)
                    best_model = None
                    print(
                        "current model hyper-parameters: session_length=%d, hidden_size=%d, lr=%.4f, dropout=%.2f\n" % (
                            session_length, hidden_size, lr, dropout))
                    print("current model Recall@20=%.5f  MRR@20=%.5f" % (best_all_hr, best_model_mrr))
                    print("the best result so far. Recall@20=%.5f  MRR@20=%.5f, params:%s \n" % (
                        best_all_hr, best_all_mrr, best_params))
    print("The best result HR@20=%.5f  MRR@20=%.5f, hyper-parameters: %s. " % (best_all_hr, best_all_mrr, best_params))
    print("over.")
