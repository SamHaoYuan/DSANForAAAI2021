import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import tqdm
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from copy import deepcopy
import os
import datetime
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda:0")
# device = torch.device("cpu")

session_length = 20
batch_size = 512
plot_num = 5000
epochs = 30


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
            for i in range(len(self.item_list) - 1):
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

    def get_batch_with_neg(self, batch_size, session_length=10, predict_length=1, all_data=None, phase="train",
                           neg_num=1, sampling_mathod="random"):
        if phase == "train":
            all_data = self.get_all_training_data_with_neg(session_length, neg_num)
            indexes = np.random.permutation(all_data.shape[0])
            all_data = all_data[indexes]
        else:
            all_data = self.get_all_testing_data_with_neg(session_length, neg_num)

        sindex = 0
        eindex = batch_size
        while eindex < all_data.shape[0]:
            batch = all_data[sindex: eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            if phase == "train":
                batch = [batch[:, :session_length], batch[:, session_length:session_length + predict_length],
                         batch[:, -neg_num:]]
            else:
                batch = [batch[:, :session_length], batch[:, session_length:]]
            yield batch

        if eindex >= all_data.shape[0]:
            batch = all_data[sindex:]
            if phase == "train":
                batch = [batch[:, :session_length], batch[:, session_length:session_length + predict_length],
                         batch[:, -neg_num:]]
            else:
                batch = [batch[:, :session_length], batch[:, session_length:]]
            yield batch

    def get_batch_tasks_with_neg(self, batch_size, session_length=10, predict_length=1, all_data=None, phase="train",
                                 neg_num=1, sampling_mathod="random"):
        if phase == "train":
            all_data = self.get_all_meta_training_data_with_neg(session_length, neg_num)
            random.shuffle(all_data)
        else:
            all_data = self.get_all_meta_testing_data_with_neg(session_length, neg_num)
        sindex = 0
        eindex = batch_size
        while eindex < len(all_data):
            batch = all_data[sindex: eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp

            session_items = [batch[i][:, :session_length] for i in range(len(batch))]

            target_item = [batch[i][:, session_length:session_length + predict_length] for i in range(len(batch))]

            neg_item = [batch[i][:, -neg_num:] for i in range(len(batch))]
            batch = [session_items, target_item, neg_item]
            yield batch

        if eindex >= len(all_data):
            batch = all_data[sindex:]
            session_items = [batch[i][:, :session_length] for i in range(len(batch))]

            target_item = [batch[i][:, session_length:session_length + predict_length] for i in range(len(batch))]

            neg_item = [batch[i][:, -neg_num:] for i in range(len(batch))]
            batch = [session_items, target_item, neg_item]
            yield batch

    def divid_and_extend_negative_samples(self, batch_data, session_length, predict_length=1, neg_num=1,
                                          method="random"):
        """
        divid and extend negative samples
        """
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
        print("The total number of training samples is", all_sessions.shape)
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
        print("The total number of testing samples is", all_sessions.shape)
        return all_sessions

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


loss_function = torch.nn.CrossEntropyLoss()


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=2, dropout=0, initializer_range=0.02):
        super(MultiHeadSelfAttention, self).__init__()
        self.config = list()

        self.hidden_size = hidden_size

        self.head_num = head_num
        if (self.hidden_size) % head_num != 0:
            raise ValueError(self.head_num, "error")
        self.head_dim = self.hidden_size // self.head_num

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.concat_weight = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        torch.nn.init.normal_(self.query.weight, 0, initializer_range)
        torch.nn.init.normal_(self.key.weight, 0, initializer_range)
        torch.nn.init.normal_(self.value.weight, 0, initializer_range)
        torch.nn.init.normal_(self.concat_weight.weight, 0, initializer_range)
        self.dropout = torch.nn.Dropout(dropout)

    def dot_score(self, encoder_output):
        query = self.dropout(self.query(encoder_output))
        key = self.dropout(self.key(encoder_output))
        # head_num * batch_size * session_length * head_dim
        querys = torch.stack(query.chunk(self.head_num, -1), 0)
        keys = torch.stack(key.chunk(self.head_num, -1), 0)
        # head_num * batch_size * session_length * session_length
        dots = querys.matmul(keys.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        #         print(len(dots),dots[0].shape)
        return dots

    def forward(self, encoder_outputs, mask=None):
        attention_energies = self.dot_score(encoder_outputs)
        value = self.dropout(self.value(encoder_outputs))

        values = torch.stack(value.chunk(self.head_num, -1))

        if mask is not None:
            eye = torch.eye(mask.shape[-1]).to(device)
            new_mask = torch.clamp_max((1 - (1 - mask.float()).unsqueeze(1).permute(0, 2, 1).bmm(
                (1 - mask.float()).unsqueeze(1))) + eye, 1)
            attention_energies = attention_energies - new_mask * 1e12
            weights = F.softmax(attention_energies, dim=-1)
            weights = weights * (1 - new_mask)
        else:
            weights = F.softmax(attention_energies, dim=2)

        # head_num * batch_size * session_length * head_dim
        outputs = weights.matmul(values)
        # batch_size * session_length * hidden_size
        outputs = torch.cat([outputs[i] for i in range(outputs.shape[0])], dim=-1)
        outputs = self.dropout(self.concat_weight(outputs))

        return outputs


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, initializer_range=0.02):
        super(PositionWiseFeedForward, self).__init__()
        self.final1 = torch.nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.final2 = torch.nn.Linear(hidden_size * 4, hidden_size, bias=True)
        torch.nn.init.normal_(self.final1.weight, 0, initializer_range)
        torch.nn.init.normal_(self.final2.weight, 0, initializer_range)

    def forward(self, x):
        x = F.gelu(self.final1(x))
        x = self.final2(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=2, dropout=0, attention_dropout=0,
                 initializer_range=0.02):
        super(TransformerLayer, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.mh = MultiHeadSelfAttention(hidden_size=hidden_size, activate=activate, head_num=head_num,
                                         dropout=attention_dropout, initializer_range=initializer_range)
        self.pffn = PositionWiseFeedForward(hidden_size, initializer_range=initializer_range)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, encoder_outputs, mask=None):
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.mh(encoder_outputs, mask)))
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.pffn(encoder_outputs)))
        return encoder_outputs


class BERT(torch.nn.Module):
    def __init__(self, hidden_size=100, itemNum=0, posNum=0, padding_idx=0, dropout=0.5, attention_dropout=0,
                 head_num=2, sa_layer_num=1,
                 activate="relu", initializer_range=0.02):
        super(BERT, self).__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.session_length = session_length
        self.sa_layer_num = sa_layer_num
        self.transformers = torch.nn.ModuleList([TransformerLayer(hidden_size, head_num=head_num, dropout=dropout,
                                                                  attention_dropout=attention_dropout,
                                                                  initializer_range=initializer_range) for _ in
                                                 range(sa_layer_num)])

    def forward(self, compute_output, attention_mask):
        for sa_i in range(self.sa_layer_num):
            compute_output = self.transformers[sa_i](compute_output, attention_mask)
        return compute_output


class BERT4Rec(torch.nn.Module):
    def __init__(self, hidden_size=64, itemNum=0, posNum=0, padding_idx=0, dropout=0.5, attention_dropout=0, head_num=2,
                 sa_layer_num=1,
                 activate="relu", initializer_range=0.02):
        super(BERT4Rec, self).__init__()
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.session_length = session_length
        self.sa_layer_num = sa_layer_num
        self.activate = torch.relu
        self.dropout = torch.nn.Dropout(dropout)

        self.mask_index = torch.tensor(itemNum + 1).to(device)
        self.mask_position = torch.tensor(posNum + 1).to(device)
        self.item_embedding = torch.nn.Embedding(itemNum + 2, hidden_size, padding_idx=self.padding_idx)
        self.position_embedding = torch.nn.Embedding(posNum + 2, hidden_size, padding_idx=self.padding_idx)
        self.bert = BERT(hidden_size=hidden_size, dropout=dropout, attention_dropout=attention_dropout,
                         head_num=head_num, sa_layer_num=sa_layer_num,
                         activate=activate, initializer_range=initializer_range)

        torch.nn.init.normal_(self.item_embedding.weight, 0, initializer_range)
        torch.nn.init.constant_(self.item_embedding.weight[0], 0)
        torch.nn.init.normal_(self.position_embedding.weight, 0, initializer_range)
        torch.nn.init.constant_(self.position_embedding.weight[0], 0)
        self.projection = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        torch.nn.init.normal_(self.projection.weight, 0, initializer_range)
        self.output_bias = torch.nn.Parameter(torch.zeros(itemNum, ))
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, session, mask_indexes=None):

        mask = (session != 0).float()

        mask = mask.unsqueeze(2).repeat((1, 1, self.hidden_size))
        session_item_embeddings = self.item_embedding(session) * mask
        positions = torch.arange(0, session.shape[1]).unsqueeze(0).repeat((session.shape[0], 1)).to(device)
        session_position_embeddings = self.position_embedding(positions) * mask
        session_item_vecs = self.dropout(self.layer_norm(session_item_embeddings + session_position_embeddings))
        attention_mask = (session == self.padding_idx)
        if mask_indexes is not None:
            compute_output = self.dropout(self.bert(session_item_vecs, attention_mask).gather(1, mask_indexes))
        else:
            compute_output = self.dropout(self.bert(session_item_vecs, attention_mask)[:, -1, :])
        compute_output = F.gelu(self.dropout(self.projection(compute_output)))
        result = torch.matmul(compute_output, self.item_embedding.weight[1:-1].t()) + self.output_bias
        return result

    def predict_top_k(self, session, k=20):
        result = self.forward(session)
        result = torch.topk(result, k, dim=1)[1]

        return result


epochs = 50


def train(args):
    hidden_size = args["hidden_size"] if "hidden_size" in args.keys() else 100
    attention_dropout = args["attention_dropout"] if "attention_dropout" in args.keys() else 0.2
    dropout = args["dropout"] if "dropout" in args.keys() else 0.5
    lr = args["lr"] if "lr" in args.keys() else 5e-4
    sa_layer_num = args["sa_layer_num"] if "sa_layer_num" in args.keys() else 1
    amsgrad = args["amsgrad"] if "amsgrad" in args.keys() else True
    session_length = args["session_length"] if "session_length" in args.keys() else 20
    head_num = args["head_num"] if "head_num" in args.keys() else 1
    model = BERT4Rec(hidden_size=hidden_size, itemNum=dataset.index_count, posNum=session_length, padding_idx=0,
                     dropout=dropout,
                     activate="selu", attention_dropout=attention_dropout, head_num=head_num,
                     sa_layer_num=sa_layer_num).to(device)
    opti = torch.optim.Adam(model.parameters(), lr=lr)
    patience = args["patience"] if "patience" in args.keys() else 5
    best_model_hr = 0.0
    best_model_mrr = 0.0
    best_r1m = 0.0
    best_model = None
    predict_nums = [1, 5, 10, 20]
    no_improvement_epoch = 0
    start_train_time = datetime.datetime.now()
    for epoch in range(epochs):
        batch_losses = []
        epoch_losses = []
        model.train()
        for i, batch_data in enumerate(dataset.get_batch(batch_size, session_length, phase="train")):
            mask_item = torch.ones_like(torch.tensor(batch_data[1])) * dataset.index_count + 1
            sessions = torch.cat([torch.tensor(batch_data[0]), mask_item], dim=-1)
            target_items = torch.tensor(batch_data[1]).squeeze().to(device) - 1
            result_pos = model(sessions.to(device))
            loss = loss_function(result_pos, target_items)
            opti.zero_grad()
            loss.backward()
            opti.step()
            batch_losses.append(loss.cpu().detach().numpy())
            epoch_losses.append(loss.cpu().detach().numpy())
            if i % plot_num == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("[%s] [%d/%d] %d mean_batch_loss : %0.6f" % (time, epoch + 1, epochs, i, np.mean(batch_losses)))
                batch_losses = []

        model.eval()
        with torch.no_grad():
            start_test_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("Start predicting", start_test_time)
            rrs = [0 for _ in range(len(predict_nums))]
            hit_nums = [0 for _ in range(len(predict_nums))]
            ndcgs = [0 for _ in range(len(predict_nums))]
            for i, batch_data in enumerate(dataset.get_batch(batch_size, session_length, phase="test")):
                mask_item = torch.ones_like(torch.tensor(batch_data[1])) * dataset.index_count + 1
                sessions = torch.cat([torch.tensor(batch_data[0]), mask_item], dim=-1).to(device)

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
    end_train_time = datetime.datetime.now()
    print("training and testting over, Total time", end_train_time - start_train_time)
    return best_model, best_model_hr, best_model_mrr


# dataset = SessionDataSet(train_file="../data/diginetica/train.txt", test_file="../data/diginetica/test.txt")
dataset = SessionDataSet(train_file="../data/retailrocket/train.txt", test_file="../data/retailrocket/test.txt")

hidden_sizes = [100]  # rr:100 dn:100
dropouts = [0.25]  # rr:0.3 dn:0.3
attention_dropouts = [0]  # rr:0 dn:0
lrs = [1e-3]  # rr:1e-3 dn:5e-4
session_lengths = [50]  # rr:50 dn:50
sa_layer_nums = [2]  # rr:4 dn:4
patience = 5
head_nums = [4]  # rr:2 dn:4
amsgrads = [True]
best_params = ""
best_all_model = 0.0
best_all_hr = 0.0
best_all_mrr = 0.0
best_all_r1m = 0.0
for session_length in session_lengths:
    for hidden_size in hidden_sizes:
        for amsgrad in amsgrads:
            for attention_dropout in attention_dropouts:
                for dropout in dropouts:
                    for lr in lrs:
                        for sa_layer_num in sa_layer_nums:
                            for head_num in head_nums:
                                args = {}
                                print(
                                    "current model hyper-parameters: session_length=%d, hidden_size=%d, lr=%.4f,head_num=%d, amsgrad=%s, attention_dropout=%.2f, dropout=%.2f, sa_layer_num=%d. \n" % (
                                        session_length, hidden_size, lr, head_num, str(amsgrad), attention_dropout,
                                        dropout,
                                        sa_layer_num))
                                args["session_length"] = session_length
                                args["hidden_size"] = hidden_size
                                args["amsgrad"] = amsgrad
                                args["attention_dropout"] = attention_dropout
                                args["dropout"] = dropout
                                args["sa_layer_num"] = sa_layer_num
                                args["lr"] = lr
                                args["head_num"] = head_num
                                args["patience"] = patience
                                best_model, best_model_hr, best_model_mrr = train(args)
                                if best_model_hr + best_model_mrr > best_all_r1m:
                                    print("best model change")
                                    best_all_r1m = best_model_hr + best_model_mrr
                                    best_all_hr = best_model_hr
                                    best_all_mrr = best_model_mrr
                                    best_all_model = best_model
                                    best_params = "session_length-%d, hidden_size-%d, lr-%.4f,head_num=%d, amsgrad-%s, attention_dropout-%.2f, dropout-%.2f, sa_layer_num-%d" % (
                                        session_length, hidden_size, lr, head_num, str(amsgrad), attention_dropout,
                                        dropout,
                                        sa_layer_num)
                                best_model = None
                                print(
                                    "current model hyper-parameters: session_length=%d, hidden_size=%d, lr=%.4f,head_num=%d, amsgrad=%s, attention_dropout=%.2f, dropout=%.2f, sa_layer_num=%d. \n" % (
                                        session_length, hidden_size, lr, head_num, str(amsgrad), attention_dropout,
                                        dropout,
                                        sa_layer_num))
                                print("current model HR@20=%.5f  MRR@20=%.5f." % (best_model_hr, best_model_mrr))
                                print("the best result so far. HR@20=%.5f  MRR@20=%.5f, hyper-parameters: %s. \n" % (
                                    best_all_hr, best_all_mrr, best_params))
print("The best result HR@20=%.5f  MRR@20=%.5f, hyper-parameters: %s. " % (best_all_hr, best_all_mrr, best_params))
print("over.")
