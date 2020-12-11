import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import pickle
import time


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[1]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[2])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]

        items, n_node, A, alias_inputs = [], [], [], []

        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))

        max_n_node = np.max(n_node)

        for u_input in inputs:

            node = np.unique(u_input)

            items.append(node.tolist() + (max_n_node - len(node)) * [0])

            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):

                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

                u_A[v][u] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        return alias_inputs, A, items, mask, targets


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()

        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # inputs.shape->(batch_size,max_session_len,hidden_size * 2)
        inputs = torch.cat([input_in, input_out], 2)
        # gi.shape=gh.shape->(batch_size,max_session_len,hidden_size * 3)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        #  i_r.shape=i_i.shape=i_n.shape->(batch_size,max_session_len,hidden_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def GNNCell2(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah

        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)

        hy = (1 - inputgate) * hidden + inputgate * newgate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid

        self.embedding = nn.Embedding(self.n_node, self.hidden_size)

        self.gnn = GNN(self.hidden_size, step=opt.step)

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):

        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        # (batch_size,latent_size)->(batch_size,1,latent_size)
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        # (batch_size,seq_length,latent_size)
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size

        alpha = self.linear_three(torch.sigmoid(q1 + q2))

        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


device = torch.device("cuda:0")


def trans_to_cuda(variable):
    return variable.to(device)


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    model.optimizer.step()
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    predict_nums = [1, 5, 10, 20]
    model.eval()
    hit, mrr, ndcg = [[] for _ in range(len(predict_nums))], [[] for _ in range(len(predict_nums))], [[] for _ in
                                                                                                      range(len(
                                                                                                          predict_nums))]
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            for j, k in enumerate(predict_nums):
                hit[j].append(np.isin(target - 1, score[:k]))
                if len(np.where(score[:k] == target - 1)[0]) == 0:
                    mrr[j].append(0)
                    ndcg[j].append(0)
                else:
                    mrr[j].append(1 / (np.where(score[:k] == target - 1)[0][0] + 1))
                    ndcg[j].append(1 / np.log2(np.where(score[:k] == target - 1)[0][0] + 2))
    hit = np.mean(hit, axis=1)
    mrr = np.mean(mrr, axis=1)
    ndcg = np.mean(ndcg, axis=1)
    return hit, mrr, ndcg


class Config(object):
    def __init__(self):
        # self.dataset = "diginetica_3"
        self.dataset = "retailrocket"
        self.batchSize = 100  # original: 100
        self.hiddenSize = 100  # original: 100
        self.epoch = 10
        self.lr = 1e-3  # original: 1e-3
        self.lr_dc = 0.1  # orginal: 1e-1
        self.lr_dc_step = 3  # orginal: 3
        self.l2 = 1e-5  # orginal: 1e-5
        self.step = 1
        self.patience = 100
        self.nonhybrid = False
        self.validation = False
        self.valid_portion = 0.1


def main():
    opt = Config()
    print(opt)
    print(
        f"dataset:{opt.dataset},batchsize:{opt.batchSize},hiddenSize:{opt.hiddenSize},epoch:{opt.epoch}"
        f",lr:{opt.lr},lr_dc:{opt.lr_dc},lr_dc_step:{opt.lr_dc_step},l2:{opt.l2},step:{opt.step},"
        f"patience:{opt.patience}")

    train_data = pickle.load(open('../data/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../data/' + opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=False)
    test_data = Data(test_data, shuffle=False)

    # n_node should be bigger than the biggest idx of item (not the sum of the items)
    if "_3" in opt.dataset:
        if "diginetica" in opt.dataset:
            n_node = 40841
        elif 'yoochoose1_64' in opt.dataset:
            n_node = 15834
        elif 'yoochoose1_4' in opt.dataset:
            n_node = 27053
        elif "retailrocket" in opt.dataset:
            n_node = 36969
        else:
            n_node = 310
    else:
        if "diginetica" in opt.dataset:
            n_node = 43098
        elif 'yoochoose1_64' in opt.dataset:
            n_node = 37484
        elif 'yoochoose1_4' in opt.dataset:
            n_node = 37484
        elif "retailrocket" in opt.dataset:
            n_node = 48990
        else:
            n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))
    predict_nums = [1, 5, 10, 20]
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        print(train_data.inputs.shape)
        hit, mrr, ndcg = train_test(model, train_data, test_data)
        flag = 0
        print('Current Result:')
        for i, k in enumerate(predict_nums):
            print('\HR@%d:\t%.5f\tMMR@%d:\t%.5f\tNDCG@%d:\t%.5f' % (k, hit[i], k, mrr[i], k, ndcg[i]))
        if hit[-1] >= best_result[0]:
            best_result[0] = hit[-1]
            best_epoch[0] = epoch
            flag = 1
        if mrr[-1] >= best_result[1]:
            best_result[1] = mrr[-1]
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\HR@20:\t%.5f\tMMR@20:\t%.5f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
