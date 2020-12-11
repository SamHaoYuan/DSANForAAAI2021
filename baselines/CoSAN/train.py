# -*- coding: UTF-8 -*-
import torch
from torch.utils.data import DataLoader, Dataset
import os
import pickle
from CoSAN import CoSAN
from dataloader import SessionDataSet
from metric import cal_hr, cal_mrr, cal_ndcg
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def generate_config(dropouts, heads, lrs, regs, ks, session_lengths, embedding_dims, config):
    configs = []
    for dropout in dropouts:
        for head in heads:
            for lr in lrs:
                for reg in regs:
                    for k in ks:
                        for session_length in session_lengths:
                            for embedding_dim in embedding_dims:
                                new_config = config.copy()
                                new_config["dropout"] = dropout
                                new_config["head"] = head
                                new_config["lr"] = lr
                                new_config["reg"] = reg
                                new_config["k"] = k
                                new_config["session_length"] = session_length
                                new_config["embedding_dim"] = embedding_dim
                                configs += [new_config]
    return configs


dataset = "retailrocket"
training_data = pickle.load(open("../../data/" + dataset + "/train.txt", "rb"))
train_sessions = pickle.load(open(f"./{dataset}/train_sessions.txt", "rb"))
train_neighbors = pickle.load(open(f"./{dataset}/train_neighbors.txt", "rb"))
train_neighbor_weights = pickle.load(open(f"./{dataset}/train_neighbor_weights.txt", "rb"))
training_labels = training_data[2]

testing_data = pickle.load(open("../../data/" + dataset + "/test.txt", "rb"))
test_sessions = pickle.load(open(f"./{dataset}/test_sessions.txt", "rb"))
test_neighbors = pickle.load(open(f"./{dataset}/test_neighbors.txt", "rb"))
test_neighbor_weights = pickle.load(open(f"./{dataset}/test_neighbor_weights.txt", "rb"))
testing_labels = testing_data[2]

config = {
    "training_session_num": 433649,  # dn:526136 rr:433649
    "testing_session_num": 15133,  # dn:44280 rr:15133
    "session_num": 433649,  # dn:526136 rr:433649
    "item_num": 36969,  # dn:40841 rr:36969
    "session_length": 50,  # dn:70 rr:50
    "embedding_dim": 10,
    "alpha": 0.5,
    "head": 1,
    "self_attention_layers": 2,
    "k": 10,  # dn:5 rr:10
    "dropout": 0.3,  # dn:0.25 rr:0.3
    "lr": 1e-3,
    "reg": 1e-4,
    "batch_size": 128,
    "epoch": 30,
    "use_cuda": True,
    "metrics": [5, 10, 20]
}
if config["use_cuda"]:
    device = torch.device("cuda")
    config["device"] = device
else:
    device = torch.device("cpu")
    config["device"] = device

configs = generate_config(dropouts=[0.3, 0.5, 0.7], heads=[1], lrs=[1e-3], regs=[1e-4],
                          ks=[10], session_lengths=[50], embedding_dims=[200, 300],
                          config=config)

training_dataset = SessionDataSet(sessions=train_sessions, neighbors=train_neighbors,
                                  neighbor_weights=train_neighbor_weights, labels=training_labels,
                                  session_length=config["session_length"], k=config["k"])
testing_dataset = SessionDataSet(sessions=test_sessions, neighbors=test_neighbors,
                                 neighbor_weights=test_neighbor_weights, labels=testing_labels,
                                 session_length=config["session_length"], k=config["k"])

training_dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=config["batch_size"], shuffle=True)

for config in configs:
    model = CoSAN(config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["reg"], amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()
    print(config)

    best_hr = 0
    best_mrr = 0
    best_ndcg = 0
    total_loss = []
    for epoch in range(config["epoch"]):
        print("epoch:(%d/%d)" % (epoch + 1, config["epoch"]))
        print("Start Training:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # train
        model.train()
        cur_loss = 0
        for i, data in enumerate(training_dataloader):
            session_ids, sessions, neighbors, neighbor_weights, labels = data
            neighbors = neighbors[:, :, :config["k"]]
            neighbor_weights = neighbor_weights[:, :, :config["k"]]
            optimizer.zero_grad()

            # forward & backward
            outputs = model(session_ids.to(device), sessions.to(device), neighbors.to(device),
                            neighbor_weights.to(device), is_train=True)

            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            cur_loss += loss.item() * sessions.shape[0]
            if i % 1000 == 0 and i != 0:
                print('(%d/%d) loss: %f' % (i, len(training_dataloader), loss.item()))

        print(f"total loss: {cur_loss / len(training_dataloader.dataset)}")
        total_loss += [cur_loss]

        # test
        model.eval()
        test_loss = 0
        print("Start testing:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        with torch.no_grad():
            hrs = [0 for _ in range(len(config["metrics"]))]
            mrrs = [0 for _ in range(len(config["metrics"]))]
            ndcgs = [0 for _ in range(len(config["metrics"]))]
            for data in testing_dataloader:
                session_ids, sessions, neighbors, neighbor_weights, labels = data
                session_ids = session_ids + config["training_session_num"]
                neighbors = neighbors[:, :, :config["k"]]
                neighbor_weights = neighbor_weights[:, :, :config["k"]]

                # forward & backward
                outputs = model(session_ids.to(device), sessions.to(device), neighbors.to(device),
                                neighbor_weights.to(device), is_train=False)
                loss = criterion(outputs, labels.to(device))

                test_loss += loss.item() * sessions.shape[0]

                # metric
                result = torch.topk(outputs, k=config["metrics"][-1], dim=1)[1]
                for i, k in enumerate(config["metrics"]):
                    hrs[i] += cal_hr(result[:, :k].cpu().numpy(), labels.cpu().numpy())
                    mrrs[i] += cal_mrr(result[:, :k].cpu().numpy(), labels.cpu().numpy())
                    ndcgs[i] += cal_ndcg(result[:, :k].cpu().numpy(), labels.cpu().numpy())

            test_loss = test_loss / len(testing_dataloader.dataset)

            for i, k in enumerate(config["metrics"]):
                hrs[i] = hrs[i] / len(testing_dataloader.dataset)
                mrrs[i] = mrrs[i] / len(testing_dataloader.dataset)
                ndcgs[i] = ndcgs[i] / len(testing_dataloader.dataset)
                print(f'HR@{k}: {hrs[i]:.4f} MRR@{k}: {mrrs[i]:.4f} NDCG@{k}: {ndcgs[i]:.4f}')

            if hrs[-1] > best_hr:
                best_hr = hrs[-1]
                best_mrr = mrrs[-1]
                best_ndcg = ndcgs[-1]

                for i, k in enumerate(config["metrics"]):
                    print(f'best ever HR@{k}: {hrs[i]:.4f} MRR@{k}: {mrrs[i]:.4f} NDCG@{k}: {ndcgs[i]:.4f}')
            print('================================')
