# -*- coding: UTF-8 -*-
import torch
import math
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from random import sample


def cal_similarity(s, m, t):
    """
    calculate similarity between current session and neighbourhood session
    :param s: current session
    :param m: candidate neighbourhood session
    :param t: timestamp
    :return: similarity
    """
    s = set(s[:t + 1])
    m = set(m)
    return len(s & m) / math.sqrt(len(s) * len(m))


def find_neighbors(sessions, candidate_neighbor_sessions, k=20, M=-1):
    # cache, only the original sessions
    item2session = {}  # {item:[session_ids]}
    session_length = 3
    for i, neighbor_session in enumerate(candidate_neighbor_sessions):
        if session_length == 3:
            for item in neighbor_session:
                if item in item2session:
                    item2session[item].add(i)
                else:
                    item2session[item] = set()
                    item2session[item].add(i)
        session_length = len(neighbor_session)
    # find neighbor
    neighbors_all = []
    neighbor_weights_all = []
    for i, session in enumerate(sessions):
        if i % 1000 == 0:
            print(f"{i}/{len(sessions)}")
        neighbor_session = []
        neighbor_session_weights = []
        for j, item in enumerate(session):
            neighbors = []
            neighbor_weights = []
            matched_session_indices = list(item2session[item])
            if M != -1 and len(matched_session_indices) > M:
                matched_session_indices = sample(matched_session_indices, M)
            if len(matched_session_indices) > k:
                for matched_session_index in matched_session_indices:
                    similarity = cal_similarity(session,
                                                candidate_neighbor_sessions[matched_session_index], j)
                    neighbors += [matched_session_index + 1]  # +1 for embedding layer padding
                    neighbor_weights += [similarity]
            else:
                neighbors = [0 for _ in range(k)]
                neighbor_weights = [0.0 for _ in range(k)]
                for idx, matched_session_index in enumerate(matched_session_indices):
                    similarity = cal_similarity(session,
                                                candidate_neighbor_sessions[matched_session_index], j)
                    neighbors[idx] = matched_session_index + 1  # +1 for embedding layer padding
                    neighbor_weights[idx] = similarity
            # sort
            neighbors = np.array(neighbors)
            neighbor_weights = np.array(neighbor_weights)
            mask = np.argsort(neighbor_weights)[::-1][:k]
            neighbors = list(neighbors[mask])
            neighbor_weights = list(neighbor_weights[mask])

            neighbor_session += [neighbors]
            neighbor_session_weights += [neighbor_weights]
        neighbors_all += [neighbor_session]
        neighbor_weights_all += [neighbor_session_weights]
    return neighbors_all, neighbor_weights_all


class SessionDataSet(Dataset):
    def __init__(self, sessions, neighbors, neighbor_weights, labels, session_length=20, k=20, padding_idx=0):
        self.sessions = sessions
        self.neighbors = neighbors
        self.neighbor_weights = neighbor_weights
        self.labels = labels
        # padding
        self.padding(session_length=session_length, k=k, padding_idx=padding_idx)
        # change to tensor
        self.sessions = torch.tensor(self.sessions, dtype=torch.long)
        self.neighbors = torch.tensor(self.neighbors, dtype=torch.long)
        self.neighbor_weights = torch.tensor(self.neighbor_weights, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __getitem__(self, idx):
        # idx+1 for embedding layer padding
        return idx + 1, self.sessions[idx], self.neighbors[idx], self.neighbor_weights[idx], self.labels[idx]

    def __len__(self):
        return self.sessions.shape[0]

    def padding(self, session_length=20, k=20, padding_idx=0):
        """
        padding
        """
        for i in range(len(self.sessions)):
            # padding session
            length = len(self.sessions[i])
            if length >= session_length:
                # get last session_length items
                self.sessions[i] = self.sessions[i][-session_length:]
                self.neighbors[i] = self.neighbors[i][-session_length:]
                self.neighbor_weights[i] = self.neighbor_weights[i][-session_length:]
            else:
                self.sessions[i] = (session_length - length) * [padding_idx] + self.sessions[i]
                self.neighbors[i] = (session_length - length) * [[0] * k] + self.neighbors[i]
                self.neighbor_weights[i] = (session_length - length) * [[0] * k] + self.neighbor_weights[i]
            # padding neighbor, guaranteeing that the number of neighbors is greater than k
            for j in range(len(self.neighbors[i])):
                self.neighbors[i][j] = self.neighbors[i][j][:k]
                self.neighbor_weights[i][j] = self.neighbor_weights[i][j][:k]


if __name__ == '__main__':
    dataset = "retailrocket"  # diginetica
    M = -1  
    k = 15

    training_data = pickle.load(open("../../data/" + dataset + "/train.txt", "rb"))
    training_sessions = training_data[1]
    training_labels = training_data[2]
    testing_data = pickle.load(open("../../data/" + dataset + "/test.txt", "rb"))
    testing_sessions = testing_data[1]
    testing_labels = testing_data[2]
    max_session_length = 0
    training_seqs = []
    for i in range(len(training_sessions)):
        seq = training_sessions[i] + [training_labels[i]]
        training_seqs += [seq]
        if len(seq) > max_session_length:
            max_session_length = len(seq)
    for session in testing_sessions:
        if len(session) > max_session_length:
            max_session_length = len(session)
    print(f"max session length: {max_session_length}")
    item_set = set()
    for session in training_sessions:
        for item in session:
            item_set.add(item)
    for item in training_labels:
        item_set.add(item)
    filtered_testing_sessions = []
    for session in testing_sessions:
        filtered_session = []
        for item in session:
            if item in item_set:
                filtered_session += [item]
            else:
                print("!")
        filtered_testing_sessions += [filtered_session]
    testing_sessions = filtered_testing_sessions
    training_neighbors, training_neighbor_weights = find_neighbors(sessions=training_sessions,
                                                                   candidate_neighbor_sessions=training_seqs, k=k, M=M)
    pickle.dump(training_sessions, open(f"./{dataset}/train_sessions.txt", "wb"), protocol=4)
    pickle.dump(training_neighbors, open(f"./{dataset}/train_neighbors.txt", "wb"), protocol=4)
    pickle.dump(training_neighbor_weights, open(f"./{dataset}/train_neighbor_weights.txt", "wb"), protocol=4)

    testing_neighbors, testing_neighbor_weights = find_neighbors(sessions=testing_sessions,
                                                                 candidate_neighbor_sessions=training_seqs, k=k, M=M)
    pickle.dump(testing_sessions, open(f"./{dataset}/test_sessions.txt", "wb"), protocol=4)
    pickle.dump(testing_neighbors, open(f"./{dataset}/test_neighbors.txt", "wb"), protocol=4)
    pickle.dump(testing_neighbor_weights, open(f"./{dataset}/test_neighbor_weights.txt", "wb"), protocol=4)
