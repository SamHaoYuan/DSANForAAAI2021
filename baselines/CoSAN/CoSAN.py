# -*- coding: UTF-8 -*-
import torch
import math
import numpy as np


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, hidden_dim, head, dropout=0):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_dim % head == 0
        self.d_k = hidden_dim // head
        self.hidden_dim = hidden_dim
        self.head = head
        self.q_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout)

    def attention(self, q, k, v, mask=None):
        hidden_dim = q.shape[-1]
        scores = q @ k.transpose(-2, -1) / math.sqrt(hidden_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        scores = self.dropout(torch.softmax(scores, dim=-1))
        return scores @ v

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        session_length = q.shape[1]
        if mask is not None:
            # (b, len) -> (b, head, len, len)
            mask = mask.unsqueeze(1).expand(batch_size, self.head, session_length)
            mask = mask.unsqueeze(2).expand(batch_size, self.head, session_length, session_length)
        q = self.dropout(torch.relu(self.q_linear(q).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)))
        k = self.k_linear(k).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        output = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return output


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_linear1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.ffn_linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.ffn_linear2(self.dropout(torch.relu(self.ffn_linear1(x))))


class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim, head, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.head = head
        self.dropout = torch.nn.Dropout(p=dropout)
        self.self_attention = MultiHeadSelfAttention(hidden_dim=self.hidden_dim, head=self.head, dropout=dropout)
        self.feed_forward_network = FeedForwardNetwork(hidden_dim=self.hidden_dim, dropout=dropout)
        self.layer_norm = torch.nn.LayerNorm(self.hidden_dim)

    def forward(self, x, mask):
        x = self.layer_norm(x)
        self_attention_output = x + self.dropout(self.self_attention(q=x, k=x, v=x, mask=mask))
        output = self_attention_output + self.dropout(self.feed_forward_network(self.layer_norm(self_attention_output)))
        return output


class CoSAN(torch.nn.Module):
    def __init__(self, config):
        super(CoSAN, self).__init__()
        self.session_num = config["session_num"]  # training session + testing session
        self.item_num = config["item_num"]
        self.embedding_dim = config["embedding_dim"]
        self.alpha = config["alpha"]
        self.head = config["head"]
        self.self_attention_layers = config["self_attention_layers"]
        self.dropout = config["dropout"]

        self.session_embedding = torch.nn.Embedding(self.session_num, embedding_dim=self.embedding_dim, padding_idx=0)
        self.item_embedding = torch.nn.Embedding(self.item_num, embedding_dim=self.embedding_dim, padding_idx=0)
        self.encoder = torch.nn.ModuleList(
            [Encoder(hidden_dim=self.embedding_dim, head=self.head, dropout=self.dropout) for _ in
             range(self.self_attention_layers)])
        self.predict_linear = torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=True)

    def forward(self, session_id, session, neighbor_id, neighbor_weight, is_train=True):
        """
        :param is_train: train or test
        :param session_id: (b, )
        :param session: (b, len)
        :param neighbor_id: (b, len, k)
        :param neighbor_weight: (b, len, k)
        """
        # session_embedding = self.session_embedding(session_id)  # (b, dim)
        item_embedding = self.item_embedding(session)  # (b, len, dim)
        neighbor_session_embedding = self.session_embedding(neighbor_id)  # (b, len, k, dim)
        f = neighbor_weight.unsqueeze(2) @ neighbor_session_embedding  # (b, len, 1, dim)
        f = f.squeeze()  # (b, len, dim)
        collaborative_item_embedding = item_embedding + f * self.alpha  # (b, len, dim)
        mask = (session != 0).float()  # (b, len)
        for layer in self.encoder:
            collaborative_item_embedding = layer(collaborative_item_embedding, mask)  # (b, len, dim)

        final_session_embedding = collaborative_item_embedding[:, -1, :].squeeze()
        scores = final_session_embedding @ self.item_embedding.weight.T

        return scores