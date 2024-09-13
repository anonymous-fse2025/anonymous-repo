import torch.nn as nn
import torch.nn.functional as F
import torch
from Transfomer import TransformerBlock
from rightTransfomer import rightTransformerBlock
from Embedding import Embedding
from Multihead_Attention import MultiHeadedAttention
from postionEmbedding import PositionalEmbedding
from LayerNorm import LayerNorm
from SubLayerConnection import *
from DenseLayer import DenseLayer
import numpy as np


class GGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_steps, dropout=0.1):
        super(GGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        self.input_transform = nn.Linear(input_dim, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reset_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden_transform = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, adj_matrix):
        hidden_states = self.input_transform(node_features)

        for _ in range(self.num_steps):
            messages = torch.bmm(adj_matrix, hidden_states)

            combined = torch.cat([hidden_states, messages], dim=-1)
            update_gate = torch.sigmoid(self.update_gate(combined))
            reset_gate = torch.sigmoid(self.reset_gate(combined))

            reset_hidden = reset_gate * hidden_states
            combined_hidden = torch.cat([reset_hidden, messages], dim=-1)
            hidden_states = update_gate * hidden_states + (1 - update_gate) * torch.tanh(self.hidden_transform(combined_hidden))

            hidden_states = self.dropout(hidden_states)

        return hidden_states


class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.feed_forward_hidden = 4 * self.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen

        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.token_embedding = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 1)
        self.token_embedding1 = nn.Embedding(args.Nl_Vocsize, self.embedding_size)
        self.text_embedding = nn.Embedding(20, self.embedding_size)
        self.conv1 = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.conv2 = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))

        self.ggnn_blocks = nn.ModuleList([
            GGNN(self.embedding_size, self.embedding_size, num_steps=3, dropout=0.1) for _ in range(5)
        ])

        self.positional_embedding = nn.Parameter(torch.zeros(1, args.NlLen, self.embedding_size))
        self.norm = LayerNorm(self.embedding_size)

        self.lstm = nn.LSTM(
            self.embedding_size // 2,
            self.embedding_size // 4,
            batch_first=True,
            bidirectional=True
        )
        self.res_linear = nn.Linear(self.embedding_size, 2)
        self.res_linear2 = nn.Linear(self.embedding_size, 1)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_node, inputtype, inputad, res, inputtext, linenode, linetype, linemus):
        nlmask = (input_node > 0)
        resmask = (input_node == 2)

        nodeem = self.token_embedding(input_node)
        nodeem = torch.cat([nodeem, inputtext.unsqueeze(-1).float()], dim=-1)
        lineem = self.token_embedding1(linenode)

        x = torch.cat([nodeem, lineem], dim=1)

        for ggnn in self.ggnn_blocks:
            x = ggnn(x, inputad.float())

        x = x[:, :input_node.size(1)]

        res_softmax = F.softmax(
            self.res_linear2(x).squeeze(-1).masked_fill(~resmask, -1e9),
            dim=-1
        )

        loss = -torch.log(res_softmax.clamp(min=1e-10, max=1)) * res
        loss = loss.sum(dim=-1)

        return loss, res_softmax, x
