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

class GGANN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GGANN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRUCell(out_features, out_features)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        N = Wh.size()[0]
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        h_updated = self.gru(h_prime, h)
        return h_updated

class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.token_embedding = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 1)
        self.token_embedding1 = nn.Embedding(args.Nl_Vocsize, self.embedding_size - 2)
        self.ggann_layer1 = GGANN(self.embedding_size, self.embedding_size)
        self.ggann_layer2 = GGANN(self.embedding_size, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)]
        )
        self.resLinear = nn.Linear(self.embedding_size, 2)
        self.resLinear2 = nn.Linear(self.embedding_size, 1)
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_node, inputtype, inputad, res, inputtext, linenode, linetype, linemus, modification, churn):
        nlmask = torch.gt(input_node, 0)
        resmask = torch.eq(input_node, 2)
        inputad = inputad.float()
        batch_size, seq_len = input_node.size()
        nodeem = self.token_embedding(input_node)
        nodeem = torch.cat([nodeem, inputtext.unsqueeze(-1).float()], dim=-1)
        lineem = self.token_embedding1(linenode)
        lineem = torch.cat([lineem, modification.unsqueeze(-1).float(), churn.unsqueeze(-1).float()], dim=-1)
        x = torch.cat([nodeem, lineem], dim=1)
        x_flat = x.view(-1, x.size(-1))
        adj = inputad.repeat(batch_size, 1)
        x_ggann = self.ggann_layer1(x_flat, adj)
        x_ggann = F.elu(x_ggann)
        x_ggann = self.ggann_layer2(x_ggann, adj)
        x_ggann = x_ggann.view(batch_size, seq_len, -1)
        for trans in self.transformerBlocks:
            x_ggann = trans.forward(x_ggann, nlmask, inputad)
        x = x_ggann[:, :input_node.size(1)]
        resSoftmax = F.softmax(self.resLinear2(x).squeeze(-1).masked_fill(resmask == 0, -1e9), dim=-1)
        loss = -torch.log(resSoftmax.clamp(min=1e-10, max=1)) * res
        loss = loss.sum(dim=-1)
        return loss, resSoftmax, x
