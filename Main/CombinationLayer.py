import torch.nn as nn
import torch
import math


class CombinationLayer(nn.Module):
    def forward(self, query, key, value, dropout=None):
        d_k = query.size(-1)  # 缓存维度大小
        scale = 1 / math.sqrt(d_k)  # 计算缩放因子

        # 计算缩放后的点积
        query_key = query * key * scale
        query_value = query * value * scale

        # 使用广播机制计算加权和
        weights = torch.softmax(torch.stack([query_key, query_value], dim=-1), dim=-1)
        combined = weights[..., 0] * key + weights[..., 1] * value

        if dropout:
            combined = dropout(combined)

        return combined
