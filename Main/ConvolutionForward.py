import torch.nn as nn
from gelu import GELU


class ConvolutionLayer(nn.Module):
    def __init__(self, dmodel, layernum, kernelsize=3, dropout=0.1, activation=None):
        super(ConvolutionLayer, self).__init__()

        self.activation = activation if activation else GELU()
        self.dropout = nn.Dropout(dropout)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize - 1) // 2),
            self.activation,
            self.dropout,
            nn.Conv1d(layernum, dmodel, kernelsize, padding=(kernelsize - 1) // 2)  # 修正输出通道数为 dmodel
        )

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).expand(-1, -1, x.size(2))
        x = x.masked_fill(mask == 0, 0)

        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        out = x.permute(0, 2, 1)

        return out
