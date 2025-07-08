from torch import nn
import torch
import torch.nn.functional as F
class MLPAligner(torch.nn.Module):
    """对齐网络：将modelA激活映射到modelB空间"""
    def __init__(self, cfg):
        super().__init__()
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(input_dim, hidden_dim),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(hidden_dim, output_dim)
        self.linear = nn.Linear(cfg["A_in"], cfg["B_in"])
        self.dropout = nn.Dropout(p=0.5)  # p 是丢弃概率，默认是 0.5
    def forward(self, x):
        
        x = self.linear(x)
        x = F.dropout(x, p=0.5, training=self.training)  
        return x

