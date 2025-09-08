import torch
import torch.nn as nn

class KasimRMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))  # gamma

    def forward(self, x):
        # RMS: karelerin ortalamasını al
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized