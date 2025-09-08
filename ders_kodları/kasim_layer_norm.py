import torch
import torch.nn as nn

class KasimLayerNorm(nn.Module):

    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)  
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight * x_normalized
  