import torch
import torch.nn as nn
from kasim_multi_head_attention import Kasim_Multi_Head_Attention
from kasim_RMSnorm import KasimRMSNorm
from kasim_MLP import KasimMLP

class KasimDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, context_length):
        super().__init__()
        self.self_attention = Kasim_Multi_Head_Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            output_dim=embed_dim,  # Çıktı boyutu embed_dim olmalı
            context_length=context_length,
            dropout_rate=0.5
        )
        self.norm1 = KasimRMSNorm(embed_dim)
        self.mlp = KasimMLP(embed_dim, hidden_dim=4*embed_dim)  # hidden_dim genelde 4x
        self.norm2 = KasimRMSNorm(embed_dim)

    def forward(self, x):
        # Self-attention bloğu
        residual = x
        x = self.norm1(x)  # Pre-normalization
        x = self.self_attention(x)
        x = residual + x   # Residual connection

        # MLP bloğu
        residual = x
        x = self.norm2(x)  # Pre-normalization
        x = self.mlp(x)
        x = residual + x   # Residual connection

        return x
