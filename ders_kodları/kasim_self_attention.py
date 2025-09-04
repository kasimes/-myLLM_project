import torch
import torch.nn as nn

class KasimSelfAttention(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.q_weight = nn.Linear(embed_dim, output_dim, bias=False)
        self.k_weight = nn.Linear(embed_dim, output_dim, bias=False)
        self.v_weight = nn.Linear(embed_dim, output_dim, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        Q = self.q_weight(x)  # (batch_size, seq_len, output_dim)
        K = self.k_weight(x)  # (batch_size, seq_len, output_dim)
        V = self.v_weight(x)  # (batch_size, seq_len, output_dim)

        # Dot-product attention (3D tensor için)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores / (Q.shape[-1] ** 0.5), dim=-1)

        output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, output_dim)
        return output
