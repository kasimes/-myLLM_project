import torch
import torch.nn as nn

class KasimCausalAttention(nn.Module):
    def __init__(self, embed_dim, output_dim, context_length, dropou_rate=0):
        super().__init__()
        self.embed_dim = embed_dim

        self.q_weight = nn.Linear(embed_dim, output_dim, bias=False)
        self.k_weight = nn.Linear(embed_dim, output_dim, bias=False)
        self.v_weight = nn.Linear(embed_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropou_rate)

        # Causal mask
        self.register_buffer("causal_mask", torch.tril(torch.ones(context_length, context_length)))
        self.context_length = context_length

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.q_weight(x)
        K = self.k_weight(x)
        V = self.v_weight(x)

        # Dot-product attention
        attn_scres = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)

        # Mask
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        attn_scres = attn_scres.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_scres = attn_scres.softmax(dim=-1)
        attn_scres = self.dropout(attn_scres)

        return torch.matmul(attn_scres, V)
