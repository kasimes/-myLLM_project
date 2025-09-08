import torch 
from torch import nn
from kasim_causal_attention import KasimCausalAttention

class Kasim_Multi_Head_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim, context_length, dropout_rate=0):
        super().__init__()
        
        self.context_length = context_length
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate
        )
        self.projection = nn.Linear(embed_dim, output_dim)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        
        # (batch, seq, embed) -> (seq, batch, embed)
        x = x.transpose(0, 1)
        
        # Attention mask için doğru boyut
        attention_mask = self.mask[:seq_len, :seq_len]
        
        # MultiheadAttention forward
        out, _ = self.multi_head_attention(x, x, x, attn_mask=attention_mask)
        
        # (seq, batch, embed) -> (batch, seq, embed)
        out = out.transpose(0, 1)
        
        # Final projection
        out = self.projection(out)
        
        return out