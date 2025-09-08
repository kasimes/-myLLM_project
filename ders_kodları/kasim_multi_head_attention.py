import torch 
from torch import nn
from kasim_causal_attention import KasimCausalAttention

class Kasim_Multi_Head_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads,output_dim,context_length,dropout_rate=0 ):
        super().__init__()

        self.context_length=context_length
        self.multi_head_attention=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=dropout_rate)
        self.projection=nn.Linear(embed_dim,output_dim)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        number_of_tokens=x.shape[0]
        x=x[self.context_legenth]
        attention_mask=self.mask[:number_of_tokens,:number_of_tokens]
        out, _=self.multi_head_attention(x,x,x,attn_mask=attention_mask)
        out=self.projection(out)
        return out  