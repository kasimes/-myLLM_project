import torch
import torch.nn as nn

from kasim_RMSnorm import KasimRMSNorm
from kasim_decoder_block import KasimDecoderBlock

def get_rotary_positional_encoding(input: torch.Tensor, Base=10000):
    # input: (batch_size, seq_len, dim)
    batch_size, seq_len, dim = input.shape
    assert dim % 2 == 0, "dim must be even"

    half_dim = dim // 2
    freq_index = torch.arange(0, half_dim, device=input.device, dtype=torch.float32)
    freq = 1.0 / (Base ** (freq_index / dim))
    position = torch.arange(0, seq_len, device=input.device, dtype=torch.float32).unsqueeze(1)

    angles = position * freq
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    input_even = input[:, :, :half_dim]
    input_odd = input[:, :, half_dim:]

    input_even_rotated = input_even * cos_angles - input_odd * sin_angles
    input_odd_rotated = input_even * sin_angles + input_odd * cos_angles

    input_rotated = torch.empty_like(input)
    input_rotated[:, :, :half_dim] = input_even_rotated
    input_rotated[:, :, half_dim:] = input_odd_rotated

    return input_rotated


class Kasim_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.get_pos = get_rotary_positional_encoding
    
        self.layers= nn.Sequential(*[
            KasimDecoderBlock(embed_dim,num_heads, context_length)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim,vocab_size)
       
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0) 
        x= self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.get_pos(x)
        x= self.layers(x)
        x= self.lm_head(x)
            
        return x
