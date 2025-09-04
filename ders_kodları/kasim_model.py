import torch
import torch.nn as nn
from kasim_causal_attention import KasimCausalAttention

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
    def __init__(self, vocab_size, embed_dim, context_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.get_pos = get_rotary_positional_encoding
    
        self.self_attention = KasimCausalAttention(embed_dim, embed_dim,context_length, dropou_rate=0.5)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        embeddings = self.get_pos(embeddings)
        embeddings = self.self_attention(embeddings)
        return embeddings
