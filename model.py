import math
import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads, context_length):
        super().__init__()
        assert width % heads == 0

        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

        mask = torch.tril(
            torch.ones(context_length, context_length, dtype=torch.bool)
        )
        self.register_buffer("mask", mask.view(1, 1, context_length, context_length))

    def forward(self, x):
        batch, length, width = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_width).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        scores = scores.masked_fill(
            ~self.mask[:, :, :length, :length],
            -1e9,
        )
        weights = F.softmax(scores, dim=-1)

        result = weights @ v
        result = result.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(result)

class TransformerBlock(nn.Module):
    def __init__(self, width, heads, context_length):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, context_length)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class SmallGPT(nn.Module):
    def __init__(self, vocab_size, context_length, width=192, heads=4, layers=4):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context_length, width)
        self.blocks = nn.ModuleList(
            TransformerBlock(width, heads, context_length)
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, token_ids):
        batch, length = token_ids.shape
        assert length <= self.context_length

        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        return self.output(self.final_norm(x))
