# type: ignore

import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, mask=False):
        super().__init__()
        self.input_dim = input_dim # D
        self.W_q = nn.Linear(input_dim, input_dim) # D x D 
        self.W_k = nn.Linear(input_dim, input_dim) # D x D
        self.W_v = nn.Linear(input_dim, input_dim) # D x D
        self.scale = input_dim ** 0.5
        self.mask = mask

    def forward(self, X):  # (N, S, D)
        N, S, D = X.shape

        Q = self.W_q(X) # (N, S, D)
        K = self.W_k(X) # (N, S, D)
        V = self.W_v(X) # (N, S, D)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (N, S, S)

        if self.mask:
            attn_mask = torch.triu(torch.ones(S, S, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(attn_mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)  # (N, S, D)
    
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.ff(x))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, mask=False):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.mask = mask

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, X, attn_mask=None):  # attn_mask: (N, 1, S, S)
        N, S, D = X.shape
        H = self.num_heads
        d = self.head_dim

        Q = self.W_q(X).view(N, S, H, d).transpose(1, 2)
        K = self.W_k(X).view(N, S, H, d).transpose(1, 2)
        V = self.W_v(X).view(N, S, H, d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (N, H, S, S)

        if self.mask:
            causal_mask = torch.triu(torch.ones(S, S, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(N, S, D)

        return self.out_proj(out)
    
class CrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads

        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, X, context, context_mask=None):  # context_mask: (N, 1, 1, S)
        N, T, _ = X.shape
        S = context.size(1)

        Q = self.W_q(X).view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(context).view(N, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(context).view(N, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (N, H, T, S)

        if context_mask is not None:
            scores = scores.masked_fill(context_mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(N, T, self.model_dim)
        return self.out_proj(out)

