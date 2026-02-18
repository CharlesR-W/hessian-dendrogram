"""1-layer transformer for modular addition (grokking task)."""

import torch
import torch.nn as nn


class ModAddTransformer(nn.Module):
    """Minimal transformer for a + b mod p.

    Input: [a, b, =] where a,b in {0..p-1} and = is token index p.
    Output: logits over {0..p-1} from the = position.

    Architecture: token embed + pos embed -> 1 self-attention layer -> unembed.
    No layer norm, no MLP, no bias. Designed for full Hessian computation.
    """

    def __init__(self, p: int = 113, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        self.token_embed = nn.Embedding(p + 1, d_model)  # p numbers + = token
        self.pos_embed = nn.Embedding(3, d_model)         # 3 positions

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape
        tok_emb = self.token_embed(tokens)
        pos = torch.arange(S, device=tokens.device)
        x = tok_emb + self.pos_embed(pos)

        # Multi-head self-attention
        Q = self.W_Q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = self.W_O(
            torch.matmul(attn, V)
            .transpose(1, 2)
            .contiguous()
            .view(B, S, self.d_model)
        )

        x = x + out  # residual
        return self.unembed(x[:, -1, :])  # read from = position
