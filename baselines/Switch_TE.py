import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from embed import PositionalEmbedding, TokenEmbedding, DataEmbedding

""" Mixture-of-Experts (MoE)-based Transformer encoder """

class MoEFFN(nn.Module):
    def __init__(
        self,
        d_model: int, 
        dim_ff: int, 
        n_experts: int=4, 
        topk: int=2, 
        sparse_gating: bool=True
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Linear(dim_ff, d_model)
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts)
        self.topk = topk
        self.sparse_gating = sparse_gating

    def forward(self, x) -> torch.Tensor:
        # x: (N, d_model)
        w_logits = self.gate(x)             # (N, E)
        w = F.softmax(w_logits, dim=-1)     # (N, E)

        if self.sparse_gating:
            idx = torch.topk(w, self.topk, dim=-1).indices        # (N, k)
            mask = torch.zeros_like(w).scatter(1, idx, 1.0)    # (N, E)
            w = w * mask
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)

        out = x.new_zeros(x.shape)
        for i, expert in enumerate(self.experts):
            out += expert(x) * w[:, i].unsqueeze(-1)  # weighted sum

        return out
    
class MoEEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, nhead: int, dim_ff: int, 
        n_experts :int=4, topk: int=2, sparse_gating: bool=True, dropout: float=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.ffn = MoEFFN(d_model, dim_ff, n_experts, topk, sparse_gating=sparse_gating)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        """
        src: (L, B, D)
        """
        a, _ = self.attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src2 = self.norm1(src + self.drop(a))

        b, seq = src2.shape[1], src2.shape[0]
        flat = src2.transpose(0, 1).reshape(b * seq, -1)  # (B*L, D)
        y = self.ffn(flat).reshape(b, seq, -1).transpose(0, 1)  # (L, B, D)

        return self.norm2(src2 + self.drop(y))


class SwitchTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = DataEmbedding(
            c_in=config.c_in, 
            d_model=config.d_model, 
            dropout=config.dropout
        )
        encoder_layer = MoEEncoderLayer(
            d_model=config.d_model, 
            nhead=config.nhead, dim_ff=config.d_ff, n_experts = config.num_experts,
            topk=config.topk, sparse_gating=config.sparse_gating,
            dropout=config.dropout
        )
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.d_model, config.c_out)

    def forward(self, x):
        x = self.input_proj(x)     # (B, L, D)
        x = x.transpose(0, 1)      # (L, B, D)
        for layer in self.encoder:
            x = layer(x)          
        last_token = x[-1]         # (B, D)
        return self.head(last_token) 
    


