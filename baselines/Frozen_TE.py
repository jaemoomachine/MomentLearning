import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class FrozenFFNEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int = None,
        dropout: float = 0.1,
        ff_init: str = "normal",
        init_params: dict = None,
    ):
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # build FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self._init_and_freeze_ffn(ff_init, init_params or {})

    def _init_and_freeze_ffn(self, mode: str, params: dict):
        """
        Initialize the two Linear layers in the FFN from the specified distribution,
        then freeze their weight and bias.
        params for 'normal': {'mean':0., 'std':1.}
        params for 'uniform': {'a':-1., 'b':1.}
        """
        # extract linear layers
        linear_layers = [m for m in self.ffn if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2, f"Expected 2 Linear layers, found {len(linear_layers)}"
        fc1, fc2 = linear_layers

        if mode == "normal":
            mean = params.get("mean", 0.0)
            std = params.get("std", 1.0)
            nn.init.normal_(fc1.weight, mean=mean, std=std)
            nn.init.normal_(fc1.bias, mean=mean, std=std)
            nn.init.normal_(fc2.weight, mean=mean, std=std)
            nn.init.normal_(fc2.bias, mean=mean, std=std)
        elif mode == "uniform":
            a = params.get("a", -math.sqrt(1 / fc1.in_features))
            b = params.get("b", math.sqrt(1 / fc1.in_features))
            nn.init.uniform_(fc1.weight, a=a, b=b)
            nn.init.uniform_(fc1.bias, a=a, b=b)
            nn.init.uniform_(fc2.weight, a=a, b=b)
            nn.init.uniform_(fc2.bias, a=a, b=b)
        else:
            raise ValueError(f"Unknown init mode: {mode}")

        # freeze parameters
        for p in fc1.parameters():
            p.requires_grad = False
        for p in fc2.parameters():
            p.requires_grad = False

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # self-attention + residual
        attn_out, _ = self.attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.drop(attn_out)
        src = self.norm1(src)

        # frozen FFN + residual
        inter = self.ffn(src)
        src = src + self.drop(inter)
        return self.norm2(src)


class FrozenTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = DataEmbedding(
            c_in=config.c_in, 
            d_model=config.d_model, 
            dropout=config.dropout
        )
        
        layer = FrozenFFNEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            d_ff=config.d_ff,
            dropout=config.dropout,
            ff_init=config.ff_init
        )
        self.enc = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.d_model, config.c_out)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x: (batch, seq_len, c_in)
        x = self.input_proj(x).transpose(0, 1)  # (seq_len, batch, d_model)

        for layer in self.enc:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return self.head(x[0])
