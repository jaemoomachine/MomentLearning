import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from embed import PositionalEmbedding, TokenEmbedding, DataEmbedding

""" Guo et al. (2025) Less is more: Embracing sparsity and interpolation with Esiformer for time series forecasting """

class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        sparsity=0.9,
        connectivity=None,
        small_world=False,
        dynamic=False,
        deltaT=6000,
        Tend=150000,
        alpha=0.1,
        max_size=int(1e8),
    ):
        super().__init__()
        assert sparsity < 1.0
        self.in_features, self.out_features = in_features, out_features
        self.dynamic, self.max_size = dynamic, max_size
        if connectivity is None and not small_world:
            nnz = round((1.0 - sparsity) * in_features * out_features)
            if in_features * out_features <= 1e8:
                idx = np.random.choice(in_features * out_features, nnz, replace=False)
                row = torch.from_numpy(idx // in_features)
                col = torch.from_numpy(idx % in_features)
            else:
                row = torch.randint(0, out_features, (nnz,))
                col = torch.randint(0, in_features, (nnz,))
            indices = torch.stack((row, col), dim=0)
        elif small_world:
            nnz = round((1.0 - sparsity) * in_features * out_features)
            inputs = torch.arange(in_features, dtype=torch.float)
            outputs = torch.arange(out_features, dtype=torch.float)
            indices = small_world_chunker(inputs, outputs, nnz).to_sparse().indices()
        else:
            indices = connectivity
        indices = indices.long()
        values = torch.empty(indices.size(1))
        self.register_buffer('indices', indices)
        self.weights = nn.Parameter(values)
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.sparsity = sparsity
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        flat = inputs.flatten(end_dim=-2)
        weight_mat = torch.sparse.FloatTensor(self.indices, self.weights, (self.out_features, self.in_features)).to_dense()
        out = torch.mm(weight_mat, flat.t()).t()
        if self.bias is not None:
            out = out + self.bias
        return out.view(*inputs.shape[:-1], self.out_features)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, sparsity={self.sparsity}'


class SparseFFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        dim_ff: int = None,
        sparsity: float = 0.01,
        activation=nn.GELU,
        dropout: float = 0.1,
        dynamic: bool = False,
        deltaT: int = 6000,
        Tend: int = 150000,
        alpha: float = 0.1,
        max_size: int = int(1e8),
    ):
        super().__init__()
        dim_ff = dim_ff or (4 * d_model)
        self.fc1 = SparseLinear(
            in_features=d_model,
            out_features=dim_ff,
            bias=True,
            sparsity=sparsity,
            small_world=False,
            dynamic=dynamic,
            deltaT=deltaT,
            Tend=Tend,
            alpha=alpha,
            max_size=max_size,
        )
        self.act = activation()
        self.drop = nn.Dropout(dropout)
        self.fc2 = SparseLinear(
            in_features=dim_ff,
            out_features=d_model,
            bias=True,
            sparsity=sparsity,
            small_world=False,
            dynamic=dynamic,
            deltaT=deltaT,
            Tend=Tend,
            alpha=alpha,
            max_size=max_size,
        )

    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return self.drop(y)
    

class SparseEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int = None,
        sparsity: float = 0.01,
        dropout: float = 0.1,
        dynamic: bool = False,
        deltaT: int = 6000,
        Tend: int = 150000,
        alpha: float = 0.1,
        max_size: int = int(1e8),
    ):
        super().__init__()
        dim_ff = dim_ff or (4 * d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.ffn = SparseFFN(
            d_model,
            dim_ff=dim_ff,
            sparsity=sparsity,
            dropout=dropout,
            dynamic=dynamic,
            deltaT=deltaT,
            Tend=Tend,
            alpha=alpha,
            max_size=max_size,
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: (seq_len, batch, d_model)
        attn_out, _ = self.attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        # residual + dropout + norm
        src = src + self.drop(attn_out)
        src = self.norm1(src)

        # sparse feed-forward
        inter = self.ffn(src)
        src = src + self.drop(inter)
        return self.norm2(src)


class SparseTE(nn.Module):
    def __init__(
        self, config,   sparsity: float = 0.01,
        dynamic: bool = True,
        deltaT: int = 6000,
        Tend: int = 150000,
        alpha: float = 0.1,
        max_size: int = int(1e8),
    ):
        super().__init__()
        self.input_proj = DataEmbedding(
            c_in=config.c_in, 
            d_model=config.d_model, 
            dropout=config.dropout
        )
        
        layer = SparseEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_ff=config.d_ff,
            sparsity=sparsity,
            dropout=config.dropout,
            dynamic=dynamic,
            deltaT=deltaT,
            Tend=Tend,
            alpha=alpha,
            max_size=max_size,
        )
        self.enc = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.d_model, config.c_out)


    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x: (batch, seq_len, c_in)
        x = self.input_proj(x).transpose(0, 1)  # (seq_len, batch, d_model)
        for layer in self.enc:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.head(x[0])# forecast from first time step
    

def small_world_chunker(inputs, outputs, nnz):
    """Utility function for small world initialization as presented in the write up Bipartite_small_world_network"""
    pair_distance = inputs.view(-1, 1) - outputs
    arg = torch.abs(pair_distance) + 1.0

    # lambda search
    L, U = 1e-5, 5.0
    lamb = 1.0  # initial guess
    itr = 1
    error_threshold = 10.0
    max_itr = 1000
    P = arg ** (-lamb)
    P_sum = P.sum()
    error = abs(P_sum - nnz)

    while error > error_threshold:
        assert (
            itr <= max_itr
        ), "No solution found; please try different network sizes and sparsity levels"
        if P_sum < nnz:
            U = lamb
            lamb = (lamb + L) / 2.0
        elif P_sum > nnz:
            L = lamb
            lamb = (lamb + U) / 2.0

            P = arg ** (-lamb)
            P_sum = P.sum()
            error = abs(P_sum - nnz)
            itr += 1
    return P

