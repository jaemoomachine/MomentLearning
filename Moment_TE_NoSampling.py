import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from embed import PositionalEmbedding, TokenEmbedding, DataEmbedding

class MomentLinear_NoSampling(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        K: int = 4
    ):
        super().__init__()
        self.in_f, self.out_f, self.K = in_features, out_features, K

        # learnable moment parameters γ₀…γ_K
        self.gamma = nn.Parameter(torch.zeros(K+1))

        # optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # sample U once and fix it
        U = torch.randn(out_features, in_features)
        self.register_buffer('U', U)
        self.frozen = False

    def _hermite(self, U: torch.Tensor, k: int) -> torch.Tensor:
        # probabilists' Hermite polynomial H_k(U)
        H = torch.zeros_like(U)
        for p in range(0, k//2 + 1):
            coeff = ((-1)**p) / (math.factorial(p) * math.factorial(k-2*p) * (2**p))
            H = H + coeff * U.pow(k-2*p)
        return math.factorial(k) * H

    def _edgeworth_transform(self, U: torch.Tensor) -> torch.Tensor:
        # X = U + Σ_{k=3}^K [ γₖ / k! * H_{k-1}(U) ]
        X = U
        for k in range(3, self.K+1):
            X = X + (self.gamma[k] / math.factorial(k)) * self._hermite(U, k-1)
        return X

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        returns: (..., out_features)
        """
        # W = γ₁ + exp(γ₂) * EdgeworthTransform(U)
        sigma = torch.exp(self.gamma[2])
        X = self._edgeworth_transform(self.U)
        W = self.gamma[1] + sigma * X

        # apply linear transform
        return F.linear(x, W, self.bias)
    
    
class MomentFFN_NoSampling(nn.Module):
    def __init__(
        self, 
        d_model : int, d_ff : int, 
        bias: bool=True, 
        L_sub : int=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [d_model] + [d_ff]*(L_sub-1) + [d_model]
        for i in range(L_sub):
            self.layers.append(MomentLinear_NoSampling(in_features=dims[i], out_features=dims[i+1], 
                                            bias=True))
            if i < L_sub-1:
                self.layers.append(nn.GELU())

    def forward(self, x) -> torch.Tensor:
        out = x
        for m in self.layers:
            out = m(out)
        return out
