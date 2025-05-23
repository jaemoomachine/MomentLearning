class MomentLinear(nn.Module):
    def __init__(
        self, 
        in_features : int, out_features : int, 
        bias: bool=True, 
        K: int=4, 
        beta: float=0.01
    ):
        
        super().__init__()
        self.in_f, self.out_f, self.K = in_features, out_features, K
        self.beta = beta
        # K+1 moment parameters
        self.gamma = nn.Parameter(torch.zeros(K+1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        # buffers for fixed weights and mask
        self.register_buffer('fixed_W', torch.zeros(out_features, in_features))
        self.register_buffer('mask', torch.zeros(out_features, in_features, dtype=torch.bool))

    def _hermite(self, U, k) -> torch.Tensor:
        H = torch.zeros_like(U)
        for p in range(0, k//2 + 1):
            coeff = ((-1)**p) / (math.factorial(p) * math.factorial(k-2*p) * (2**p))
            H = H + coeff * U.pow(k-2*p)
        return math.factorial(k) * H

    def _edgeworth_transform(self, U) -> torch.Tensor:
        X = U
        for k in range(3, self.K+1):
            X = X + (self.gamma[k] / math.factorial(k)) * self._hermite(U, (k-1))
        return X

    def partial_freeze(self):
        # freeze a fraction beta of weight entries each call
        numel = self.out_f * self.in_f
        num_to_freeze = int(self.beta * numel)
        if num_to_freeze <= 0:
            return
        # choose random positions that are not yet frozen
        avail = (~self.mask).flatten().nonzero(as_tuple=False).flatten()
        if avail.numel() == 0:
            self.gamma.requires_grad = False
            return
        choose = avail[torch.randperm(avail.numel(), device=avail.device)[:num_to_freeze]]
        new_mask = self.mask.clone().flatten()
        new_mask[choose] = True
        mask2d = new_mask.view(self.out_f, self.in_f)
        # compute fresh full weight
        with torch.no_grad():
            U = torch.randn(self.out_f, self.in_f, device=self.mask.device)
            sigma = torch.exp(self.gamma[2])
            X = self._edgeworth_transform(U)
            W_full = self.gamma[1] + sigma * X
            # update fixed_W only on newly frozen positions
            self.fixed_W[mask2d] = W_full[mask2d]
        # update mask
        self.mask.copy_(mask2d)
        if self.mask.all():
            self.gamma.requires_grad = False
            
    def full_freeze(self):
        # freeze all entries permanently
        with torch.no_grad():
            U = torch.randn(self.out_f, self.in_f, device=self.mask.device)
            sigma = torch.exp(self.gamma[2])
            X = self._edgeworth_transform(U)
            self.fixed_W[:] = self.gamma[1] + sigma * X
        self.mask[:] = True
        self.gamma.requires_grad = False

    def forward(self, x) -> torch.Tensor:

        mask_flat = self.mask.view(-1)  # (out_f*in_f,)
        U_flat = torch.zeros_like(mask_flat, dtype=x.dtype, device=x.device)
        num_unfrozen = (~mask_flat).sum()
        if num_unfrozen > 0:
            U_flat_unfrozen = torch.randn(num_unfrozen, device=x.device, dtype=x.dtype)
            U_flat[~mask_flat] = U_flat_unfrozen
        U = U_flat.view(self.out_f, self.in_f)

        sigma = torch.exp(self.gamma[2])
        W_rand = self.gamma[1] + sigma * self._edgeworth_transform(U)

        W = torch.where(self.mask, self.fixed_W, W_rand)
        return F.linear(x, W, self.bias)

class MomentFFN(nn.Module):
    def __init__(
        self, 
        d_model : int, d_ff : int, 
        bias: bool=True, 
        L_sub : int=2,
        K: int=4, 
        beta: float=0.01        

    ):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [d_model] + [d_ff]*(L_sub-1) + [d_model]
        for i in range(L_sub):
            self.layers.append(MomentLinear(in_features=dims[i], out_features=dims[i+1], 
                                            bias=True, K=K, beta=beta))
            if i < L_sub-1:
                self.layers.append(nn.GELU())

    def partial_freeze(self):
        for m in self.layers:
            if isinstance(m, MomentLinear):
                m.partial_freeze()

    def full_freeze(self):
        for m in self.layers:
            if isinstance(m, MomentLinear):
                m.full_freeze()

    def forward(self, x) -> torch.Tensor:
        out = x
        for m in self.layers:
            out = m(out)
        return out

class MoME(nn.Module):
    def __init__(
        self, 
        d_model : int, d_ff : int, 
        num_experts: int=4,
        bias: bool=True, 
        L_sub : int=2,
        K: int=4, 
        beta: float=0.01,
        topk: int=2,
        sparse_gating: bool=True
    ):    
        super().__init__()
        self.experts = nn.ModuleList([
            MomentFFN(d_model=d_model, d_ff=d_ff, bias=bias,
                      L_sub=L_sub, K=K, beta=beta)
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)
        self.topk = topk
        self.sparse_gating = sparse_gating

    def partial_freeze(self):
        for e in self.experts:
            e.partial_freeze()

    def full_freeze(self):
        for e in self.experts:
            e.full_freeze()

    def forward(self, x) -> torch.Tensor:
        w = F.softmax(self.gate(x), dim=-1)  # (N, E)

        if self.sparse_gating:
            idx = torch.topk(w, self.topk, dim=-1).indices        # (N, k)
            mask = torch.zeros_like(w).scatter(1, idx, 1.0)    # (N, E)
            w = w * mask
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)       # normalize over top-k only

        out = 0
        for i, exp in enumerate(self.experts):
            out += exp(x) * w[..., i:i+1]  # (N, d_model)

        return out
class HMoME_Node(nn.Module):
    def __init__(self, d_model, d_ff, bias, num_experts, expert_depth,
                 L_sub, K, topk, beta, sparse_gating):
        super().__init__()
        self.expert_depth = expert_depth
        self.M = num_experts
        self.k = topk
        self.sparse_gating = sparse_gating
        self.gate = nn.Linear(d_model, num_experts)

        if expert_depth == 1:
            self.branches = nn.ModuleList([
                MomentFFN(d_model=d_model, d_ff=d_ff, bias=bias,
                          L_sub=L_sub, K=K, beta=beta)
                for _ in range(num_experts)
            ])
        else:
            self.branches = nn.ModuleList([
                HMoME_Node(d_model=d_model, d_ff=d_ff, bias=bias,
                           num_experts=num_experts,
                           expert_depth=expert_depth - 1,
                           L_sub=L_sub, K=K, topk=topk,
                           beta=beta, sparse_gating=sparse_gating)
                for _ in range(num_experts)
            ])

    def forward(self, x):
        # x: either (B, S, D) or (N, D) flattened
        if x.dim() == 3:
            # sequence case
            B, S, D = x.shape
            logits = self.gate(x)                # (B, S, M)
            alphas = F.softmax(logits, dim=-1)   # (B, S, M)
            if self.sparse_gating:
                topk_idx = torch.topk(alphas, self.k, dim=-1).indices  # (B,S,k)
                mask     = torch.zeros_like(alphas).scatter_(-1, topk_idx, 1.0)
                alphas   = alphas * mask
                alphas   = alphas / (alphas.sum(-1, keepdim=True) + 1e-9)
            outs    = [branch(x) for branch in self.branches]  # list of (B,S,D)
            stacked = torch.stack(outs, dim=-1)               # (B,S,D,M)
            # unsqueeze at dim=2 => (B,S,1,M)
            out     = (stacked * alphas.unsqueeze(2)).sum(dim=-1)  # (B,S,D)
            return out

        elif x.dim() == 2:
            # flattened case (e.g. inside TransformerEncoderLayer)
            N, D = x.shape
            logits = self.gate(x)                # (N, M)
            alphas = F.softmax(logits, dim=-1)   # (N, M)
            if self.sparse_gating:
                topk_idx = torch.topk(alphas, self.k, dim=-1).indices  # (N,k)
                mask     = torch.zeros_like(alphas).scatter_(-1, topk_idx, 1.0)
                alphas   = alphas * mask
                alphas   = alphas / (alphas.sum(-1, keepdim=True) + 1e-9)
            outs    = [branch(x) for branch in self.branches]  # list of (N,D)
            stacked = torch.stack(outs, dim=-1)               # (N,D,M)
            # unsqueeze at dim=1 => (N,1,M)
            out     = (stacked * alphas.unsqueeze(1)).sum(dim=-1)  # (N,D)
            return out

        else:
            raise ValueError(f"Unsupported tensor dimension {x.dim()}")

    def partial_freeze(self):
        for branch in self.branches:
            if hasattr(branch, 'partial_freeze'):
                branch.partial_freeze()

    def full_freeze(self):
        for branch in self.branches:
            if hasattr(branch, 'full_freeze'):
                branch.full_freeze()

class MomentEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, d_ff: int, nhead: int,
        dropout: float = 0.1,
        bias: bool = True, 
        L_sub: int = 2,
        K: int = 4, 
        beta: float = 0.01        
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim= d_model, num_heads = nhead, dropout=dropout)
        self.ffn = MomentFFN(d_model=d_model, d_ff=d_ff, bias=bias, L_sub=L_sub, K=K, beta=beta)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        """
        src: (L, B, D)
        """
        attn_out, _ = self.attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src2 = self.norm1(src + self.drop(attn_out))

        L, B, D = src2.shape
        flat = src2.transpose(0, 1).reshape(B * L, D)              # (B*L, D)
        ffn_out = self.ffn(flat).reshape(B, L, D).transpose(0, 1)  # (L, B, D)

        return self.norm2(src2 + self.drop(ffn_out))
    
    def partial_freeze(self):
        self.ffn.partial_freeze()
        
    def full_freeze(self):
        self.ffn.full_freeze()
    
class MoMEEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model : int, d_ff : int, nhead: int,
        dropout: float = 0.1,
        num_experts: int=4,
        bias: bool=True, 
        L_sub : int=2,
        K: int=4, 
        beta: float=0.01,
        topk: int=2,
        sparse_gating: bool=True    
    ):
        
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.ffn = MoME(d_model=d_model, d_ff=d_ff, num_experts=num_experts, bias=bias, L_sub=L_sub, K=K, beta=beta,
                       topk=topk, sparse_gating=sparse_gating)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        """
        src: (L, B, D)
        """
        attn_out, _ = self.attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src2 = self.norm1(src + self.drop(attn_out))

        L, B, D = src2.shape
        flat = src2.transpose(0, 1).reshape(B * L, D)              # (B*L, D)
        ffn_out = self.ffn(flat).reshape(B, L, D).transpose(0, 1)  # (L, B, D)

        return self.norm2(src2 + self.drop(ffn_out))
    
    def partial_freeze(self):
        self.ffn.partial_freeze()
        
    def full_freeze(self):
        self.ffn.full_freeze()

class MomentTE(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.input_proj = DataEmbedding(
            c_in=config.c_in, 
            d_model=config.d_model, 
            dropout=config.dropout
        )
        d_ff = config.d_ff or 4 * config.d_model

        if config.ffn_type == 'moment_ffn':
            layer = MomentEncoderLayer(
                d_model=config.d_model, d_ff=d_ff,
                nhead=config.nhead, dropout=config.dropout,
                bias=config.bias, L_sub=config.L_sub,
                K=config.K, beta=config.beta
            )

        elif config.ffn_type == 'mome':
            layer = MoMEEncoderLayer(
                d_model=config.d_model, d_ff=d_ff,
                nhead=config.nhead, num_experts=config.num_experts,
                dropout=config.dropout, bias=config.bias,
                L_sub=config.L_sub, K=config.K, beta=config.beta,
                topk=config.topk, sparse_gating=config.sparse_gating
            )

        elif config.ffn_type == 'hmome':
            layer = HMoMEEncoderLayer(
                d_model=config.d_model, d_ff=d_ff,
                nhead=config.nhead, num_experts=config.num_experts,
                expert_depth=config.expert_depth,
                dropout=config.dropout, bias=config.bias,
                L_sub=config.L_sub, K=config.K, beta=config.beta,
                topk=config.topk, sparse_gating=config.sparse_gating
            )

        else:
            raise ValueError(f"Unsupported ffn_type: {config.ffn_type}")

        self.enc = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.d_model, config.c_out)

    def forward(self, x) -> torch.Tensor:
        x = self.input_proj(x).transpose(0, 1)
        for layer in self.enc:
            x = layer(x)        
        return self.head(x[0])

    def partial_freeze(self):
        # encoder.layers는 내부 ModuleList
        for lyr in self.enc:
            if hasattr(lyr, 'partial_freeze'):
                lyr.partial_freeze()

    def full_freeze(self):
        for lyr in self.enc:
            if hasattr(lyr, 'full_freeze'):
                lyr.full_freeze()

