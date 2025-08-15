import torch
import torch.nn as nn

from embed import PositionalEmbedding, TokenEmbedding, DataEmbedding

""" Standard fully trainable Transformer encoder (TE) """

class StandardTE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = DataEmbedding(
            c_in=config.c_in, 
            d_model=config.d_model, 
            dropout=config.dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, 
            nhead=config.nhead, 
            dim_feedforward=config.d_ff or 4 * config.d_model,
            dropout=config.dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.d_model, config.c_out)

    def forward(self, x):
        x = self.input_proj(x)     # (B, L, D)
        x = x.transpose(0, 1)      # (L, B, D)
        encoded = self.encoder(x)  # (L, B, D)
        last_token = encoded[-1]   # (B, D)
        return self.head(last_token)  # (B, C_out)

