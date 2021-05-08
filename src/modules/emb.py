import numpy as np
import torch
from torch import nn
import math

class Embeddings(nn.Module):
    def __init__(self, n_token, d_emb):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_emb)
        self.d_emb = d_emb

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_emb)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]
        

class CPEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_layers = []
        for k in config['attributes']:
            emb_layers += [[k, Embeddings(n_token=config['n_tokens'][k], d_emb=config['emb_dims'][k])]]
        self.emb_layers = nn.ModuleDict(emb_layers)
        
        sum_emb_dims = sum(config['emb_dims'].values())
        self.proj = nn.Linear(sum_emb_dims, config['d_model']) if config['d_model'] != sum_emb_dims else None
        self.pos_emb = PositionalEncoding(d_model=config['d_model'], max_len=config['max_len'])
        self.dropout = nn.Dropout(p=config['dropout'])
        

    def forward(self, x):
        embs = []
        for i, k in enumerate(self.emb_layers.keys()):
            embs += [self.emb_layers[k](x[..., i])]
        embs = torch.cat(embs, dim=-1)
        if self.proj is not None:
            embs = self.proj(embs)
        pos_emb = self.pos_emb(embs)
        return self.dropout(embs + pos_emb)