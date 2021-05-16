import numpy as np
import torch
from torch import nn
import math
from collections import OrderedDict

from .utils import simple_sample


class CPHeadLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        heads = []
        for k in config['attributes']:
            d = config['emb_dims']['ttype'] if k != 'ttype' else 0
            heads += [[k, nn.Linear(config['d_model'] + d, config['n_tokens'][k])]]
        self.heads = nn.ModuleDict(heads)
        self.type_emb = None

    def register_embeddings(self, type_emb):
        self.type_emb = type_emb

    def forward(self, h, y):
        logits = {'ttype': self.heads['ttype'](h)}
        y_type = self.type_emb(y[..., 0])
        h_ext = torch.cat([h, y_type], dim=-1)
        for k in self.config['attributes'][1:]:
            logits[k] = self.heads[k](h_ext)
        return OrderedDict(logits)

    def infer(self, h, temperatures:dict=None):
        def sample(h, key):
            t = 1. if temperatures is None else temperatures[key]
            logits = self.heads[key](h)[0]
            return simple_sample(logits / t)[..., 0].unsqueeze(0)

        y_type = sample(h, 'ttype')
        h_cat = torch.cat([h, self.type_emb(y_type)], dim=-1)
        res = [y_type[0, -1]]
        for k in self.config['attributes'][1:]:
            res += [sample(h_cat, k)[0, -1]]
        
        return np.array(res)
