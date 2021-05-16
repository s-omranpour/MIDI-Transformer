from typing import Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from fast_transformers.attention import AttentionLayer
from fast_transformers.builders import AttentionBuilder
from fast_transformers.masking import FullMask, LengthMask, TriangularCausalMask, BaseMask


class EncoderBlock(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_inner=None, 
                 n_heads=8, 
                 head_dims=None,
                 self_attention_type='full',
                 cross_attention_type='full',
                 is_decoder=False, 
                 dropout=0.1, 
                 activation='gelu'):

        super().__init__()

        d_inner = d_inner or d_model*4
        self.self_attention_type = self_attention_type
        self.self_attention = AttentionLayer(
            AttentionBuilder.from_kwargs(query_dimensions=d_model // n_heads).get(self_attention_type),
            d_model,
            n_heads,
            d_keys=d_model // n_heads,
            d_values=d_model // n_heads
        )

        
        if is_decoder:
            self.norm_cross = nn.LayerNorm(d_model)
            self.cross_attention = AttentionLayer(
                AttentionBuilder.from_kwargs(query_dimensions=d_model // n_heads).get(cross_attention_type),
                d_model,
                n_heads,
                d_keys=d_model // n_heads,
                d_values=d_model // n_heads
            )
        else:
            self.cross_attention = None
            self.norm_cross = None
            

        self.is_decoder = is_decoder
        self.ff1 = nn.Linear(d_model, d_inner)
        self.ff2 = nn.Linear(d_inner, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu


    def forward(self, 
                x, 
                x_mask=None, 
                x_length_mask= None, 
                memory= None, 
                memory_mask= None, 
                memory_length_mask= None):

        if x_mask is None:
            if self.self_attention_type.startswith('causal'):
                x_mask = TriangularCausalMask(x.size(1), device=x.device)
            else:
                x_mask = FullMask(x.size(1), device=x.device)
        elif isinstance(x_mask, torch.Tensor):
            x_mask = FullMask(x_mask, device=x.device)

        if x_length_mask is None:
            x_length_mask = LengthMask(x.new_full((x.size(0),), x.size(1), dtype=int), device=x.device)
        elif isinstance(x_length_mask, torch.Tensor):
            x_length_mask = LengthMask(x_length_mask, device=x.device)
        
        x = x + self.self_attention(
            x, x, x, 
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask,
        )
        x = self.norm1(x)

        if self.is_decoder and memory is not None:
            if memory_mask is None:
                memory_mask = FullMask(x.size(1), memory.size(1), device=memory.device)
            elif isinstance(memory_mask, torch.Tensor) and len(memory_mask.shape) < 3:
                memory_mask = FullMask(memory_mas.bool(), device=memory.device)
            
            if memory_length_mask is None:
                memory_length_mask = LengthMask(
                    memory.new_full((memory.size(0),), memory.size(1), dtype=int), 
                    device=x.device)

            elif isinstance(memory_length_mask, torch.Tensor):
                memory_length_mask = LengthMask(memory_length_mask, device=x.device)

            x = x + self.cross_attention(
                    x, memory, memory,
                    attn_mask=memory_mask,
                    query_lengths=None,
                    key_lengths=memory_length_mask,
                )
            x = self.norm_cross(x)

        y = self.dropout(self.activation(self.ff1(x)))
        y = self.dropout(self.ff2(y))
        return self.norm2(x+y)



class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList(
            EncoderBlock(
                d_model=config['d_model'], 
                d_inner=config.get('d_inner'), 
                n_heads=config.get('n_heads'),
                head_dims=config.get('head_dims'),
                self_attention_type=config['self_attention_type'],
                cross_attention_type=config['cross_attention_type'] if 'cross_attention_type' in config else config['self_attention_type'],
                is_decoder=config['is_decoder'],
                dropout=config['dropout'], 
                activation=config['activation']
            )
            for _ in range(config['n_layers'])
        )
        self.norm = nn.LayerNorm(config['d_model']) if config['final_norm'] else None


    def forward(self, 
                x: torch.Tensor, 
                x_mask: Union[torch.Tensor, BaseMask] = None, 
                x_length_mask: Union[torch.Tensor, LengthMask] = None, 
                memory: torch.Tensor = None, 
                memory_mask: Union[torch.Tensor, BaseMask] = None, 
                memory_length_mask: Union[torch.Tensor, LengthMask] = None):

        h = x
        for layer in self.layers:
            h = layer(h, x_mask, x_length_mask, memory, memory_mask, memory_length_mask)
        if self.norm is not None:
            h = self.norm(h)
        return h