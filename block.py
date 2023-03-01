import torch
import torch.nn as nn
import copy

from Layers.self_attention import SelfAttention


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.norm_linears = [copy.copy(nn.LayerNorm(config.embedding_num)) for _ in range(2)]
        # one block of the whole model consist of one attention and oen multi-layer perception
        self.attention = SelfAttention(config.head_num, config.dim_model, dropout=config.attention_drop)
        self.multi_perc = nn.Sequential(
            nn.Linear(config.embedding_num, 6 * config.embedding_num),
            nn.GELU(),
            nn.Linear(6 * config.embedding_num, config.embedding_num),
            nn.Dropout(config.resid_drop)
        )

    def forward(self, embedding):
        # the shortcut connection and forward process of block
        x = embedding + self.attention(self.norm_linears[0](embedding))
        x += self.multi_perc(self.norm_linears[1](embedding))
        return x
