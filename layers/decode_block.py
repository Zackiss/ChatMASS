import torch
import torch.nn as nn
from layers.self_attention import SelfAttention


class Block(nn.Module):
    """"""
    def __init__(self, config):
        super(Block, self).__init__()
        self.norm_linear = nn.LayerNorm(config.embedding_dim).to(config.device)
        self.norm_linear_fin = nn.LayerNorm(config.embedding_dim).to(config.device)
        # one block of the whole model consist of one attention and oen multi-layer perception
        self.multi_perc = nn.Sequential(
            nn.Linear(config.embedding_dim, 6 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(6 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.resid_drop)
        ).to(config.device)
        self.attention = SelfAttention(config, dropout=config.attention_drop)

    def forward(self, embedding, encoder_output):
        """the shortcut connection and forward process of block"""
        z = self.norm_linear(embedding)
        x = self.attention(z)
        x += embedding
        x += self.multi_perc(self.norm_linear_fin(embedding))
        return x