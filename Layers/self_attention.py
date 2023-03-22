import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    """"""
    def __init__(self, config, dropout=0.1):
        super(SelfAttention, self).__init__()
        # get dimension of word vectors
        assert config.dim_model % config.head_num == 0
        self.config = config
        self.to(self.config.device)
        dropout = config.attention_drop
        self.dim_k = config.dim_model // config.head_num
        # specify number of heads
        self.head_num = config.head_num
        # register linear layers used for projection
        self.linear_Q = nn.Linear(config.dim_model, config.dim_model)
        self.linear_K = nn.Linear(config.dim_model, config.dim_model)
        self.linear_V = nn.Linear(config.dim_model, config.dim_model)
        self.project_linears = [self.linear_Q, self.linear_K, self.linear_V]
        self.concat_linear = nn.Linear(config.dim_model, config.dim_model)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.QK_result = None
        # define dropout method
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, norm_embed: torch.Tensor, query=None, key=None, value=None):
        """"""
        query = norm_embed if query is None else query
        key = norm_embed if key is None else key
        value = norm_embed if value is None else value

        batch_size, seq_length, embed_dim = query.size()
        assert batch_size > 0
        assert seq_length <= self.config.block_size
        assert embed_dim > 0
        # number of words involved
        # project query,key,value from (dim_model) to (head) x (dimension of word vector), assign for each word
        # Wq, Wk, Wv involved as weight matrices of project layers
        query, key, value = [linear_layer(mat).view(batch_size, -1, self.head_num, self.dim_k).transpose(1, 2)
                             for linear_layer, mat in zip(self.project_linears, (query, key, value))]
        # get dimension of transformed word vectors
        dim_k = query.size(-1)
        # calculate similarity with Query and Key, make it smooth by dividing dim_k
        similarity = torch.matmul(query, key.transpose(-2, -1))
        similarity /= np.sqrt(dim_k)
        # apply mask on similarity matrix
        similarity = similarity.masked_fill(self.mask[:, :, :seq_length, :seq_length] == 0, float("-inf"))

        # apply softmax and dropout on similarity matrix
        self.QK_result = self.dropout(similarity.softmax(dim=-1))
        del query
        del key
        # calculate attention
        x = torch.matmul(self.QK_result, value)
        del value
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.head_num * self.dim_k)
        )
        # use linear layer to concat heads (attentions for different words) into single smaller head
        return self.concat_linear(x)
