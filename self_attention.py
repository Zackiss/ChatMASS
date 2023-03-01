import torch
import copy
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, head_num, dim_model, dropout=0.1):
        super(SelfAttention, self).__init__()
        # get dimension of word vectors
        assert dim_model % head_num == 0
        self.dim_k = dim_model // head_num
        # specify number of heads
        self.head_num = head_num
        # register linear layers used for projection
        self.project_linears = [copy.copy(nn.Linear(dim_model, dim_model)) for _ in range(3)]
        self.concat_linear = nn.Linear(dim_model, dim_model)
        self.QK_result = None
        # define dropout method
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask:
            mask = mask.unsqueeze(1)

        # number of words involved
        batch_size = query.size(0)

        # project query,key,value from (dim_model) to (head) x (dimension of word vector), assign for each word
        # Wq, Wk, Wv involved as weight matrices of project layers
        query, key, value = [linear_layer(mat).view(batch_size, -1, self.head_num, self.dim_k).transpose(1, 2)
                             for linear_layer, mat in zip(self.project_linears, (query, key, value))]
        # get dimension of transformed word vectors
        dim_k = query.size(-1)
        # calculate similarity with Query and Key, make it smooth by dividing dim_k
        similarity = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(dim_k)
        # apply mask on similarity matrix
        if mask is not None:
            similarity = similarity.masked_fill(mask == 0, -1e9)
        # apply softmax and dropout on similarity matrix
        self.QK_result = self.dropout(similarity.softmax(dim=-1))
        # calculate attention
        x = torch.matmul(self.QK_result, value)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.head_num * self.dim_k)
        )
        del query
        del key
        del value

        # use linear layer to concat heads (attentions for different words) into single smaller head
        return self.concat_linear(x)
