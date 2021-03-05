import torch as t
import torch.nn.functional as F
import numpy as np


class MultiHeadAttentionBlock(t.nn.Module):
    """
    multi head attention layer combine with rezero
    """
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        self.rezero_alpha = t.nn.Parameter(t.Tensor([0]))
        self.rezero_beta = t.nn.Parameter(t.Tensor([0]))
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query, key, attention_mask, last_attention=None):
        if last_attention is not None:
            residual = query
            residual_attention = last_attention
            net, attention = self.multi_head_attention(query, key, attention_mask, last_attention)
            net = self.dropout(net)
            net = self.rezero_alpha * net
            net += residual
            attention += self.rezero_beta * attention
            return net, attention
        else:
            residual = query
            # residual_attention = last_attention
            net, attention = self.multi_head_attention(query, key, attention_mask, last_attention)
            net = self.dropout(net)
            net = self.rezero_alpha * net
            net += residual
            attention = self.rezero_beta * attention
            return net, attention


class MultiHeadAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(MultiHeadAttention, self).__init__()
        self.dropout = t.nn.Dropout(dropout)
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.output_dim = input_size
        self.key_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        self.query_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        t.nn.init.xavier_normal_(self.key_projection.weight)
        t.nn.init.xavier_normal_(self.query_projection.weight)
        self.scale = np.sqrt(self.hidden_size)
        self.linear = t.nn.Linear(self.num_head * self.hidden_size, input_size)
        t.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, query, key, attention_mask, last_attention):
        # key = value
        batch_size, query_lenth, query_dim = query.size()
        key_lenth = key.size(1)
        query_projection = self.query_projection(query).view(batch_size, query_lenth, self.num_head, self.hidden_size).permute(0, 2, 1, 3)
        # B, N, QL, H
        key_projection = self.key_projection(key).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 3, 1)
        # B, N, H, KL
        attention_matrix = (query_projection @ key_projection) / self.scale

        # B, N, QL, KL
        # if attention_mask is not None:
        if last_attention is not None:
            attention_matrix = attention_matrix + last_attention
        res_att = attention_matrix
        attention_matrix = attention_matrix.masked_fill(~attention_mask.unsqueeze(1), -float('inf'))
        attention_matrix = F.softmax(attention_matrix, -1)
        attention_matrix = attention_matrix.masked_fill(t.isnan(attention_matrix), 0)
        attention_matrix = self.dropout(attention_matrix)
        weighted = attention_matrix @ key_projection.transpose(-1, -2)
        # B, N, QL, KL * B, N, KL, H -> B, N，QL, H
        output = weighted.permute(0, 2, 1, 3).contiguous().view(batch_size, query_lenth, self.num_head * self.hidden_size)
        output = self.linear(output)
        return output, res_att