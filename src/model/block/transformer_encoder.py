import torch as t
from src.model.layer.multi_head_attention import MultiHeadAttentionBlock
from src.model.layer.feed_forward import FeedForward


class TransformerEncoder(t.nn.Module):
    """
    transformer encoder
    """
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer):
        super(TransformerEncoder, self).__init__()
        self.layers = t.nn.ModuleList(
            [TransformerEncoderLayer(input_size, feed_forward_size, hidden_size, dropout, num_head) for _ in range(num_layer)]
        )
    def forward(self, net, src_mask, self_attention_mask):
        non_pad_mask = ~src_mask.unsqueeze(-1)
        for layer in self.layers:
            net = layer(net, non_pad_mask, self_attention_mask)
        return net


class TransformerEncoderLayer(t.nn.Module):
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention_block = MultiHeadAttentionBlock(input_size, hidden_size, dropout, num_head)
        self.feed_foward_block = FeedForward(input_size, feed_forward_size, dropout)

    def forward(self, src, src_mask, self_attention_mask):
        net = self.multi_head_attention_block(src, src, self_attention_mask)
        # net.masked_fill_(src_mask, 0.0)
        net = self.feed_foward_block(net)
        # net.masked_fill_(src_mask, 0.0)
        return net