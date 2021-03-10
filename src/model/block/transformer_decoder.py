import torch as t
from src.model.layer.multi_head_attention import MultiHeadAttentionBlock
from src.model.layer.feed_forward import FeedForward


class TransformerDecoder(t.nn.Module):
    """
    transformer decoder
    """
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head, num_layer):
        super(TransformerDecoder, self).__init__()
        self.layers = t.nn.ModuleList(
            [TransformerDecoderLayer(input_size, feed_forward_size, hidden_size, dropout, num_head) for _ in range(num_layer)]
        )

    def forward(self, net, src_mask, encoder_output, self_attention_mask, dot_attention_mask):
        no_pad_mask = ~src_mask.unsqueeze(-1)
        for layer in self.layers:
            net = layer(net, no_pad_mask, encoder_output, self_attention_mask, dot_attention_mask)
        return net


class TransformerDecoderLayer(t.nn.Module):
    def __init__(self, input_size, feed_forward_size, hidden_size, dropout, num_head):
        super(TransformerDecoderLayer, self).__init__()
        self.multi_head_self_attention_block = MultiHeadAttentionBlock(input_size, hidden_size, dropout, num_head)
        self.multi_head_dot_attention_block = MultiHeadAttentionBlock(input_size, hidden_size, dropout, num_head)
        self.feed_foward_block = FeedForward(input_size, feed_forward_size, dropout)

    def forward(self, src, src_mask, encoder_output, self_attention_mask, dot_attention_mask):
        net = self.multi_head_self_attention_block(src, src, self_attention_mask)
        # net.masked_fill_(src_mask, 0.0)
        net = self.multi_head_dot_attention_block(net, encoder_output, dot_attention_mask)
        # net.masked_fill_(src_mask, 0.0)
        net = self.feed_foward_block(net)  
        # net.masked_fill_(src_mask, 0.0)
        return net