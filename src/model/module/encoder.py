import torch as t 
from src.model.block import TransformerEncoder
from .mask import ContextSelfAttentionMask
from src.model.layer import Conv1dSubsampling




class Encoder(t.nn.Module):
    def __init__(
        self, model_size=512, dropout=0.1, feed_forward_size=1024, hidden_size=128, 
        num_layer=12, left=250, right=3, num_head=4, vocab_size=6000
        ):
        super(Encoder, self).__init__()
        self.sub_sample = Conv1dSubsampling(
            input_size=80, model_size=model_size,dropout=dropout
        )
        self.self_attention_mask = ContextSelfAttentionMask(
            left=left, right=right
        )
        self.transformer_encoder = TransformerEncoder(
            input_size=model_size, feed_forward_size=feed_forward_size, hidden_size=hidden_size, 
            dropout=dropout, num_head=num_head, num_layer=num_layer
        )
        self.ctc_output = CtcOutput(
            model_size=model_size, vocab_size=vocab_size, language_type=5)
    
    def forward(self, feature, feature_length):
        feature, feature_pad_mask, feature_length, feature_max_length = self.sub_sample(feature, feature_length)
        self_attention_mask = self.self_attention_mask(feature_pad_mask)
        encoded_feature = self.transformer_encoder(feature, feature_pad_mask, self_attention_mask)
        ctc_output, ctc_language_output = self.ctc_output(encoded_feature)
        return encoded_feature, ctc_language_output, ctc_output, feature_length, feature_max_length



class CtcOutput(t.nn.Module):
    def __init__(self, model_size=512, vocab_size=6000, language_type=4):
        super(CtcOutput, self).__init__()
        self.language_type = language_type
        self.linear = t.nn.Linear(model_size, vocab_size, bias=True)
        t.nn.init.xavier_normal_(self.linear.weight)
        if self.language_type!=0:
            self.language_linear = t.nn.Linear(model_size, language_type, bias=True)
            t.nn.init.xavier_normal_(self.language_linear.weight)

    def forward(self, x):
        if self.language_type!=0:
            vocab_out = self.linear(x)
            language_out = self.language_linear(x)
            return vocab_out, language_out
        else:
            vocab_out = self.linear(x)
            return vocab_out, None





    