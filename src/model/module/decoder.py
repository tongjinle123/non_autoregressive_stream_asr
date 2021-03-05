import torch as t 
import torch.nn.functional as F
from src.model.block import TransformerDecoder
from src.model.layer import Embedding
from src.model.module.mask import ContextSelfAttentionMask, ForwardSelfAttentionMask, PadMask


class Decoder(t.nn.Module):
    def __init__(
        self, model_size=512, dropout=0.1, feed_forward_size=1024, hidden_size=128, 
        num_head=4, num_layer=6, vocab_size=6000, blank_id=5, place_id=9, max_length=150
        ):
        super(Decoder, self).__init__()
        self.blank_id = blank_id
        self.place_id = place_id
        self.max_length = max_length
        self.forward_self_attention_mask = ForwardSelfAttentionMask()
        self.converter = PadMask()
        self.embedding = Embedding(
            vocab_size=vocab_size, embedding_size=model_size, padding_idx=0, max_length=256, dropout=dropout, 
            scale_word_embedding=True
        )
        self.transformer_decoder = TransformerDecoder(
            input_size=model_size, feed_forward_size=feed_forward_size, hidden_size=hidden_size, 
            dropout=dropout, num_head=num_head, num_layer=num_layer
        )
        self.output = DecoderOutput(model_size=model_size, vocab_size=vocab_size, language_type=4)

    def forward(self, token, token_mask, encoder_output, dot_attention_mask):
        feature = self.embedding(token)
        self_attention_mask = self.forward_self_attention_mask(token_mask)
        feature = self.transformer_decoder(
            feature, token_mask, encoder_output, self_attention_mask, dot_attention_mask
        )
        decoder_output, decoder_language_output = self.output(feature)
        return decoder_output, decoder_language_output



class DecoderOutput(t.nn.Module):
    def __init__(self, model_size=512, vocab_size=4000, language_type=4):
        super(DecoderOutput, self).__init__()
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


