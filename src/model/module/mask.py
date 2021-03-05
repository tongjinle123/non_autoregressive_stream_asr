from numpy.lib.function_base import place
import torch as t 
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad


class PadMask(t.nn.Module):
    def __init__(self):
        super(PadMask, self).__init__()
        self.register_buffer('position', t.arange(0, 2048, dtype=t.long))
    
    def forward(self, length, max_length: int):
        seq_range = self.position.narrow(0, 0, max_length)
        mask = seq_range < length.unsqueeze(1)
        return mask
        

class ForwardSelfAttentionMask(t.nn.Module):
    def __init__(self):
        super(ForwardSelfAttentionMask, self).__init__()

    def forward(self, mask):
        self_attention_mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
        forward_self_attention_mask = t.tril(self_attention_mask)
        return forward_self_attention_mask


class ContextSelfAttentionMask(t.nn.Module):
    def __init__(self, left=250, right=0):
        super(ContextSelfAttentionMask, self).__init__()
        self.left = left
        self.right = right
    
    def forward(self, mask):
        context_self_attention_mask = mask.unsqueeze(-1) * mask.unsqueeze(1)
        if self.left:
            context_self_attention_mask = t.triu(context_self_attention_mask, -self.left)
        if self.right:
            context_self_attention_mask = t.tril(context_self_attention_mask, self.right)
        return context_self_attention_mask


class TriggerDotAttentionMask(t.nn.Module):
    def __init__(self, blank_id=5, place_id=9,trigger_eps=5, max_token_length=150):
        super(TriggerDotAttentionMask, self).__init__()
        self.blank_id = float(blank_id)
        self.place_id = int(place_id)
        self.max_token_length = max_token_length
        self.trigger_eps = trigger_eps + 1
        self.pad_mask = PadMask()

    def forward(self, ctc_linear_output_id):
        # prcess ctc output 
        ctc_linear_output_id = t.argmax(ctc_linear_output_id, -1)
        length = ctc_linear_output_id.size(1)
        former = t.nn.functional.pad(ctc_linear_output_id, (1, 0), value=self.blank_id)
        current = t.nn.functional.pad(ctc_linear_output_id, (0, 1), value=self.blank_id)
        ctc_linear_output_id = ctc_linear_output_id.masked_fill((current == former).narrow(1, 0, length), self.blank_id)

        # 
        blank_mask = ctc_linear_output_id.ne(self.blank_id).long()
        # build input token 
        input_token_length = blank_mask.sum(-1) + 1
        input_token_length.clamp_max_(self.max_token_length+1)
        input_mask = self.pad_mask(input_token_length, input_token_length.max())
        input_ = input_mask.long() * self.place_id

        # build trigger_mask 
        pad_column = blank_mask.sum(1).ne(0).long()
        blank_mask = t.cat([blank_mask, pad_column.unsqueeze(1)], 1)
        split_size: List[int] = blank_mask.sum(1).tolist()
        position = blank_mask.nonzero().narrow(1,1,1).squeeze(1) + 1 + self.trigger_eps
        trigger_mask = self.pad_mask(position, length).long()
        trigger_mask = trigger_mask.split_with_sizes(split_size, 0)
        trigger_mask = pad_sequence(trigger_mask, batch_first=True).eq(1)
        return input_, input_mask, trigger_mask

class InputRegularizer(t.nn.Module):
    def __init__(self):
        super(InputRegularizer, self).__init__()

    def forward(self, input_, input_mask, length):
        seq_length = input_.size(1)
        input_ = input_.narrow(1, 0, length) if seq_length > length else \
            pad(input_, (0, int(length-seq_length)), value=0.0)
        input_mask = input_mask.narrow(1, 0, length) if seq_length > length else \
            pad(input_mask, (0, int(length-seq_length)), value=False)
        return input_, input_mask

class MaskRegularizer(t.nn.Module):
    def __init__(self):
        super(MaskRegularizer, self).__init__()
    
    def forward(self, trigger_mask, length):
        seq_length = trigger_mask.size(1)
        trigger_mask = trigger_mask.narrow(1, 0, length) if seq_length > length else \
            pad(trigger_mask, (0,0,0,int(length-seq_length)), value=0.0)
        return trigger_mask

        