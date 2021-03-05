import torch as t 


class CtcOutput(t.nn.Module):
    def __init__(self, model_size=512, vocab_size=4000, language_type=4):
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



