import torch as t 
import math

class PositionEmbedding(t.nn.Module):
    def __init__(self, d_model=80, max_len=2048, dropout=0.1):
        super(PositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.alpha = t.nn.Parameter(t.tensor(1.0))
        self.register_pe()
        self.dropout = t.nn.Dropout(dropout)

    def register_pe(self):
        self.register_buffer('pe', t.zeros(self.max_len, self.d_model))
        position = t.arange(0, self.max_len, dtype=t.float32).unsqueeze(1)
        div_term = t.exp(t.arange(0, self.d_model, 2, dtype=t.float32) * -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = t.sin(position * div_term)
        self.pe[:, 1::2] = t.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        """Add positional encoding.
        Args:
            x (t.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            t.Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        # x = x + self.alpha * self.pe[:, :x.size(1)]
        length = x.shape[1]
        x = x + self.alpha * self.pe.narrow(1, 0, length)
        return self.dropout(x)