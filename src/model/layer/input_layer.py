import torch as t 
from torch.nn import GELU
from src.model.module.mask import PadMask
from src.model.layer.position_embedding import PositionEmbedding


class Conv1dSubsampling(t.nn.Module):
    def __init__(self, input_size, model_size, dropout=0.0):
        super(Conv1dSubsampling, self).__init__()
        assert input_size == 80
        assert model_size == 512 # use 80 and 512 for fast implement
        self.pad_mask = PadMask()
        self.conv = t.nn.Sequential(
            t.nn.Conv1d(in_channels=80, out_channels=256, kernel_size=3, stride=2, groups=16),
            GELU(),
            t.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, groups=64),
            GELU(),
        )
        self.pos_emb = PositionEmbedding(model_size, max_len=500)

        t.nn.init.kaiming_normal_(self.conv[0].weight)
        t.nn.init.kaiming_normal_(self.conv[2].weight)
        t.nn.init.zeros_(self.conv[0].bias)
        t.nn.init.zeros_(self.conv[2].bias)

    def forward(self, x, feature_length):
        feature_length = (feature_length - 7) // 4 + 1
        feature_max_length = feature_length.max()
        x_mask = self.pad_mask(feature_length, feature_max_length)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.pos_emb(x)
        return x, x_mask, feature_length, feature_max_length