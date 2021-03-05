import torch as t 


class SpecAugment(t.nn.Module):
    def __init__(self, n_time_mask=2, n_freq_mask=2, time_mask_length=80, freq_mask_length=20, p=0.2):
        super(SpecAugment, self).__init__()
        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask
        self.time_mask_length = time_mask_length
        self.freq_mask_length = freq_mask_length
        self.register_buffer('position', t.arange(0, 2048))
        self.p = p

    def _mask_freq(self, feature):
        batch_size, max_time, max_freq = feature.size()
        device = feature.device
        sub_freq_mask_length = (t.rand(batch_size, device=device) * self.freq_mask_length).long()
        start = t.randint(low=0, high=max_freq, size=(batch_size,), device=device).long()
        end = start + sub_freq_mask_length
        position = self.position[:max_freq].repeat(batch_size, 1)
        mask = ((position > start.unsqueeze(-1)) & (position < end.unsqueeze(-1)))
        feature = feature.masked_fill(mask.unsqueeze(1), value=0.0)
        return feature

    def _mask_time(self, feature, feature_length):
        batch_size, max_time, max_freq = feature.size()
        device = feature.device
        sub_time_mask_length = (t.rand(batch_size, device=device) * max_time).long()
        feature_length_limit = (feature_length * self.p).long()
        sub_time_mask_length = t.Tensor.where(
            sub_time_mask_length, sub_time_mask_length < feature_length_limit, feature_length_limit
            )
        start = t.randint(low=0, high=max_time, size=(batch_size,), device=device).long()
        end = start + sub_time_mask_length
        position = self.position[:max_time].repeat(batch_size, 1)
        mask = ((position > start.unsqueeze(-1)) & (position < end.unsqueeze(-1)))
        feature = feature.masked_fill(mask.unsqueeze(2), value=0.0)
        return feature

    def forward(self, feature, feature_length):
        # feature B, L, F
        # feature length B, 
        if self.training:
            for _ in range(self.n_freq_mask):
                feature = self._mask_freq(feature)
            for _ in range(self.n_time_mask):
                feature = self._mask_time(feature, feature_length)
            return feature
        else:
            return feature

    