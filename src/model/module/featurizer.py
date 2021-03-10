import torchaudio as ta 
import torch as t
from src.model.layer.spec_augment import SpecAugment
from tqdm import tqdm 
import random
ta.set_audio_backend('sox_io')


class Featurizer(t.nn.Module):
    def __init__(
        self, n_freq_mask=1, n_time_mask=1, 
        freq_mask_length=20, time_mask_length=80
        ):
        super(Featurizer, self).__init__()
        self.n_freq_mask = n_freq_mask
        self.n_time_mask = n_time_mask
        self.freq_mask_length = freq_mask_length
        self.time_mask_length = time_mask_length
        self._spec_augment_layer = SpecAugment(
            n_time_mask=n_time_mask, n_freq_mask=n_freq_mask, time_mask_length=time_mask_length, freq_mask_length=freq_mask_length, p=0.1)
    
    def load_audiofiles(self, audio_files: list):
        sigs, lengths = [], []
        for file in audio_files:
            sig, _ = ta.load(file)
            sigs.append(sig[0])
            lengths.append(sig.shape[1])
        sigs = t.nn.utils.rnn.pad_sequence(sigs, batch_first=True)
        lengths = t.LongTensor(lengths) // 160 + 1
        return sigs, lengths

    def _log_fbank(self, sig):
        sig = ta.compliance.kaldi.fbank(sig, num_mel_bins=80).unsqueeze(0)
        return sig

    def _speed_perturb(self, sig, rate):
        if self.training:
            # rate = random.choice([0.9, 0.92,0.94,0.96,0.98,1.0,1.02,1.04,1.06,1.08,1.1])
            return t.nn.functional.interpolate(sig.unsqueeze(0), scale_factor=rate, mode='linear').squeeze(0)
        else:
            return sig

    def _cmvn(self, sig):
        sig = (sig - self.mean) / self.std
        return sig
    
    def _spec_augment(self, sig, sig_length):
        sig = self._spec_augment_layer(sig, sig_length)
        return sig

    def forward(self, sig):
        #single sample forward!!!!
        # sig = self._speed_perturb(sig)
        sig = self._log_fbank(sig)
        length = sig.size(1)
        sig = self._cmvn(sig)
        sig = self._spec_augment(sig, length)
        return sig, length

    def cal_cmvn(self, wav_file_list, use_cuda=False):
        means, stds = [], []
        for file in tqdm(wav_file_list):
            sig, length = self.load_audiofiles([file])
            if use_cuda:
                feature = self._log_fbank(sig.cuda())
            else:
                feature = self._log_fbank(sig)
            means.append(feature.mean())
            stds.append(feature.std())
        
        return (sum(means) / len(means)).item(), (sum(stds) / len(stds)).item()
    
    def line(self, file, rate=1.0):
        sig, _ = ta.load(file)
        if rate != 1.0:
            sig = self._speed_perturb(sig, rate)
        sig = self._log_fbank(sig)
        length = sig.size(1)
        sig = self._cmvn(sig)
        return sig, length

    def load_mean_std(self, path):
        a = t.load(path)
        self.mean = a['mean']
        self.std = a['std']
        print(f'loaded mean, std: {a}')

# inst_featurizer = Featurizer(mean=-4.972533750821624, std=3.6903241143683174)