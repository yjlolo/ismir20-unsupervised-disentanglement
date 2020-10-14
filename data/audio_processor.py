import sys
import numpy as np
import math
import librosa
import torch
from torch import nn
from nnAudio import Spectrogram


SR = 22050
NFFT = 2048
HOP = 256
NMEL = 256
NBIN = 210
DURATION = 1.875
CHUNK_DUR = 0.5
CHUNK_SIZE = int(SR * CHUNK_DUR) // HOP
PITCH_SHIFT = 2
X_MIN = -36.0437
X_MAX = 9.7666

use_cuda = True if torch.cuda.is_available() else False


class ReadAudio():
    def __init__(self, sr=SR, offset=0.0, verbose=True, duration=DURATION):
        self.sr = sr
        self.offset = offset
        self.verbose = verbose
        self.duration = duration

    def __call__(self, x):
        y, _ = librosa.load(x, sr=self.sr, duration=self.duration, offset=self.offset) 
        if self.verbose:
            print(str(x), len(y) / self.sr)
        return y


class FixLength():
    def __init__(self, target_duration=15, sr=SR):
        self.target_len = target_duration
        self.sr = SR
        self.fix_len = int(target_duration * SR)

    def __call__(self, x):
        return librosa.util.fix_length(x, self.fix_len)


class Zscore():
    def __init__(self, divide_sigma=False):
        self.divide_sigma = divide_sigma

    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        assert x.ndim <= 2
        x -= x.mean(axis=0)
        if self.divide_sigma:
            x /= x.std(axis=0)
        return x


class LogCompress():
    def __call__(self, x):
        return torch.log(sys.float_info.epsilon + x)


class Clipping():
    def __init__(self, clip_min=-100, clip_max=100):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, x):
        x[x <= self.clip_min] = self.clip_min
        x[x >= self.clip_max] = self.clip_max
        return x


class MinMaxNorm:
    def __init__(self, min_val=-1, max_val=1, x_min=None, x_max=None):
        self.min_val = min_val
        self.max_val = max_val
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x):
        if self.x_min is None:
            self.x_min = x.min()

        if self.x_max is None:
            self.x_max = x.max()

        nom = x - self.x_min
        den = self.x_max - self.x_min

        return (self.max_val - self.min_val) * (nom / den) + self.min_val


class PickOneFrame:
    def __init__(self, frame_idx):
        self.frame_idx = frame_idx
    def __call__(self, x):
        return x[:, :, self.frame_idx:self.frame_idx + 1]

class LoadNpArray():
    def __init__(self, n_sample=None):
        self.n_sample = n_sample
    def __call__(self, x):
        x = np.load(x)
        if self.n_sample is not None:
            x = x[:self.n_sample]
        return x


class LoadTorchTensor():
    def __call__(self, x):
        x = torch.load(x)
        x = x.cuda() if use_cuda else x
        return x


class ToTensor():
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            y = x.cuda() if use_cuda else x
        else:
            y = torch.from_numpy(x).cuda() if use_cuda else torch.from_numpy(x)
        return y


class Transpose():
    def __call__(self, x):
        return x.transpose_(1, -1)


class Melspec_to_MFCC(nn.Module):
    def __init__(self, n_mels=256, n_mfcc=128, retain=30, norm='ortho'):
        super().__init__()
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.retain = retain
        self.norm = norm
        self.n = torch.arange(float(n_mels))
        self.k = torch.arange(float(n_mfcc)).unsqueeze(1)
    def forward(self, x):
        dct = torch.cos(math.pi / float(self.n_mels) * (self.n + 0.5) * self.k)  # size (n_mfcc, n_mels)
        if self.norm is None:
            dct *= 2.0
        else:
            assert self.norm == "ortho"
            dct[0] *= 1.0 / math.sqrt(2.0)
            dct *= math.sqrt(2.0 / float(self.n_mels))
        
        dct = dct.to(x.device)
        return torch.matmul(x, dct.transpose(0, 1))[:, :self.retain]


class ExtractSpectrogram():
    """Wrapper for nnAudio Spectrogram layers; to accomodate mutiple choices of spectrograms"""
    def __init__(self, sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL, n_bins=NBIN, mode='mel'):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_bins = n_bins

        if mode == 'mel':
            self.spectrogram = Spectrogram.MelSpectrogram(sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, fmin=20, fmax=11000)
        elif mode == 'cqt':
            self.spectrogram = Spectrogram.CQT(sr=sr, hop_length=hop_length, fmin=22.5, n_bins=n_bins, bins_per_octave=24, pad_mode='constant')

    def __call__(self, x):
        return self.spectrogram(x.float())


class PitchShift():
    def __init__(self, shift):
        self.shift = shift
    def __call__(self, x):
        # shift = np.random.randint(-self.shift, self.shift)
        scale = 2. ** (self.shift / 12.)
        window = len(x)
        xp = np.arange(window, dtype=np.float32)
        x = np.interp(scale * xp, np.arange(window, dtype=np.float32), x).astype(np.float32)
        return x

class SpecChunking():
    def __init__(self, duration=CHUNK_DUR, sr=SR, hop_length=HOP, only_first_seg=False):
        """
        Slice spectrogram into non-overlapping chunks. Discard chunks shorter than the specified duration.

        :params duration: the duration (in sec.) of each spectrogram chunk
        :params sr: sampling frequency used to read waveform; used to calculate the chunk size
        :params hop_length: hop size used to derive spectrogram; used to calculate the chunk size
        """
        self.duration = duration
        self.sr = sr
        self.hop_length = hop_length
        self.chunk_size = int(sr * duration) // hop_length
        self.only_first_seg = only_first_seg
        #self.overlap_size = int(self.chunk_size * overlap)
        #self.reverse = reverse

    def __call__(self, x):
        time_dim = -1  # assume input spectrogram with shape (freq, time) or (batch, freq, time)

        y = torch.split(x, self.chunk_size, dim=time_dim)[:-1]
        y = torch.cat(y, dim=0)

        if self.only_first_seg:
            y = y[0].unsqueeze(0)

        return y
