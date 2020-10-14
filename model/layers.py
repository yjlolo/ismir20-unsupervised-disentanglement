import torch
import torch.nn as nn
import librosa
from data.audio_processor import * 


class HarmonicRenderer(nn.Module):
    def __init__(self, num_notes=82, notes_per_octave=12, n_fft=2048, n_mels=256, sr=22050, fmin='A0', share_param=False):
        super(HarmonicRenderer, self).__init__()
        self.num_notes = num_notes
        self.notes_per_octave = notes_per_octave
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sr = sr
        self.fmin = librosa.note_to_hz(fmin)
        self.share_param = share_param
        self.mel_filter = torch.from_numpy(librosa.filters.mel(sr, n_fft, n_mels=n_mels)[:, 1:]).cuda()

        self.f = torch.linspace(0, sr//2, n_fft//2).cuda() # create the frequency axis
        
        std_scale = 5
        if share_param:
          self.omega = [torch.ones(1)]
          self.std = [std_scale * torch.ones(1)]
        else:
          self.omega = [torch.ones(int(sr / 2 / self.get_f0(i))).cuda() for i in range(num_notes)]
          self.std = [std_scale * torch.ones(int(sr / 2 / self.get_f0(i))).cuda() for i in range(num_notes)]

        
        self.omega = torch.nn.ParameterList([torch.nn.Parameter(p) for p in self.omega]).cuda()
        self.std = torch.nn.ParameterList([torch.nn.Parameter(p) for p in self.std]).cuda()
        self.omega.requires_grad = False
        

    def get_f0(self, idx):
        return self.fmin * 2**(idx / self.notes_per_octave)

    def get_column(self, idx):
        if self.share_param:
          omega = self.omega[0] * torch.ones(k_max)
          std = self.std[0] * torch.ones(k_max)
        else:
          omega = self.omega[idx].cuda()
          std = self.std[idx].cuda()
        assert len(omega) == len(std)
        k_max = len(omega)
        f0 = self.get_f0(idx)
        
        y = 0
        for i, k in enumerate(range(1, k_max + 1)):
            y += omega[i] * torch.exp(-0.5 * ((self.f - k * f0) / std[i])**2).cuda()
            
        return y

    def normalize(self, x):
        x = LogCompress()(x)
        x = Clipping(clip_min=X_MIN, clip_max=X_MAX)(x)
        x = MinMaxNorm(x_min=X_MIN, x_max=X_MAX)(x)
        return x

    def get_w(self):
        w = torch.zeros(self.num_notes, self.n_mels)
        for col_idx in range(self.num_notes):
            w[col_idx] = self.normalize(torch.matmul(self.get_column(col_idx), self.mel_filter.float().t()))
        return w.cuda()

    def forward(self, y):
        w = self.get_w()
        return torch.matmul(y, w)
