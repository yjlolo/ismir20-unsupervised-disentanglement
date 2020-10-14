import copy
from math import pi
import numpy as np
from librosa import note_to_hz, stft
from librosa.filters import mel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from base import BaseModel, BaseGMVAE
from model.loss import update_mask
from data.audio_processor import Melspec_to_MFCC, MinMaxNorm, LogCompress, Clipping, PitchShift
from model.layers import HarmonicRenderer


class Classifier(BaseModel):
    def __init__(self, input_size=256, target='instrument'):
        super().__init__()
        assert target in ['instrument', 'pitch']
        if target == 'instrument':
            n_class = 12
        elif target == 'pitch':
            n_class = 82
        self.target = target
        self.clfr = fc(3, [input_size, 512, 512, 512], activation='tanh', batchNorm='after')
        self.logit = fc(1, [512, n_class], activation=None, batchNorm=False)

    def get_feat(self, x):
        for layer in self.clfr:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)
        for layer in self.logit:
            x = layer(x)
        return x


class HarmonicVAE(BaseModel):
    def __init__(self, input_size=256, latent_dim=8, temperature=1.0, min_temperature=0.5, decay_rate=0.013862944,
                 n_pitch=82,
                 decoding='cat', pitch_embedding='onehot', learn_pitch_emb=False, encode_mfcc=False,
                 gumbel=False, hard_gumbel=True, use_hp=False, hp_share=False,
                 act='tanh', bn='none', decoder_arch='wide'):
        super(HarmonicVAE, self).__init__()

        assert decoding in ['sf', 'cat']
        if decoding == 'sf':
            assert pitch_embedding in ['harmonic', 'harmonic_v2']
        else:
            assert pitch_embedding in ['onehot', 'random']
        if use_hp:
            assert pitch_embedding in ['harmonic', 'harmonic_v2']

        freq_bin = input_size
        if encode_mfcc:
            self.mfcc = Melspec_to_MFCC(n_mels=freq_bin)
            input_size = 30

        self.n_pitch = n_pitch
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.freq_bin = freq_bin
        self.decoding = decoding
        self.encode_mfcc = encode_mfcc
        self.pitch_embedding = pitch_embedding
        self.learn_pitch_emb = learn_pitch_emb
        self.gumbel = gumbel
        self.hard_gumbel = hard_gumbel
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
        self.use_hp = use_hp
        self.hp_share = hp_share

        assert decoder_arch in ['wide', 'deep']
        assert bn in ['none', 'before', 'after']
        assert act in ['tanh', 'relu']

        '''Timbre Encoder'''
        self.timbre_encoder = nn.ModuleList([
            # *fc(3, [input_size, 512, 256, 128], activation=act, batchNorm=bn),
            *fc(3, [input_size, 512, 512, 512], activation=act, batchNorm=bn),
            Gaussian(512, latent_dim)
        ])
        if encode_mfcc:
            self.timbre_encoder = nn.ModuleList([self.mfcc, *self.timbre_encoder])

        '''Pitch Embedding'''
        self.emb = PitchEmbedding(mode=pitch_embedding, trainable=learn_pitch_emb,
                                  n_pitch=n_pitch, dim=latent_dim, n_mels=freq_bin)

        '''Pitch Encoder'''
        if not gumbel:
            self.pitch_encoder = self.emb
        else:
            self.pitch_encoder = nn.ModuleList([
                *fc(3, [freq_bin, 512, 512, 512], activation=act, batchNorm=bn),
                GumbelSoftmax(512, n_pitch),
                self.emb
            ])

        '''Decoder'''
        if decoding == 'sf':
            d_in = latent_dim
        else:
            if pitch_embedding == 'onehot':
                d_in = latent_dim + n_pitch
            elif pitch_embedding == 'random':
                d_in = latent_dim + latent_dim
        if use_hp:
            d_in = latent_dim * 2
        if decoder_arch == 'wide':
            pre_layers = fc(3, [d_in, 512, 512, 512], activation=act, batchNorm=bn)
        else:
            pre_layers = fc(3, [d_in, 128, 256, 512], activation=act, batchNorm=bn)
        self.decoder = nn.ModuleList([
            *pre_layers,
            *fc(1, [512, freq_bin], activation='tanh', batchNorm=False)
        ])

    def set_temperature(self, temperature):
        self.temperature = temperature

    def project_harmonic(self, x):
        return self.harmonic_projector(x)

    def encode_pitch(self, x, y, determine=False):
        if not self.gumbel:
            zp = self.pitch_encoder(batch_onehot(y, self.n_pitch))
            prob = None
            logits = None
        else:
            n_layers = len(self.pitch_encoder)
            for i, layer in enumerate(self.pitch_encoder):
                if i == n_layers - 2:
                    logits, prob, x = layer(x, self.temperature, self.hard_gumbel, determine)
                else:
                    x = layer(x)
            zp = x

        return prob, logits, zp

    def encode_timbre(self, x, determine=False):
        for layer in self.timbre_encoder:
            x = layer(x)
        mu, logvar, zt = x
        if determine:
            return mu, logvar, mu
        return mu, logvar, zt

    def decode(self, zt, zp):
        if self.decoding == 'sf' and not self.use_hp:
            for layer in self.decoder:
                zt = layer(zt)
            h = zt
            x_hat = torch.tanh(h + zp)
        else:
            h = torch.cat([zt, zp], dim=-1)
            for layer in self.decoder:
                h = layer(h)
            x_hat = h
        return x_hat, h

    def forward(self, x, y, determine=False):
        prob, logits, zp = self.encode_pitch(x, y, determine=determine)
        if self.use_hp:
            zp = self.project_harmonic(zp)
        mu, logvar, zt = self.encode_timbre(x, determine=determine)

        x_hat, h = self.decode(zt, zp)
        return x_hat, h, mu, logvar, zt, zp, logits, prob


def batch_onehot(y, n_class):
    y_onehot = torch.FloatTensor(y.shape[0], n_class).to(y.device)
    y_onehot.zero_()
    return y_onehot.scatter_(1, y, 1)


def fc(n_layer, n_channel, activation='tanh', batchNorm=True):
    """Construction of fc. layers.

    :param n_layer: Number of fc. layers
    :param n_channel: Number of in/output neurons for each layer ( len(n_channel) = n_layer + 1 )
    :param activation: [ 'tanh' | None ]
    :param batchNorm: [ True | False ]
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    if activation is not None:
        assert activation in ['relu', 'tanh', 'lrelu']

    if isinstance(batchNorm, str):
        assert batchNorm in ['none', 'before', 'after']
    elif isinstance(batchNorm, bool):
        if not batchNorm:
            batchNorm = 'none'
        else:
            raise ValueError("batchNorm cannot set True!")

    if activation == 'relu':
        act = nn.ReLU()
    elif activation == 'lrelu':
        act = nn.LeakyReLU()
    else:
        act = nn.Tanh()

    fc_layers = []
    for i in range(n_layer):
        layer = [nn.Linear(n_channel[i], n_channel[i + 1])]
        if batchNorm == 'before':
            layer.append(nn.BatchNorm1d(n_channel[i + 1]))
        if activation:
            layer.append(act)
        if batchNorm == 'after':
            layer.append(nn.BatchNorm1d(n_channel[i + 1]))
        fc_layers += layer

    return nn.ModuleList([*fc_layers])


class PitchEmbedding(nn.Module):
    def __init__(self, mode, trainable, n_pitch=82, dim=None, **kwargs):
        super(PitchEmbedding, self).__init__()
        assert mode == 'onehot'

        self.mode = mode
        self.trainable = False if mode == 'onehot' else trainable
        self.n_pitch = n_pitch
        self.dim = dim
        self.kwargs = kwargs
        self.emb = self._get_embedding()
        self.emb.weight.requires_grad = self.trainable
        self.weight = self.emb.weight

    def _get_embedding(self):
        if self.mode == 'onehot':
            emb = nn.Embedding(self.n_pitch, self.n_pitch)
            classes = torch.arange(0, self.n_pitch).view(-1, 1)
            w = torch.zeros(self.n_pitch, self.n_pitch)
            w.scatter_(1, classes, 1)
            emb.weight = nn.Parameter(w)
        else:
            raise NotImplementedError

        return emb

    def forward(self, yp):
        return self.emb(yp)


class GumbelSoftmax(nn.Module):
    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False, determine=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        # categorical_dim = 10
        if determine:
            y = logits
        else:
            y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False, determine=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard, determine=determine)
        return logits, prob, y


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.logvar = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        noise = torch.distributions.normal.Normal(0, 1).sample(sample_shape=std.size())
        noise = noise.to(std.device)
        z = mu + noise * std
        return z

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z
