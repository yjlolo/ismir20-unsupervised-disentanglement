import copy
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tsnecuda import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from test import *
import model.model as module_arch
from data.audio_processor import *


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.track_loss = ['loss', 'recon', 'kld', 'lmse', 'contrast', 'cycle', 'cycle_mse', 'cycle_ce', 'pseudo', 'klc']

        self.train_metrics = MetricTracker(*self.track_loss, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(*self.track_loss, *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.pitch_map = {i: n for n, i in enumerate(data_loader.dataset.pitch_map)}
        self.dynamic_map = {i: n for n, i in enumerate(data_loader.dataset.dynamic_map)}
        self.pitchclass_map = {i: n for n, i in enumerate(data_loader.dataset.pitchclass_map)}
        self.tf_map = {v: data_loader.dataset.family_map[k] for k,v in data_loader.dataset.instrument_map.items()}

        self.plot_step = 25

        self.recon_sample = np.random.choice(valid_data_loader.sampler.indices, size=10, replace=False)
        pitches = np.random.choice(82, size=len(self.recon_sample))
        self.sample_to_pitch = {k: v for k, v in zip(self.recon_sample, pitches)}

        self.spec_ext = ExtractSpectrogram(sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL, mode='mel')
        self.x_max, self.x_min = 9.7666, -36.0437

        self.init_temp = self.model.temperature
        self.min_temp = self.model.min_temperature
        self.decay_rate = self.model.decay_rate

        self.pseudo_train = config['trainer']['pseudo_train']
        self.labeled = config['trainer']['labeled']
        self.labeled_sample = np.random.choice(data_loader.sampler.indices, size=int(len(data_loader.sampler.indices) * self.labeled), replace=False)

        self.freeze_encoder = config['trainer']['freeze_encoder']
        self.pitch_shift = config['trainer']['pitch_shift']

    def data_transform(self, x, **kwargs):
        def get_idx(at_time=0.2, pitch_shift=2):
            compensate_duration = 0.05
            load_duration = at_time + compensate_duration  # add 0.05s more after the targeted time instant
            # desired_idx = int(at_time * SR)
            if pitch_shift != 0:
                pitch_shift = np.random.randint(-pitch_shift, pitch_shift)
            # shift = -2
            scale = 2. ** (pitch_shift / 12.)
            idx_comp = int(compensate_duration * scale**(-1) * SR / HOP)  # the corresponding number of indices to be compensated
            if pitch_shift < 0:
                n_sample = int(scale**(-1) * load_duration * SR)
                # n_sample = int(scale**(-1) * load_duration * SR)
                # assert n_sample > desired_idx
                desired_idx = int((load_duration * SR) / HOP) - idx_comp

            if pitch_shift >= 0:
                n_sample = int(load_duration * SR)
                desired_idx = int((scale**(-1) * n_sample) / HOP) - idx_comp

            return pitch_shift, n_sample, desired_idx
        
        shift, n_sample, desired_idx = get_idx(**kwargs)
        
        x = LoadNpArray(n_sample=n_sample)(x)
        x = PitchShift(shift=shift)(x)

        x = ToTensor()(x)
        x = self.spec_ext(x)
        x = LogCompress()(x)
        x = Clipping(clip_min=self.x_min, clip_max=self.x_max)(x)
        x = MinMaxNorm(x_min=self.x_min, x_max=self.x_max)(x)
        x = x[:, :, desired_idx]
        return x, shift
   
    def get_gumb_temp(self, epoch, init_temp, min_temp, decay_rate):
        temp = np.maximum(init_temp * np.exp(-decay_rate * epoch), min_temp)
        return temp

    def get_ps_label(self, yp, ps):
        y_shift = torch.from_numpy(np.array(ps)).unsqueeze(-1).to(self.device)
        y_ps = yp + y_shift
        mask_l = torch.where(y_ps >= 0, torch.ones_like(y_ps), torch.zeros_like(y_ps))
        mask_u = torch.where(y_ps <= 81, torch.ones_like(y_ps), torch.zeros_like(y_ps))
        mask = mask_l * mask_u
        y_ps *= mask
        if self.pitch_shift == 0: assert (y_ps == yp).sum() == len(y_ps)
        return yp, y_ps, mask.float(), torch.ones_like(yp)

    def get_pseudo_label(self, logit, supervised_idx, pitch_label, pitch_shift):
        '''Algorithm for creating pseudo labels for pitch-shifted samples
        '''
        supervised = True if len(supervised_idx) > 0 else False
        # initialize masks for both original and pitch-shiftedd samples
        m, m_ps = torch.zeros_like(pitch_label).float(), torch.zeros_like(pitch_label).float()
        '''Original samples'''
        # pseudo labels are defined from the inferred catogrical distribution
        y_pseudo = torch.argmax(logit, dim=-1, keepdim=True)
        if supervised:
            supervised_idx = supervised_idx.long()
            # replace pseudo with supervised labels
            # NOTE: psuedo labels become true if supervised portion is 100%
            y_pseudo[supervised_idx] = pitch_label[supervised_idx]
            # only the supervised indices are un-masked for the orignal samples
            m[supervised_idx] = 1  # cross-entropy induced by pseudo labels will be masked

        '''Pitch-shifted samples'''
        # exploit pseudo labels if if
        if self.pseudo_train:
            m_ps += 1

        # un-mask supervised labels regardlessly
        if supervised:
            m_ps[supervised_idx] = 1

        if m_ps.gt(1).any(): print("mask has entry larger than 1 before being multiplied with exclusion mask")

        # further mask the out-of-range pitches based on pseudo labels
        _, y_ps_pseudo, m_ps_ext, _ = self.get_ps_label(y_pseudo, pitch_shift)
        m_ps *= m_ps_ext

        if m_ps.gt(1).any(): print("mask has entry larger than 1 AFTER being multiplied with exclusion mask")

        return y_pseudo, y_ps_pseudo, m, m_ps

    def get_data(self, x, n_semitone=2):
        for i, x_i in enumerate(x):
            x_ps, ps = self.data_transform(x_i, at_time=0.2, pitch_shift=n_semitone)
            x_ori, _ = self.data_transform(x_i, at_time=0.2, pitch_shift=0)
            if i == 0:
                ps_cat = [ps]
                x_ps_cat = x_ps
                x_ori_cat = x_ori
            else:
                ps_cat.append(ps)
                x_ps_cat= torch.cat([x_ps_cat, x_ps])
                x_ori_cat = torch.cat([x_ori_cat, x_ori])

        return x_ori_cat, x_ps_cat, ps_cat

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        torch.manual_seed(1111)
        for batch_idx, (x, idx, y) in enumerate(self.data_loader):
            supervised_idx = torch.from_numpy(np.array([i for i, v in enumerate(idx.numpy()) 
                                 if v in self.labeled_sample], dtype='float')).to(self.device)

            y = torch.stack(y, dim=1).to(self.device)
            yp = y[:, 1:2]
            x1, x2, ps_cat = self.get_data(x, n_semitone=self.pitch_shift)

            self.optimizer.zero_grad()
            x1_hat, h1, mu1, logvar1, z1_t, z1_p, logits1, prob1 = self.model(x1, yp)
            if self.model.gumbel:
                y_pseudo, y_ps_pseudo, m, m_ps = self.get_pseudo_label(logits1, supervised_idx, yp, ps_cat)
            else:
                y_pseudo, y_ps_pseudo, m, m_ps = self.get_ps_label(yp, ps_cat)

            x2_hat, h2, mu2, logvar2, z2_t, z2_p, logits2, prob2 = self.model(x2, y_ps_pseudo)

            # con_loss = self.nt_xent_criterion(mu1, mu2)

            dict_loss = self.criterion(self.model, self.pseudo_train, self.device,
                            x1, x1_hat, x2, x2_hat,
                            mu1, logvar1, z1_t, z1_p, 
                            mu2, logvar2, z2_t, z2_p, 
                            logits1=logits1, logits2=logits2, prob1=prob1, prob2=prob2,
                            epoch=epoch, mask=m_ps.float(), mask_y=m.float(),
                            y=y_pseudo.squeeze(-1), y_ps=y_ps_pseudo.squeeze(-1))

            for k, v in dict_loss.items():
                if torch.isnan(v): print(k)

            for name, p in self.model.named_parameters():
                if torch.isnan(p).any(): print(name)

            if dict_loss['cycle'].requires_grad:
                dict_loss['loss'].backward(retain_graph=True)
            else:
                dict_loss['loss'].backward()
            self.optimizer.step()
            pre_tim_op = copy.deepcopy(list(self.model.timbre_encoder.parameters()))
            pre_pitch_op = copy.deepcopy(list(self.model.pitch_encoder.parameters()))

            if dict_loss['cycle'].requires_grad:
                self.optimizer.zero_grad()
                dict_loss['cycle'].backward()
                if self.freeze_encoder:
                    for i, param in enumerate(self.model.timbre_encoder.parameters()):
                        param.grad[:] = 0
                    for i, param in enumerate(self.model.pitch_encoder.parameters()):
                        if param.grad is not None: param.grad[:] = 0

                self.optimizer.step()

            for name, p in self.model.named_parameters():
                if torch.isnan(p).any(): print(name)

            if self.model.gumbel:
                temp = self.get_gumb_temp(epoch, self.init_temp, self.min_temp, self.decay_rate)
                self.model.set_temperature(temp)
            else:
                temp = 0

            for track, output in zip(self.track_loss, dict_loss):
                assert track == output
                log_val = dict_loss[track].item()
                self.train_metrics.update(track, log_val)

            if batch_idx == self.len_epoch:
                break

            if batch_idx == 0:
                idx_cat = idx
                zt_cat, zp_cat = z1_t, z1_p
                yt_cat, yp_cat, yf_cat, yc_cat, yd_cat = y[:, 0:1], y[:, 1:2], y[:, -1:], y[:, 2:3], y[:, 3:4]
                x_cat, x_hat_cat = x1, x1_hat
                mu1_cat, logvar1_cat = mu1, logvar1
                mu2_cat, logvar2_cat = mu2, logvar2
                if prob1 is not None:
                    yp_hat_cat = torch.argmax(prob1, dim=-1, keepdim=True)
                else:
                    yp_hat_cat = None
            else:
                idx_cat = torch.cat([idx_cat, idx])
                zt_cat, zp_cat = torch.cat([zt_cat, z1_t]), torch.cat([zp_cat, z1_p])
                yt_cat = torch.cat([yt_cat, y[:, 0:1]], dim=0)
                yp_cat = torch.cat([yp_cat, y[:, 1:2]], dim=0)
                yf_cat = torch.cat([yf_cat, y[:, -1:]], dim=0)
                yc_cat = torch.cat([yc_cat, y[:, 2:3]], dim=0)
                yd_cat = torch.cat([yd_cat, y[:, 3:4]], dim=0)
                mu1_cat, logvar1_cat = torch.cat([mu1_cat, mu1]), torch.cat([logvar1_cat, logvar1])
                mu2_cat, logvar2_cat = torch.cat([mu2_cat, mu2]), torch.cat([logvar2_cat, logvar2])
                x_hat_cat = torch.cat([x_hat_cat, x1_hat])
                x_cat = torch.cat([x_cat, x1])
                if prob1 is not None:
                    yp_hat_cat = torch.cat([yp_hat_cat, torch.argmax(prob1, dim=-1, keepdim=True)])
                else:
                    yp_hat_cat = None


        self.writer.set_step(epoch, 'train')
        for track, output in zip(self.track_loss, dict_loss):
            assert track == output
            self.writer.add_scalar(track, self.train_metrics.avg(track))
        for met in self.metric_ftns:
            # if met.__name__ == 'cluster_var':
            #     self.train_metrics.update(met.__name__, met(mu1_cat.cpu(), yp_cat.cpu()))
            # if met.__name__ == 'kl_gauss':
            #     self.train_metrics.update(met.__name__, met(mu1_cat, logvar1_cat, mu2_cat, logvar2_cat).item())
            if met.__name__ == 'f1' and yp_hat_cat is not None:
                self.train_metrics.update(met.__name__, met(yp_hat_cat, yp_cat, n_class=82).item())
            if met.__name__ == 'cluster_acc' and yp_hat_cat is not None:
                self.train_metrics.update(met.__name__, met(yp_hat_cat, yp_cat))
            if met.__name__ == 'nmi' and yp_hat_cat is not None:
                self.train_metrics.update(met.__name__, met(yp_hat_cat, yp_cat))

            self.writer.add_scalar(met.__name__, self.train_metrics.avg(met.__name__))

        log = self.train_metrics.result()

        if epoch % self.plot_step == 0:
            yt_cat = yt_cat.squeeze(-1).detach().cpu().numpy()
            yp_cat = yp_cat.squeeze(-1).detach().cpu().numpy()
            yf_cat = yf_cat.squeeze(-1).detach().cpu().numpy()
            yc_cat = yc_cat.squeeze(-1).detach().cpu().numpy()
            yd_cat = yd_cat.squeeze(-1).detach().cpu().numpy()
            zt_2d = TSNE(n_components=2).fit_transform(mu1_cat.cpu().data.numpy())
            fig, ax = plt.subplots(2, 4, figsize=(4*5, 2*5))

            def plot_and_color(data, ax, label_map, labels, colors=None):
                n_class = len(np.unique(labels))
                if colors is not None:
                    assert n_class == len(colors)
                else:
                    random.seed(1111)
                    colors = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_class)]
                assert len(label_map.items()) == n_class
                for k, v in label_map.items():
                    target_data = data[labels == v]
                    ax.scatter(target_data[:, 0], target_data[:, 1], c=colors[v], label=k, alpha=0.7)

            plot_and_color(zt_2d, ax[0][0], INSTRUMENT_MAP, yt_cat, colors=INSTRUMENT_COLORS)
            plot_and_color(zt_2d, ax[0][2], self.pitch_map, yp_cat, colors=PITCH_COLORS)
            plot_and_color(zt_2d, ax[0][1], FAMILY_MAP, yf_cat, colors=None)
            plot_and_color(zt_2d, ax[0][3], self.dynamic_map, yd_cat, colors=None)
            ax[1][1].imshow(self.model.emb.weight.cpu().data.numpy().T, aspect='auto', origin='lower')

        else:
            fig = None 
            ax = None

        if self.do_validation:
            val_log = self._valid_epoch(epoch, fig, ax)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        log.update({"gumbel_temp": temp})

        return log

    def _valid_epoch(self, epoch, fig=None, ax=None):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        torch.manual_seed(1111)
        with torch.no_grad():
            for batch_idx, (x, idx, y) in enumerate(self.valid_data_loader):
                y = torch.stack(y, dim=1).to(self.device)
                yp = y[:, 1:2]
                x1, x2, ps_cat = self.get_data(x, n_semitone=self.pitch_shift)

                x1_hat, h1, mu1, logvar1, z1_t, z1_p, logits1, prob1 = self.model(x1, yp)
                y_pseudo, y_ps_pseudo, m, m_ps = self.get_ps_label(yp, ps_cat)

                x2_hat, h2, mu2, logvar2, z2_t, z2_p, logits2, prob2 = self.model(x2, y_ps_pseudo)

                x1_hat_swap, _ = self.model.decode(z2_t, z1_p)
                x2_hat_swap, _ = self.model.decode(z1_t, z2_p)

                dict_loss = self.criterion(self.model, self.pseudo_train, self.device,
                                x1, x1_hat, x2, x2_hat,
                                mu1, logvar1, z1_t, z1_p, 
                                mu2, logvar2, z2_t, z2_p, 
                                logits1=logits1, logits2=logits2, prob1=prob1, prob2=prob2,
                                epoch=epoch, mask=m_ps.float(), mask_y=m.float(),
                                y=y_pseudo.squeeze(-1), y_ps=y_ps_pseudo.squeeze(-1))

                for track, output in zip(self.track_loss, dict_loss):
                    assert track == output
                    log_val = dict_loss[track].item()
                    self.valid_metrics.update(track, log_val)

                if batch_idx == 0:
                    idx_cat = idx
                    zt_cat, zp_cat = z1_t, z1_p
                    yt_cat, yp_cat, yf_cat, yc_cat, yd_cat = y[:, 0:1], y[:, 1:2], y[:, -1:], y[:, 2:3], y[:, 3:4]
                    x_cat, x_hat_cat = x1, x1_hat
                    mu1_cat, logvar1_cat = mu1, logvar1
                    mu2_cat, logvar2_cat = mu2, logvar2
                    h_cat = h1
                    if prob1 is not None:
                        yp_hat_cat = torch.argmax(prob1, dim=-1, keepdim=True)
                    else:
                        yp_hat_cat = None
                else:
                    idx_cat = torch.cat([idx_cat, idx])
                    zt_cat, zp_cat = torch.cat([zt_cat, z1_t]), torch.cat([zp_cat, z1_p])
                    yt_cat = torch.cat([yt_cat, y[:, 0:1]], dim=0)
                    yp_cat = torch.cat([yp_cat, y[:, 1:2]], dim=0)
                    yf_cat = torch.cat([yf_cat, y[:, -1:]], dim=0)
                    yc_cat = torch.cat([yc_cat, y[:, 2:3]], dim=0)
                    yd_cat = torch.cat([yd_cat, y[:, 3:4]], dim=0)
                    mu1_cat, logvar1_cat = torch.cat([mu1_cat, mu1]), torch.cat([logvar1_cat, logvar1])
                    mu2_cat, logvar2_cat = torch.cat([mu2_cat, mu2]), torch.cat([logvar2_cat, logvar2])
                    x_hat_cat = torch.cat([x_hat_cat, x1_hat])
                    x_cat = torch.cat([x_cat, x1])
                    h_cat = torch.cat([h_cat, h1])
                    if prob1 is not None:
                        yp_hat_cat = torch.cat([yp_hat_cat, torch.argmax(prob1, dim=-1, keepdim=True)])
                    else:
                        yp_hat_cat = None

        self.writer.set_step(epoch, 'valid')
        for track, output in zip(self.track_loss, dict_loss):
            assert track == output
            self.writer.add_scalar(track, self.valid_metrics.avg(track))
        for met in self.metric_ftns:
            # if met.__name__ == 'cluster_var':
            #     self.valid_metrics.update(met.__name__, met(mu1_cat.cpu(), yp_cat.cpu()))
            # if met.__name__ == 'kl_gauss':
            #     self.valid_metrics.update(met.__name__, met(mu1_cat, logvar1_cat, mu2_cat, logvar2_cat).item())
            if met.__name__ == 'f1' and yp_hat_cat is not None:
                self.valid_metrics.update(met.__name__, met(yp_hat_cat, yp_cat, n_class=82).item())
            if met.__name__ == 'cluster_acc' and yp_hat_cat is not None:
                self.valid_metrics.update(met.__name__, met(yp_hat_cat, yp_cat))
            if met.__name__ == 'nmi' and yp_hat_cat is not None:
                self.valid_metrics.update(met.__name__, met(yp_hat_cat, yp_cat))

            self.writer.add_scalar(met.__name__, self.valid_metrics.avg(met.__name__))
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        if fig is not None:
            idx_cat = idx_cat.squeeze(-1).cpu().data.numpy()
            target_idx = np.array([np.where(idx_cat == i)[0] for i in self.recon_sample])
            non_empty_idx = np.vstack([(n, i) for n, i in enumerate(target_idx) if len(i) == 1])
            target_idx = np.vstack([i for i in target_idx if len(i) == 1])[:,0]
            self.recon_sample = [self.recon_sample[i[0]] for i in non_empty_idx]
            # target_idx = np.array([np.where(idx_cat == i)[0] for i in self.recon_sample])[:,0]
            target_pitch = np.array([self.sample_to_pitch[i] for i in self.recon_sample])
            origin = x_cat.cpu().data.numpy()[target_idx]
            output = x_hat_cat.cpu().data.numpy()[target_idx]
            h_cat = h_cat.cpu().data.numpy()[target_idx]
            zt_cat = zt_cat[target_idx]
            zp_target = self.model.emb.weight[target_pitch]
            if self.model.use_hp:
                zp_target = self.model.project_harmonic(zp_target)            
            output_pswap = self.model.decode(zt_cat, zp_target)[0]
            output_pswap = output_pswap.cpu().data.numpy()
            for m, (i, j, k, l) in enumerate(zip(origin, output, h_cat, output_pswap)):
                tmp= np.vstack([i, j])
                tmp_swap = np.vstack([i, l])
                if self.model.decoding == 'sf':
                    tmp_h = np.vstack([i, k])
                if m == 0:
                    pair = tmp
                    pair_swap = tmp_swap
                    if self.model.decoding == 'sf':
                        pair_h = tmp_h
                else:
                    pair = np.vstack([pair, tmp])
                    pair_swap = np.vstack([pair_swap, tmp_swap])
                    if self.model.decoding == 'sf':
                        pair_h = np.vstack([pair_h, tmp_h])
                      
            ax[1][2].imshow(pair.T, aspect='auto', origin='lower', vmin=0, vmax=1)
            for l in range(1, 2*len(self.recon_sample), 2):
                ax[1][2].axvline(x=l+0.5, lw=1.5, c='r')

            ax[1][3].imshow(pair_swap.T, aspect='auto', origin='lower', vmin=0, vmax=1)
            for l in range(1, 2*len(self.recon_sample), 2):
                ax[1][3].axvline(x=l+0.5, lw=1.5, c='r')

            if self.model.decoding == 'sf':
                ax[1][0].imshow(pair_h.T, aspect='auto', origin='lower', vmin=0, vmax=1)
                for l in range(1, 2*len(self.recon_sample), 2):
                    ax[1][0].axvline(x=l+0.5, lw=1.5, c='r')

            self.writer.set_step(epoch, 'train')
            self.writer.add_figure('tsne', fig)
            
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

