import numpy as np
import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cycle_loss(z1_t, z2_t, z1_t_swap, z2_t_swap,
               logits1, logits2, logits1_swap, logits2_swap):
    l1 = mse_loss(z1_t_swap, z1_t)
    l2 = mse_loss(z2_t_swap, z2_t)
    cycle_mse = l1 + l2
    if logits1 is not None and logits1_swap is not None:
        c1 = ce_loss(logits1_swap, logits1.argmax(dim=-1))
        c2 = ce_loss(logits2_swap, logits2.argmax(dim=-1))
        cycle_ce = c1 + c2
    else:
        cycle_ce = torch.zeros(1)

    return cycle_mse + cycle_ce, cycle_mse, cycle_ce


def nr_vae_elbo(model, pseudo_train, device,
                x1, x1_hat, x2, x2_hat,
                mu1, logvar1, z1_t, z1_p, 
                mu2, logvar2, z2_t, z2_p, 
                w_recon=1, w_kl=1, w_lmse=1, w_contrast=1, w_cycle=1, w_pseudo=1,
                logits1=None, logits2=None, prob1=None, prob2=None,
                y=None, y_ps=None, mask_y=None, mask=None,
                pretrain_step=None, epoch=None):

    def cycle_forward(model, z1_t, z2_t, z1_p, z2_p, y, y_ps):
        x1_hat_swap, _ = model.decode(z2_t, z1_p)
        x2_hat_swap, _ = model.decode(z1_t, z2_p)
        _, logits1_swap, _ = model.encode_pitch(x1_hat_swap, y=y)
        _, _, z2_t_swap = model.encode_timbre(x1_hat_swap)
        _, logits2_swap, _ = model.encode_pitch(x2_hat_swap, y=y_ps)
        _, _, z1_t_swap = model.encode_timbre(x2_hat_swap)
        return logits1_swap, logits2_swap, z1_t_swap, z2_t_swap

    l_recon = mse_loss(x1_hat, x1) + mse_loss(x2_hat, x2, mask=mask)
    l_kld = kld_gauss(mu1, logvar1) + kld_gauss(mu2, logvar2, mask=mask) if w_kl > 0 else torch.zeros(1).to(device)
    l_lmse = mse_loss(z1_t, z2_t) if w_lmse > 0 else torch.zeros(1).to(device)

    nt_xent_criterion = NTXentLoss(x1.device, len(x1), temperature=0.5, use_cosine_similarity=True)
    l_contrast = nt_xent_criterion(mu1, mu2) if w_contrast > 0 else torch.zeros(1).to(device)

    if w_cycle > 0 and epoch > pretrain_step:
        logits1_swap, logits2_swap, z1_t_swap, z2_t_swap = cycle_forward(model, z1_t, z2_t, z1_p, z2_p, y, y_ps)
        l_cycle, l_cycle_mse, l_cycle_ce = cycle_loss(z1_t, z2_t, z1_t_swap, z2_t_swap, logits1, logits2, logits1_swap, logits2_swap)
    else: 
        l_cycle = torch.zeros(1).to(device)
        l_cycle_mse = torch.zeros(1).to(device)
        l_cycle_ce = torch.zeros(1).to(device)

    if logits1 is not None and logits2 is not None:
        if w_pseudo > 0 and pseudo_train:
            l_pseudo = ce_loss(logits1, y, mask=mask_y) + ce_loss(logits2, y_ps, mask=mask)
        else:
            l_pseudo = torch.zeros(1).to(device)
        l_klc = kld_component(logits1, prob1, torch.ones_like(prob1)*(1/prob1.size(-1)), mask=None)
        l_klc += kld_component(logits2, prob2, torch.ones_like(prob1)*(1/prob1.size(-1)), mask=None)
    else:
        l_pseudo = torch.zeros(1).to(device)
        l_klc = torch.zeros(1).to(device)
    
    if epoch <= pretrain_step:
        l = w_recon * l_recon
    else:
        l = w_recon * l_recon + w_kl * l_kld + w_lmse * l_lmse + w_contrast * l_contrast + w_pseudo * l_pseudo + l_klc

    return {
        'loss': l,
        'recon': l_recon,
        'kld': l_kld,
        'lmse': l_lmse,
        'contrast': l_contrast,
        'cycle': l_cycle,
        'cycle_mse': l_cycle_mse,
        'cycle_ce': l_cycle_ce,
        'pseudo': l_pseudo,
        'klc': l_klc
    }


def ce_loss(output, target, reduction='none', mask=None):
    if mask is not None and mask.sum() == 0:
        return torch.zeros(1).to(output.device)
    # print("check inputs to ce_loss", torch.isnan(output).any(), torch.isnan(target).any())
    l = F.cross_entropy(output, target, reduction=reduction).unsqueeze(-1)
    if torch.isnan(l.sum()): print('ce before mask is nan')
    if mask is not None:
        effect_len = sum(mask)
        l *= mask
        if torch.isnan(l.sum()): print('ce after mask is nan')
    else:
        effect_len = output.shape[0]
    l = l.sum(-1)
    if torch.isnan(l).any(): print('ce before norm is nan')
  
    l_norm = l.sum().div(effect_len)
    if torch.isnan(l_norm).any(): print('ce after norm is nan', effect_len)
    return l_norm


def mse_loss(output, target, mask=None):
    if mask is not None and mask.sum() == 0:
        return torch.zeros(1).to(output.device)

    l = F.mse_loss(output, target, reduction='none')
    if mask is not None: 
        effect_len = sum(mask)
        l *= mask
    else:
        effect_len = output.shape[0]
    l = l.sum(-1)# .div(mask.sum(-1, keepdim=True))
    l_norm = l.sum().div(effect_len)

    return l_norm


def kld_gauss(q_mu, q_logvar, mu=None, logvar=None, avg_batch=True, mask=None):
    if mask is not None and mask.sum() == 0:
        return torch.zeros(1).to(q_mu.device)

    if mu is None:
        mu = torch.zeros_like(q_mu)
    if logvar is None:
        logvar = torch.zeros_like(q_logvar)

    l = 1 + q_logvar - logvar - (torch.pow(q_mu - mu, 2) + torch.exp(q_logvar)) / torch.exp(logvar) 
    l *= -0.5 
   
    if mask is not None: 
        effect_len = sum(mask)
        l *= mask
    else:
        effect_len = q_mu.shape[0]
    
    l_norm = l.sum(dim=-1)

    if avg_batch:
        l_norm = l_norm.sum(dim=0).div(effect_len)
    
    return l_norm


def class_weighted_kld(qy, q_mu, q_logvar, mu_lookup, logvar_lookup, y=None, mask=None, target_len=None):
    target_device = q_mu.device
    qy = qy.to(target_device)
    batch, n_component = list(qy.size())
    l_norm = torch.zeros(batch, n_component, device=target_device)
    for k_i in torch.arange(0, n_component, device=target_device):
        l_norm[:, k_i] = kld_gauss(q_mu, q_logvar, mu_lookup(k_i), logvar_lookup(k_i), mask=mask, target_len=target_len, avg_batch=False)
        l_norm[:, k_i] *= qy[:, k_i]  # padded instances have been taken care of in `kld_gauss`

    # l_norm = l_norm.sum(-1).div(n_component).sum(0).div(batch)
    l_norm = l_norm.sum(-1).sum(0).div(batch)
    return l_norm


def kld_component(qy_logit, qy, py_logit, mask=None, target_len=None):
    h_qy = qy * F.log_softmax(qy_logit, dim=-1) 
    h_qy_py = qy * F.log_softmax(py_logit, dim=-1)
    l = h_qy - h_qy_py

    # if mask is not None:
    #     assert target_len is not None, "`target_len` should be given if `mask` is not None."
    #     mask, effect_len = update_mask(mask, target_len, 0)
    #     l = l * mask
    #     l_norm = l.sum(dim=-1).div(effect_len)
    #     # l_norm = apply_mask(l, mask, norm_frame=effect_len)
    # else:
    batch = h_qy.shape[0]
    l_norm = l.sum(dim=-1).mean(dim=0)

    return l_norm


def h_qy(qy_logit, qy, mask=None, target_len=None):
    l = qy * F.log_softmax(qy_logit, dim=-1)
    # if mask is not None:
    #     assert target_len is not None, "`target_len` should be given if `mask` is not None."
    #     mask, effect_len = update_mask(mask, target_len, 0)
    #     l = l * mask
    #     l_norm = l.sum(dim=-1).div(effect_len)
    # else:
    batch = l.shape[0]
    l_norm = l.sum(dim=-1).mean(dim=0)

    return l_norm
        

def update_mask(mask, target_len, n):
    effect_len = target_len.sum()
    if n > 0:
        mask = mask[(..., ) + (None, ) * n]
    return mask, effect_len


def apply_mask(l, mask, norm_frame=None):
    l *= mask
    l = l.sum()
    if norm_frame is not None:
        l /= norm_frame
    return l


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.uint8)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
  
    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
  
    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        # mask = (1 - mask).type(torch.bool)
        mask = (1 - mask).type(torch.uint8)
        return mask.to(self.device)
  
    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v
  
    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
  
    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
  
        similarity_matrix = self.similarity_function(representations, representations)
  
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
  
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
  
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
  
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
  
        return loss / (2 * self.batch_size)
