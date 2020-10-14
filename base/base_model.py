import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class BaseGMVAE(BaseModel):
    def __init__(self, input_size, latent_dim, n_component=10, pow_var_exp=0, 
                 is_featExtract=False):
        """
        Basic VAEs with a GMM as the prior distribution
        """
        super(BaseGMVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.n_component = n_component
        self.pow_var_exp = pow_var_exp
        self.is_featExtract = is_featExtract
        self._build_mu_lookup()
        self._build_logvar_lookup()

    def encode(self, x):
        """
        Implementation should include the outputs of 
        1. self._infer_latent()
        2. self._infer_class()
        """
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def _infer_latent(self, mu, logvar):
        if self.is_featExtract:
            """
            Infer mu as the feature vector
            """
            return mu, logvar, mu

        mu, logvar, z = sampling_gaussian(mu, logvar)
        return mu, logvar, z

    def _build_mu_lookup(self):
        """Initialize Gaussian means

        Follow Xavier initialization as in the paper (https://openreview.net/pdf?id=rygkk305YQ).
        This can also be done using a GMM in the latent space trained with vanilla autoencoders,
        as in https://arxiv.org/abs/1611.05148.
        """
        mu_lookup = nn.Embedding(self.n_component, self.latent_dim)
        # nn.init.xavier_uniform_(mu_lookup.weight)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=1)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, logvar_trainable=True):
        """Initialize Gaussian variances

        Follow Table 7 in the paper (https://openreview.net/pdf?id=rygkk305YQ).
        """
        logvar_lookup = nn.Embedding(self.n_component, self.latent_dim)
        init_sigma = np.exp(self.pow_var_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup
        self.logvar_bound = np.log(np.exp(-4) ** 2)  # lower bound of log variance for numerical stability

    def _bound_logvar_lookup(self):
        self.logvar_lookup.weight.data[torch.le(self.logvar_lookup.weight, self.logvar_bound)] = self.logvar_bound

    def _infer_class(self, q_z):
        logLogit_qy_z, qy_z = approx_qy_z(q_z, self.mu_lookup, self.logvar_lookup, n_component=self.n_component)
        val, y = torch.max(qy_z, dim=1)
        return logLogit_qy_z, qy_z, y

    def forward(self, x):
        raise NotImplementedError
        # mu, logvar, z, q_y, ind = self._encode(x)
        # x_predict = x_self._decode(z)
        # return [mu, logvar, z], [q_y, ind], x_predict


def sampling_gaussian(mu, logvar):
    sigma = torch.sqrt(torch.exp(logvar))
    eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size())
    eps = eps.to(sigma.device)
    z = mu + sigma * eps  # reparameterization trick
    return mu, logvar, z


def approx_qy_z(z, mu_lookup, logvar_lookup, n_component):
    """

    Approximating q(y|z) with p(y|z), the probability of z being assigned to class y.
    q(y|z) ~= p(y|z) = p(z|y) p(y) / p(z)

    :param z: Latent variables sampled from approximated posterior q(z|x)
    :param mu_lookup: i-th row corresponds to a mean vector of p(z|y = i) which is a Gaussian
    :param logvar_lookup: i-th row corresponds to a logvar vector of p(z|y = i) which is a Gaussian
    :param n_component: Number of components of the GMM prior
    """
    def log_gauss_lh(z, mu, logvar):
        """
        Calculate p(z|y), the likelihood of z w.r.t. a Gaussian component
        parameterized with mu and logvar.
        """
        llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
        llh = torch.sum(llh, dim=1)  # sum over dimensions
        return llh

    logLogit_qy_z = torch.zeros(z.shape[0], n_component)  # log-logit of q(y|z)
    for k_i in torch.arange(0, n_component):
        mu_k, logvar_k = mu_lookup(k_i), logvar_lookup(k_i)
        logLogit_qy_z[:, k_i] = log_gauss_lh(z, mu_k, logvar_k) + np.log(1 / n_component)

    qy_z = torch.nn.functional.softmax(logLogit_qy_z, dim=1)
    return logLogit_qy_z, qy_z
