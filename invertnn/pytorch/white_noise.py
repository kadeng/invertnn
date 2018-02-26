"""
This module implements a Whitening / De-Whitening Transform,
intended to be used in conjunction with other Invertible Transforms / Invertible Neural Networks.

This module may be used to add or remove "White Noise" in an Invertible ANN, in order to add
or remove Degrees of Freedom in a certain Layer via InvertibleConcatNoise operation.

The (De-)Whitening allows to decorrelate the observed noise efficiently in the inverse pass,
while making sure that generated independent noise obtains a suitable Covariance Structure in
the forward pass.

:Author: Kai Londenberg, 2018
"""

import torch
import torch.nn as nn
import torch.distributions as dists
from torch.distributions.multivariate_normal import MultivariateNormal

from invertnn.pytorch.orthogonal_transform import OrthogonalTransform, DiagonalLinearTransform
from invertnn.pytorch.invertible_transforms import InvertibleModule, InvertibleModuleTransform, InvertibleSequential, InvertibleBias
from collections import deque, OrderedDict
from typing import Callable, Deque
from torch.optim import Adam, SGD, Rprop, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logging
import numpy as np


class AdaptiveInverseWhiteningTransformer(InvertibleModule, nn.Module):

    def __init__(self, n, observation_memory_batches=5, adapt_method='zca', auto_adapt_every : int=0, *adapt_args, **adapt_kwargs):
        '''
        Create AdaptiveInverseWhiteningTransformer, which can be used to parametrically transform
        Independent Random Noise to Correlated Random Noise, and (on the inverse pass) to whiten observation noise.

        See corresponding Unit Test (tests/test_adaptive_inverse_whitening) for usage examples.

        :param n: Dimensionality
        :param observation_memory_batches: How many batches of observed noise to store in memory for decorrelation
        :param adapt_method: Can be 'zca', 'pca' or 'none'. It is recommended to use 'zca', unless gradient based optimization is also used. Use 'pca' or 'none'
                             in these cases.
        :param auto_adapt_every:  Adapt transformation every *auto_adapt_every*-steps
        :param adapt_args: Optional arguments to Adam optimizer
        :param adapt_kwargs: Optional keyword arguments to Adam optimizer
        '''
        super().__init__()
        self.observation_memory = deque(maxlen=observation_memory_batches)
        self.adapt_method = adapt_method
        self.auto_adapt_every = auto_adapt_every
        self.n_observations = 0
        self.n = n
        self.adapt_kwargs = adapt_kwargs
        self.log = logging.getLogger("WhiteNoise")
        self.log.setLevel(logging.DEBUG)
        if adapt_method=='zca':
            U = OrthogonalTransform(n, bias=False)
            Ut = U.inverted_module()
            self.delegate = InvertibleSequential(
                OrderedDict(
                    [
                     ('U', U),
                     ('S', DiagonalLinearTransform(n, bias=False)),
                     ('V', Ut),
                     ('Bias', InvertibleBias(torch.zeros(n)))]
            ))
            self.delegate.Bias.bias.data.copy_(torch.zeros((n))+torch.randn((n))*0.01)
        else:
            V = OrthogonalTransform(n, bias=True)

            self.delegate = InvertibleSequential(
                OrderedDict(
                    [('S', DiagonalLinearTransform(n, bias=False)),
                     ('V', V)]
            ))
            #self.delegate.V.bias.data.copy_(torch.zeros((n))+torch.randn((n))*0.01)
        #self.delegate.S.weight.data.copy_(torch.ones((n))+torch.randn((n))*0.01)
        self.adapt_args = adapt_args
        self.adapt_kwargs = adapt_kwargs
        self.optimizer = None
        self.scheduler = None
        self.loss_history = deque(maxlen=1000)

    def forward(self, input):
        return self.delegate.forward(input)

    def inv_jacobian_logabsdet(self, output):
        return self.delegate.inv_jacobian_logabsdet(output)

    def invert(self, output : torch.Tensor):
        self.n_observations += 1
        self.observe(output)

        if self.auto_adapt_every>0 and self.adapt_method is not None and (self.n_observations % self.auto_adapt_every)==1:
            self.auto_adapt()
        inv = self.delegate.invert(output)
        return inv

    def auto_adapt(self):
        if self.adapt_method== 'pca':
            self.adapt_pca()
        elif self.adapt_method== 'zca':

            self.adapt_zca()
        elif self.adapt_method== 'none' or self.adapt_method is None:
            pass
        else:
            raise Exception("Uknown adaptive whitening method '%s'" % (self.adapt_method))

    def transformed_distribution(self, base_distribution : dists.Distribution, cache_size=0):
        '''
        Transform a distribution using the forward (De-Whitening) Transformation.
        :param base_distribution: Base Distribution, for example torch.distributions.Normal
        :param cache_size: Size of the cache, defaults to 0
        :return: torch.distributions.TransformedDistribution which can be used to sample from the De-Whitened distribution.
        '''
        return dists.TransformedDistribution(base_distribution, transforms=[InvertibleModuleTransform(self, cache_size)])

    def observe(self, x : torch.Tensor):
        self.log.debug("Observing Noise")
        self.observation_memory.append(x.detach())

    def calc_observation_stats(self, observations):
        if len(self.observation_memory)<1:
            return None,None
        x = torch.cat(tuple(observations), dim=0)
        if len(x.shape)>2:
            x = x.transpose(1, -1) # Put channel dimension last
            x = x.view(-1, x.shape[-1]) # Make it an N x C matrix
        x_mean = x.mean(dim=0, keepdim=True) # Mean per channel / sample
        x_centered = (x-x_mean)  # Normalize
        #check_cov = torch.from_numpy(np.cov(x_centered.detach().cpu().numpy(), rowvar=False))
        x_cov = torch.matmul(x_centered.t(), x_centered).div(x_centered.shape[0]) # Calc correlation
        return x_mean, x_cov

    def adapt_zca(self):
        assert(self.adapt_method=='zca')
        # https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
        # http://joelouismarino.github.io/blog_posts/blog_whitening.html
        # http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf ( Effect of ZCA on images )
        with torch.no_grad():
            mean, cov = self.calc_observation_stats(self.observation_memory)
            if mean is None:
                return
            u, s, v = torch.svd(cov)
            self.delegate.U.W = u
            self.delegate.S.weight.data.copy_(torch.sqrt(s + 1e-5))
            #self.delegate.V.W = u.t()
            self.delegate.Bias.bias.data.copy_(mean)

    def adapt_pca(self):
        assert(self.adapt_method!='zca')
        # https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
        # http://joelouismarino.github.io/blog_posts/blog_whitening.html
        with torch.no_grad():
            mean, cov = self.calc_observation_stats(self.observation_memory)
            if mean is None:
                return
            u, s, v = torch.svd(cov)
            self.delegate.V.W = u.t()
            self.delegate.V.bias.data.copy_(mean)
            self.delegate.S.weight.data.copy_(torch.sqrt(s + 1e-5))


    def create_optimizer(self):
        self.optimizer = Adam(self.parameters(), amsgrad=True, lr=0.1, betas=[0.5,0.9], *self.adapt_args, **self.adapt_kwargs) #Adam(self.parameters(), *adapt_args, **adapt_kwargs)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True)

    def adapt_step(self, weight_corr=0.5, weight_diag=0.5, norm=2.0, update=True):
        if self.adapt_method=='zca':
            self.log.warning("Using Gradient descent based optimization with ZCA Parametrization leads to irregular error surfaces and does not work")
            self.log.warning("Either use 'pca' parametrization, or use adapt_zca() method only")
        assert(self.adapt_method!='zca')
        if self.optimizer is None:
            self.create_optimizer()
        # Warning, this does not work well, probably for numerical reasons..
        mean, cov = self.calc_observation_stats([self.delegate.invert(x) for x in self.observation_memory])
        cov_diff = (cov-torch.eye(self.n)).triu()
        cov_diff_flat = cov_diff.view(-1)
        loss = cov_diff_flat.pow(norm).mean() * weight_corr + torch.diag(cov_diff).pow(norm).mean() * weight_diag
        self.loss_history.append(loss.pow(1.0/norm).item())

        loss.backward()
        if self.delegate.S.weight.grad.norm(2).item()>float(self.n):
            #self.log.debug("Clipping S Grad from %.5f to %.5f" % (self.delegate.S.weight.grad.norm(2).item(), self.n))
            self.delegate.S.weight.grad.data.mul_(self.n/self.delegate.S.weight.grad.norm(2))
        if update:
            self.scheduler.step(loss)
            self.optimizer.step()
        return loss.pow(1.0/norm).item()

def calc_cov(x):
    if len(x.shape)>2:
            x = x.transpose(1, -1) # Put channel dimension last
            x = x.view(-1, x.shape[-1]) # Make it an N x C matrix
    x_mean = x.mean(dim=0, keepdim=True) # Mean per channel / sample
    x_centered = (x-x_mean)  # Normalize
    x_norm = x / (x.std()+1e-8)
    x_cov = torch.matmul(x_centered.t(), x_centered).div(x_centered.shape[0]) # Calc correlation
    x_corr = torch.matmul(x_norm.t(), x_norm).div(x_norm.shape[0]) # Calc correlation
    return x_mean, x_cov, x_corr

