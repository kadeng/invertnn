"""
Implementation of Linear and Convolutional Operators
with Spectral Normalization based on the power iteration method

See Myato et al: Spectral Normalization for Generative Adversarial Networks
https://arxiv.org/abs/1802.05957

Author: Kai Londenberg, 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNormedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, itercount=1):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())
        self.itercount = itercount

    def weight_svnorm(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        norm, _u, _v = max_singular_value_fully_differentiable(w_mat, self.u, Ip=self.itercount)
        self._buffers['u'] = _u.detach()
        return self.weight / norm

    def forward(self, input):
        return F.linear(input, self.weight_svnorm(), self.bias)


class SpectralNormedConv2D(nn.Conv2d):

    def __init__(self, *args, itercount=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.itercount = itercount
        self.register_buffer('u', torch.Tensor(1, self.weight.shape[0]).normal_())

    def weight_svnorm(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        norm, _u, _v = max_singular_value_fully_differentiable(w_mat, self.u, Ip=self.itercount)
        self._buffers['u'] = _u.detach()
        return self.weight / norm

    def forward(self, input):
        return F.conv2d(input, self.weight_svnorm(), self.bias, self.stride, self.padding, self.dilation, self.groups)


# Based on code from https://github.com/pfnet-research/sngan_projection
def max_singular_value_fully_differentiable(W: torch.FloatTensor, u: torch.FloatTensor = None, Ip: int = 1, eps=1e-12):
    """
    Apply power iteration for the weight parameter (fully differentiable version)
    """
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")

    if u is None:
        u = W.new(size=(1, W.shape[0])).normal_()
    _u = u
    for _ in range(Ip):
        tmpv = torch.mm(_u, W)
        _v = tmpv / (torch.norm(tmpv, 2) + eps)
        tmpu = torch.mm(_v, torch.t(W))
        _u = tmpu / (torch.norm(tmpu, 2) + eps)
    _u = torch.mm(_v, torch.t(W))
    norm = _u.norm(2)
    _u = _u / norm
    return norm, _u, _v
