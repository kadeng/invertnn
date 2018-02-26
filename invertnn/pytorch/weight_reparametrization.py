"""
Dynamic Weight Reparametrization transformations on Module parameters

Usage scenarios include
 - Centered Weight Normalization - a reportedly very useful technique to improve all neural network training
 - Equalization of Learning Rates - a technique used by "Progressive Growing of GANs for Improved Quality, Stability,
 and Variation"
 - Enforcing strictly positive weights via exp transformations - which might be a neccessity to train invertible ANNs
 - ...

Also imports the already existing "weight_normalization", so all of these methods are available via a single module.

:Author: Kai Londenberg, 2018
"""
import math

import torch.nn.init as nninit
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm, remove_weight_norm

__all__ = ['weight_reparametrization', 'remove_weight_reparametrization', 'weight_norm', 'remove_weight_norm',
           'centered_weight_norm', 'remove_centered_weight_norm', 'weight_rescaling', 'weight_rescaling_he_init']


def _exp(input):
    return input.exp()


def _log(input):
    return input.log()


def weight_reparametrization(module, name='weight', transform=_exp):
    r"""Applies componentwise weight reparametrization to a parameter in the given module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        transform (Callable, optional): Function, transforming an input tensor componentwise, returning a tensor of
        same shape as input

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_reparametrization(nn.Linear(20, 40), name='weight')

    """
    WeightReparametrization.apply(module, name, transform=transform)
    return module


def weight_rescaling(module, param='weight', scale_factor=1.0):
    '''
    Reparametrization which rescales a weight parameter by a given constant. Used by weight_rescaling_he_init below
    '''

    def rescale(param):
        return param.mul(scale_factor)
    assert(scale_factor>=0.0001)
    assert(scale_factor<=1000000.0)
    return weight_reparametrization(module, param, rescale)

def weight_rescaling_he_init(module, param='weight', nonlinearity='leaky_relu', nonlinearity_param=None, mode='fan_in',
                             skip_init=False):
    '''
    Applies the equalized learning rate initialization & weight rescaling from "Progressive Growing of GANs for
    Improved Quality, Stability, and Variation"
    Based on code from torch.nn.init.kaiming_normal
    :param module: Module whose parameter we're wrapping
    :param param: parameter name to reparametrize (defaults to 'weight')
    :param gain_nonlinearity: nonlinearity: the nonlinear function (`nn.functional` name) used after this layer
    :param nonlinearity_param: the optional parameter of the nonlinearity (None by default, resulting in 0.01 if
    nonlinearity is leaky_relu)
    :param mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
    :param skip_init: Whether to skip initialization with random noise of zero mean and unit variance. Defaults to
    False (i.e. init is performed)
    :return:
    '''
    tensor = module._parameters[param]
    if not skip_init:
        tensor.normal_(0.0, 1.0)
    fan = nninit._calculate_correct_fan(tensor, mode)
    gain = nninit.calculate_gain(nonlinearity, nonlinearity_param)
    std = gain / math.sqrt(fan)
    return weight_rescaling(module, param, std)


# Code for weight reparametrization is based on torch.nn.utils.weight_norm
class WeightReparametrization(object):

    def __init__(self, name, transform=_exp):
        self.name = name
        self.transform = transform

    def compute_weight(self, module):
        pre = getattr(module, self.name + '_pre')
        return self.transform(pre)

    @staticmethod
    def apply(module, name, transform=_exp):
        fn = WeightReparametrization(name, transform=transform)

        weight = module._parameters[name]

        # remove w from parameter list
        del module._parameters[name]

        # we reuse the weight Parameter for our purpose
        pre_weight = Parameter(transform(weight).data)
        module.register_parameter(name + '_pre', pre_weight)

        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_pre']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def remove_weight_reparametrization(module, name='weight'):
    r"""Removes the componentwise weight reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightReparametrization) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}".format(name, module))


def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class CenteredWeightNorm(object):
    def __init__(self, name, output_dim):
        self.name = name
        self.output_dim = output_dim

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        d = v.shape[self.output_dim]
        centered_v = v - v.sum(dim=self.output_dim, keepdim=True).div(float(d))
        centered_norm = _norm(centered_v, self.output_dim).add(1e-8)
        return centered_v * (g / centered_norm)

    @staticmethod
    def apply(module, name, dim, init):
        fn = CenteredWeightNorm(name, dim)
        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        g = Parameter(_norm(weight, dim).data)
        v = Parameter(weight.data)
        d = v.shape[dim]
        if init:
            # Initialize, so we start with zero-mean and unit norm
            g.data.fill_(1.0)
            v.data.sub_(v.sum(dim=dim, keepdim=True).div(float(d)))
            v.data.div_(_norm(v, dim).add(1e-8))

        module.register_parameter(name + '_g', g)
        module.register_parameter(name + '_v', v)
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def centered_weight_norm(module, name='weight', dim=0, init=True):
    r"""Applies centered weight normalization to a parameter in the given module.

    See Huang et. al: Centered Weight Normalization in Accelerating Training of Deep Neural Networks
    DOI: 10.1109/ICCV.2017.305

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    """
    CenteredWeightNorm.apply(module, name, dim, init=init)
    return module


def remove_centered_weight_norm(module, name='weight'):
    r"""Removes the centered weight normalization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, CenteredWeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("centered_weight_norm of '{}' not found in {}".format(name, module))
