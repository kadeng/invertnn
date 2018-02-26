"""

This module provides a Framework and many implementations of invertible transformations
to enable density estimation, invertible generative models and more invertible ANN architectures.

See also orthogonal_transform and white_noise packages for further invertible transforms

:Author: Kai Londenberg, 2018

Papers:

Baird et. al: One-Step Neural Network Inversion with PDF Learning and Emulation
http://leemon.com/papers/2005bsi.pdf

Dinh et al: Density Estimation using RealNVP
https://arxiv.org/abs/1605.08803

THIS IS WORK IN PROGRESS - it's not yet fully documented and covered by tests.
Therefore, expect bugs, especially when it comes to
the computation of log abs determinant of the inverse transformation.
"""

import abc

import torch
import torch.autograd as autograd
import torch.distributions as dists
import torch.nn as nn
from torch.distributions import Transform, TransformedDistribution
from collections import deque
from typing import Deque, List, Dict, Tuple
import torch.nn.functional as F

__all__ = [ 'InvertibleModule',
            'InvertibleModuleTransform',
            'InvertibleVolumePreservingMixin',
            'InvertibleComponentwiseMixin',
            'InvertibleSigmoid',
            'InvertibleTanh',
            'InvertibleLeakyReLU',
            'InvertibleBairdActivation',
            'InvertibleSequential',
            'InferenceContext',
            'InvertiblePixelShuffle',
            'InvertibleConcatNoise',
            'ApproximatelyInvertibleUpsample',
            'InvertibleConcat',
            'InvertibleShuffle',
            'InvertibleBias',
            'InvertedModule',
            'InvertibleCouplingLayer',
            'module_inversion',
            'module_inv_jacobian_logabsdet'
            ]


class InvertibleModule(object,  metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def invert(self, output):
        """Calculate the inverted transformation of forward"""
        pass

    @abc.abstractmethod
    def inv_jacobian_logabsdet(self, output):
        """
        Log of the absolute valie of the derminant of the jacobian of
        the inverted transform at output

        returns: inverted output, log of absolute value of jacobian
        """
        pass

    def inverted_module(self):
        return InvertedModule(self)

class InvertedModule(InvertibleModule, nn.Module):

    def __init__(self, inverse):
        super().__init__()
        self.inverse = inverse

    def invert(self, output):
        return self.inverse.forward(output)

    def forward(self, input):
        return self.inverse.invert(input)

    def inv_jacobian_logabsdet(self, output):
        inp = self.inverse.forward(output)
        return self.inv_jacobian_logabsdet(inp)

    def inverted_module(self):
        return self.inverse

class InvertibleModuleTransform(Transform):
    '''
    Adapter class, capable of creating a pytorch.distributions.Transform
    out of any InvertibleModule
    '''

    def __init__(self, module : InvertibleModule, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.module = module
        self.bijective = True

    def _call(self, x):
        return self.module(x)

    def _inverse(self, y):
        return self.module.invert(y)

    def log_abs_det_jacobian(self, x, y):
        return self.module.inv_jacobian_logabsdet(y)

class InvertibleVolumePreservingMixin(object):

    def inv_jacobian_logabsdet(self, output):
        """
        Log of the absolute valie of the derminant of the jacobian of
        the inverted transform at output
        """
        return self.invert(output), torch.ones((output.shape[0], 1)).to(output.device)

class InvertibleComponentwiseMixin(object):

    def inv_jacobian_logabsdet(self, output):
        """
        Log of the absolute valie of the derminant of the jacobian of
        the inverted transform at output
        """
        ovar = autograd.Variable(output.detach(), requires_grad=True)
        inverse = self.invert(ovar)
        grad = autograd.grad([inverse], [ovar], [ torch.ones_like(inverse) ])
        jacobian_log_abs_det = torch.sum(grad[0].abs().log().view(output.shape[0], -1), dim=1)
        return inverse, jacobian_log_abs_det

def module_inversion(module, output):
    if isinstance(module, InvertibleModule):
        return module.invert(output)
    else:
        raise Exception("Module %s not invertible" % (module))

def module_inv_jacobian_logabsdet(module, output):
    if isinstance(module, InvertibleModule):
        return module.inv_jacobian_logabsdet(output)
    else:
        raise Exception("Module %s not invertible" % (module))

class InvertibleSigmoid(InvertibleComponentwiseMixin, InvertibleModule, nn.Sigmoid):

    def invert(self, output):
        ox = output * 2.0 - 1.0
        return torch.log(1+ox) - torch.log(1-ox)

class InvertibleTanh(InvertibleComponentwiseMixin, InvertibleModule, nn.Tanh):

    def invert(self, output):
        return 0.5 * (torch.log(1+output) - torch.log(1-output))

class InvertibleLeakyReLU(InvertibleComponentwiseMixin, InvertibleModule, nn.LeakyReLU):

    def invert(self, output):
        return torch.where(output>0.0, output, output.div(self.negative_slope))

class InvertibleLeakyReLU(InvertibleComponentwiseMixin, InvertibleModule, nn.LeakyReLU):

    def invert(self, output):
        return torch.where(output>0.0, output, output.div(self.negative_slope))

class InvertiblePReLU(InvertibleComponentwiseMixin, InvertibleModule, nn.PReLU):

    def __init__(self, num_parameters=1, init=0.25, eps=1e-8):
        super().__init__(num_parameters, init)
        self.eps = eps

    def forward(self, input):
        return F.prelu(input, self.weight)

    def invert(self, output):
        return torch.where(output>0.0, output, output.div(self.weight+self.eps))

class InvertibleELU(InvertibleComponentwiseMixin, InvertibleModule, nn.ELU):
    r"""Applies invertible, element-wise,
    :math:`f(x) = max(0,x) + min(0, alpha * (\exp(x) - 1))`

    Args:
        alpha: the alpha value for the ELU formulation. Default: 1.0

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, alpha=1):
        super().__init__(alpha, False)
        assert(alpha>1e-8)

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)

    def invert(self, output):
        return torch.where(output>=0.0, output, torch.log(output.div(self.alpha)+1.0))

class InvertibleSELU(InvertibleComponentwiseMixin, InvertibleModule, nn.SELU):
    r"""Applies invertible, element-wise,
    :math:`f(x) = scale * (\max(0,x) + \min(0, alpha * (\exp(x) - 1)))`,
    with ``alpha=1.6732632423543772848170429916717`` and
    ``scale=1.0507009873554804934193349852946``.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self):
        super().__init__(False)


    def forward(self, input):
        return F.selu(input, self.inplace)


    def invert(self, output):
        output = output.div(1.0507009873554804934193349852946)
        return torch.where(output>=0.0, output, torch.log(output.div(1.6732632423543772848170429916717)+1.0))

class InvertibleBias(InvertibleVolumePreservingMixin, InvertibleModule, nn.Module):

    def __init__(self, bias_tensor, spatial=False):
        super().__init__()
        if not spatial:
            self.bias = nn.Parameter(bias_tensor.detach().unsqueeze(0))
        else:
            self.bias = nn.Parameter(bias_tensor.detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

    def forward(self, input):
        return input.add(self.bias)

    def invert(self, output):
        return output.add(-self.bias)

class InvertibleBairdActivation(InvertibleComponentwiseMixin, InvertibleModule, nn.Module):
    ''' Invertible Activation Function, which is normalized in the range from -1 to +1
    in both directions, from Paper
    "One-Step Neural Network Inversion with PDF Learning and Emulation"
    '''

    def __init__(self, layer_shape):
        super().__init__()
        self.layer_shape = [1]+list(layer_shape)
        self.c = nn.Parameter(torch.zeros([1]+list(layer_shape)))

    def forward(self, input):
        assert(len(input.shape)==len(self.layer_shape))
        for i, s in enumerate(input.shape):
            assert(i == 0 or s==self.layer_shape[i])
        return self.baird_activation(input, self.c)

    def invert(self, output):
        assert(len(output.shape)==len(self.layer_shape))
        for i, s in enumerate(output.shape):
            assert(i==0 or s==self.layer_shape[i])
        return self.baird_activation(output, -self.c)

    @staticmethod
    def baird_activation(x, c):
        ta = 0.5+c*torch.exp(-x)-torch.exp(-2.0 * x)/2.0;
        tb = -0.5+c*torch.exp(x)+torch.exp(2*x)/2.0;
        return torch.where(c.gt(0.0), x+torch.log(ta+torch.sqrt(torch.exp(-2.0 * x)+ta * ta)) , x-torch.log(-tb + torch.sqrt(torch.exp(2.0 * x) + tb * tb)))


class InvertibleSequential(InvertibleModule, nn.Sequential):

    def forward(self, input):
        for name, module in self._modules.items():
            print("Calling %s with input of shape %r" % (name, input.shape))
            input = module(input)
        return input


    def invert(self, output):
        activation = output
        for module in reversed(self._modules.values()):
            activation = module_inversion(module, activation)
        return activation

    def inv_jacobian_logabsdet(self, output):
        activation = output
        current_jacobian_logabsdet = torch.zeros(1).to(output.device)
        for module in reversed(self._modules.values()):
            activation, inv_jacobian_logabsdet = module_inv_jacobian_logabsdet(module, activation)
            current_jacobian_logabsdet = current_jacobian_logabsdet.add(inv_jacobian_logabsdet)
        return activation, current_jacobian_logabsdet

def pixel_shuffle_inverse(input, downscale_factor):
    batch_size, in_channels, in_height, in_width = input.size()
    channels = in_channels * (downscale_factor ** 2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view( batch_size, in_channels, out_height, downscale_factor, out_height, downscale_factor)

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)

class InvertiblePixelShuffle(InvertibleVolumePreservingMixin, InvertibleModule, nn.PixelShuffle):

    def invert(self, output):
        pixel_shuffle_inverse(output, self.upscale_factor)

class InferenceContext(object):
    """
    Context manager, intended to collect (side) outputs
    of an invertible neural network during execution.

    Typical Usage:

    with InvertibleNetworkExecutionContext() as nc:
        for real_image, real_depth_image in data_loader:
            random_input = sampler.sample_input()
            random_side_input1 = sampler.sample_side_input1()
            nc.side_inputs['side_input1'] = random_side_input1

            generated_image = invertible_generator_net.forward(random_input)
            generated_depth_image = nc.side_outputs['depth_image']
            predicted_realness = adversarial_net(generated_image, generated_depth_image)

            nc.reset()
            nc.side_outputs['depth_image'] = real_depth_image

            reconstructed_random_input = invertible_generator_net.invert(real_image)
            reconstructed_side_inputs = invertible_generator_net.side_inputs['side_input1']

            invert_side_loss_dict = nc.losses
            forward_side_output_dict = nc.outputs

            nc.reset()

    """


    _context_stack : Deque = deque()
    @classmethod
    def current(cls):
        if len(cls._context_stack)>0:
            return cls._context_stack[-1]
        else:
            return None

    def __init__(self):
        super()
        self.reset()

    def reset(self):
        self.side_outputs = dict()
        self.side_inputs = dict()
        self.forward_losses = dict()
        self.invert_losses = dict()
        self.reconstructed_side_outputs = dict()
        self.reconstructed_side_inputs = dict()

    def __enter__(self):
        self._context_stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        self.reset()
        self._context_stack.pop()

    def put_forward_loss(self, name, loss):
        self.forward_losses[name] = loss

    def put_invert_loss(self, name, loss):
        self.invert_losses[name] = loss

    def put_side_input(self, name, input):
        self.side_inputs[name] = input

    def put_side_output(self, name, input):
        self.side_outputs[name] = input

    def put_reconstructed_side_input(self, name, input):
        self.reconstructed_side_inputs[name] = input

    def put_reconstructed_side_output(self, name, output):
        self.reconstructed_side_outputs[name] = output


class ApproximatelyInvertibleUpsample(InvertibleVolumePreservingMixin, InvertibleModule, nn.Module):

    def __init__(self, scale_factor=2, inversion_error_loss=True, name=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.inversion_error_loss = inversion_error_loss
        if inversion_error_loss and name is None:
            raise Exception("ApproximatelyInvertibleUpsample requires a name if inversion_error_loss is set to True")

    def forward(self, input):
        F.interpolate(input, None, self.scale_factor, 'nearest', False)

    def invert(self, output):
        if len(output.shape)==3:
            res = F.avg_pool1d(self.scale_factor)
        elif len(output.shape)==4:
            res = F.avg_pool2d(self.scale_factor)
        else:
            raise Exception("Invalid shape %r - has to be 1D or 2D data" % (output.shape))
        if self.inversion_error_loss and self.training:
            ic: InferenceContext = InferenceContext.current()
            if ic is not None:
                reconstruction = self.forward(res.detach())
                inversion_error = F.mse_loss(reconstruction, output, reduction='mean')
                ic.put_invert_loss(self.name, inversion_error)
        return res

class InvertibleInput(InvertibleVolumePreservingMixin, InvertibleModule, nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name

        self.input = None

    def forward(self, input=None):
        ic: InferenceContext = InferenceContext.current()
        if input is None:
            if ic is not None:
                input = ic.side_inputs[self.name]
        else:
            if ic is not None:
                ic.put_side_input(self.name, input)
        return input

    def invert(self, output):
        ic: InferenceContext = InferenceContext.current()
        if ic is not None:
            ic.put_reconstructed_side_input(self.name, output)
        return output

class InvertibleConcatNoise(InvertibleModule, nn.Module):

    def __init__(self, distribution : dists.Distribution, size : int, name=None):
        super().__init__()
        self.restored_input = None
        self.distribution = distribution
        self.size = size
        self.name = name

    def forward(self, input):
        sample = self.distribution.rsample([input.shape[0]]).type_as(input).to(input.device)
        sample.log_prob = self.distribution.log_prob(sample).sum()
        ic: InferenceContext = InferenceContext.current()
        if ic is not None:
            ic.put_side_input(self.name, sample)

        return torch.cat([sample, input], dim=1)

    def invert(self, output):
        sample, input = torch.split(output, split_size_or_sections=[self.size, output.shape[1]-self.size], dim=1)
        restored_sample = sample
        restored_sample.log_prob = self.distribution.log_prob(restored_sample).sum()

        ic: InferenceContext = InferenceContext.current()
        if ic is not None:
            ic.put_reconstructed_side_input(self.name, restored_sample)
        return input

    def inv_jacobian_logabsdet(self, output):
        ic: InferenceContext = InferenceContext.current()
        if ic is not None:
            input = self.invert(output)
            restored_log_prob = ic.reconstructed_side_inputs[self.name].log_prob
            return input, restored_log_prob
        else:
            raise Exception("Missing InferenceContext")

class InvertibleConcat(InvertibleVolumePreservingMixin, InvertibleModule, nn.Module):
    '''
    Allows to concat two inputs with known fixed sizes in a certain dimension
    on inversion, the input is splitted

    During forward pass, it is either possible to first set the second input on property "self.input_b"
    on this module and then treat it like requiring just a single argument, or to pass two inputs

    On inverse pass, the output is split along the specified dimension, and only the first part is returned, whereas the second part
    is stored in the "restored_input_b" property.
    '''

    def __init__(self, size_input : int,size_side_input : int, side_input_name : str, dim=1):
        super().__init__()
        self.size_a = size_input
        self.size_b = size_side_input
        self.dim = dim
        self.side_input_name = side_input_name

    def forward(self, input_a, input_b=None):
        if input_b is None:
            ic: InferenceContext = InferenceContext.current()
            assert(ic is not None)
            assert(self.side_input_name in ic.side_inputs)
            input_b = ic.side_inputs[self.side_input_name]

        assert(input_a.shape[self.dim]==self.size_a)
        assert(input_a.shape[self.dim]==self.size_b)
        return torch.cat([input_a, input_b], dim=self.dim)

    def invert(self, output):
        input_a, input_b = torch.split(output, split_size_or_sections=[self.size_a, output.shape[self.dim]-self.size_a], dim=self.dim)
        ic: InferenceContext = InferenceContext.current()
        if ic is not None:
            ic.put_reconstructed_side_input(self.side_input_name, input_b)
        return input_a

class InvertibleShuffle(InvertibleVolumePreservingMixin, InvertibleModule, nn.Module):

    def __init__(self, shuffling_order, dim=1):
        super().__init__()
        self.dim = dim
        shuffling_order = [ int(j) for j in shuffling_order]
        self.register_buffer('shuffle_indices', torch.LongTensor(shuffling_order))
        unshuffle_dict = { i : j for j, i in enumerate(shuffling_order) }
        self.register_buffer('unshuffle_indices', torch.LongTensor([ unshuffle_dict[j] for j in sorted(unshuffle_dict.keys())]))

    def forward(self, input):
        return torch.index_select(input, dim=self.dim, index=self.shuffle_indices)

    def invert(self, input):
        return torch.index_select(input, dim=self.dim, index=self.unshuffle_indices)

class InvertibleView(InvertibleVolumePreservingMixin, InvertibleModule, nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return input.view(self.output_shape)

    def invert(self, output):
        return output.view(self.input_shape)

class InvertibleCouplingLayer(InvertibleModule, nn.Module):
    ''' Coupling Layer as in DENSITY ESTIMATION USING REAL NVP: https://arxiv.org/pdf/1605.08803.pdf

    Note that this is INVERTED, i.e. we assume that in the forward pass we map from z to x, and
    it's inverse maps x to z.

    Therefore, forward = g; invert = f

    '''
    def __init__(self, k : int, d : int, s : nn.Module, t : nn.Module, odd_layer : bool = False):
        super().__init__()
        self.s = s
        self.t = t
        self.d = d
        self.k = k
        self.D = d + k
        self.odd_layer = odd_layer

    def forward(self, input):
        assert(input.shape[1]==self.D)
        if not self.odd_layer:
            unchanged = input[:,0:self.d]
            tmp = input[:, self.d:]
        else:
            unchanged = input[:,-self.d:]
            tmp = input[:, :-self.d]
        coupled = (tmp - self.t(unchanged) ) * torch.exp(self.s(unchanged).mul(-1.0))
        if self.odd_layer:
            result = torch.cat([unchanged, coupled], dim=1)
        else:
            result = torch.cat([coupled, unchanged], dim=1)
        return result

    def invert(self, output):
        if self.odd_layer:
            unchanged = output[:,0:self.d]
            tmp = output[:, self.d:]
        else:
            unchanged = output[:,-self.d:]
            tmp = output[:, :-self.d]
        coupled = tmp * torch.exp(self.s(unchanged)) + self.t(unchanged)
        if not self.odd_layer:
            result = torch.cat([unchanged, coupled], dim=1)
        else:
            result = torch.cat([coupled, unchanged], dim=1)
        return result

    def inv_jacobian_logabsdet(self, output):
        inverse = self.invert(output)
        if not self.odd_layer:
            tmp = output[:, self.d:]
        else:
            tmp = output[:, :-self.d]
        det_diag = self.s(tmp)
        jacobian_log_abs_det = torch.sum(det_diag.view(det_diag.shape[0], -1), dim=1)
        return inverse, jacobian_log_abs_det


class InvertibleForwardRangeCompression(InvertibleComponentwiseMixin, InvertibleModule):
    """Compress the value range to lie within a given minimum and maximum.
       Uses a scaled and shifted sigmoid function to achieve this
    """

    def __init__(self, output_min: float, output_max: float):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.scalar_tensor(output_min))
        self.scale = torch.nn.Parameter(torch.scalar_tensor(output_max - output_min))

    def forward(self, input):
        return torch.sigmoid(input) * self.scale + self.bias

    def invert(self, output):
        ox = ((output - self.bias) / self.scale) * 2.0 - 1.0
        return torch.log(1 + ox) - torch.log(1 - ox)


class InvertibleBackwardRangeCompression(InvertibleComponentwiseMixin, InvertibleModule):
    """Compress the value range to lie within a given minimum and maximum.
       Uses a scaled and shifted sigmoid function to achieve this
    """

    def __init__(self, input_min: float, input_max: float):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.scalar_tensor(input_min))
        self.scale = torch.nn.Parameter(torch.scalar_tensor(input_max - input_min))

    def invert(self, output):
        return torch.sigmoid(output) * self.scale + self.bias

    def forward(self, input):
        ox = ((input - self.bias) / self.scale) * 2.0 - 1.0
        return torch.log(1 + ox) - torch.log(1 - ox)

class RegularizedRangeClamp(InvertibleVolumePreservingMixin, InvertibleModule):
    """
    Checks the value range and penalizes ( i.e. introduces a loss function )
    if it lies outside a target range. Can also clamp values to a certain range
    """
    def __init__(self, min: float, max: float, name : str, l1_weight=0.05, l2_weight=0.95, clamp_values=False, on_forward=True, on_invert=True):
        super().__init__()
        self.min = min
        self.max = max
        self.name = name
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.clamp_values = clamp_values
        self.on_forward = on_forward
        self.on_invert = on_invert

    def invert(self, output : torch.Tensor):
        if self.on_invert:
            ic : InferenceContext = InferenceContext.current()
            if (self.training and ic is not None) or self.clamp_values:
                clamped_output = output.clamp(self.min, self.max)
            if (self.training and ic is not None):
                diff = output - clamped_output
                loss = self.l2_weight * diff.pow(2).mean() + self.l1_weight * diff.abs().mean()
                ic.put_invert_loss(self.name, loss)
            if self.clamp_values:
                return clamped_output
        return output

    def forward(self, input):
        if self.on_forward:
            ic: InferenceContext = InferenceContext.current()
            if (self.training and ic is not None) or self.clamp_values:
                clamped_input = input.clamp(self.min, self.max)
            if self.training and ic is not None:
                diff = input - clamped_input
                loss = self.l2_weight * diff.pow(2).mean() + self.l1_weight * diff.abs().mean()
                ic.put_forward_loss(self.name, loss)
            if self.clamp_values:
                return clamped_input
        return input

