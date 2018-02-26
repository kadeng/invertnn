
import torch
import torch.nn as nn
import torch.distributions
import abc
import numpy as np

class Identity(nn.Module):
    ''' Dummy Module that just passes through it's input'''

    def __init__(self):
        super().__init__()

    def forward(self, arg):
        return arg


class ToDeviceWrapper(nn.Module):

    def __init__(self, wrapped_module, device=None):
        super().__init__()
        self.device = device
        if device is not None:
            wrapped_module.to(device)
        self.wrapped = wrapped_module

    def _apply(self, fn):
        fname : str = str(fn)
        if fname.startswith("<function Module.to.<locals>.convert"):
            if self.device is None:
                super()._apply(fn)
            else:
                print("Ignoring movement of module wrapped into ToDevice module")

    def forward(self, input):
        if self.device is not None:
            return self.wrapped.forward(input.to(self.device))
        else:
            return self.wrapped.forward(input)

class View(nn.Module):

    def __init__(self, *shape):
        '''
        Simple Module to reshape the output into a new shape
        :param shape: New shape
        '''
        super().__init__()
        self.shape = shape

    def forward(self, input):
        '''
        Apply the reshaping
        :param input: Input to reshape
        :return: reshaped input
        '''
        return input.view(input.shape[0], *self.shape)

class Reshape(nn.Module):

    def __init__(self, *shape):
        '''
        Simple Module to reshape the output into a new shape
        :param shape: New shape. May contain single-element lists, in which case the corresponding element from the input shape will be used
        '''
        super().__init__()
        self.shape = shape

    def forward(self, input):
        '''
        Apply the reshaping
        :param input: Input to reshape
        :return: reshaped input
        '''
        target_shape = [ input.shape[k[0]] if type(k)==list else k for k in self.shape ]
        return input.view(input.shape[0], *target_shape)

class CustomParameterInitialization(object,  metaclass=abc.ABCMeta):
    '''
    Abstract Base class / Interface to mark a layer as having
    a customized initialization method.
    '''

    @abc.abstractmethod
    def init_params(self):
        """Perform custom parameter initialization"""
        pass


class LinearCustomInit(CustomParameterInitialization, nn.Linear):

    def init_params(self):
        pass


class Conv2DCustomInit(CustomParameterInitialization, nn.Conv2d):

    def init_params(self):
        pass

class ScalingInitializer(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def calc_scale(self, shape):
        pass

    @abc.abstractmethod
    def initialize(self, tensor, do_scale=True):
        pass

    def __call__(self, tensor, do_scale=True):
        self.initialize(tensor)

class HeInitializer(ScalingInitializer):

    def __init__(self, base_distribution : torch.distributions.Distribution , gain=1.0):
        super().__init__()
        if gain == 'relu':
            gain = np.sqrt(2)
        self.base_distribution = base_distribution
        self.gain = gain

    def calc_fan_in(self, shape):
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])
        else:
            raise RuntimeError(
                "This initializer only works with shapes of length >= 2")
        return fan_in

    def calc_scale(self, shape):
        fan_in = self.calc_fan_in(shape)
        std = self.gain * np.sqrt(1.0 / fan_in)
        return std

    def sample(self, shape, do_scale=True):
        if do_scale:
            return self.base_distribution.sample(shape)*self.calc_scale(shape)
        else:
            return self.base_distribution.sample(shape)

    def initialize(self, tensor, do_scale=True):
        tensor.data.copy_(self.sample(tensor.shape, do_scale))



class CombinedLayer(CustomParameterInitialization, nn.Module):

    def __init__(self, interaction, nonlinearity=None, initializer=None, normalizer1=None, normalizer2=None, dropout=None):
        super().__init__()
        self.interaction = interaction
        self.nonlinearity = nonlinearity
        self.initializer = initializer
        self.normalizer = normalizer
        self.dropout = dropout
        self.init_params()

    def init_params(self):
        if self.interaction is not None and self.initializer is not None:
            self.initializer(self.interaction.weight)


