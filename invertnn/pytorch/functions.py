"""
Various PyTorch utility functions which did not fit elsewhere

:Author Kai Londenberg, 2018
"""


import torch
import math
from numbers import Number
import PIL
import numpy as np
import io
import json

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def covariance(x):
    x_mean = x.mean(dim=0, keepdim=True) # Mean per channel / sample
    x_centered = (x-x_mean)  # Normalize
    x_cov = torch.matmul(x_centered.t(), x_centered).div(x_centered.shape[0]) # Calc correlation
    return x_cov

def save_model(net, savedir, name, variant='last', metadata=dict()):
    torch.save(net, savedir + "/" + name + "." + variant + ".model.pickle")
    torch.save(net.state_dict(), savedir + "/" + name + "." + variant + ".state.pickle")
    with io.open(savedir + "/" + name + "." + variant + ".metadata.json", "w", encoding='utf-8') as fh:
        json.dump(metadata, fh)


def load_model(savedir, name, variant):
    return torch.load(savedir + "/" + name + "." + variant + ".model.pickle")

def save_image(path, image_tensor):
    result = PIL.Image.fromarray((image_tensor.cpu().detach().numpy().transpose([1,2,0]) * 255).astype(np.uint8))
    result.save(path)


def check_numerics(tensor):
    tsum = tensor.sum().item()
    if not np.isfinite(tsum):
        raise RuntimeError("Tensor not finite")