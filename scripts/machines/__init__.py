import torch
import torch.nn as nn

from .BasicMachine import BasicMachine
from .MMaskedRASCGAN import MMaskedRASCGAN
from .BasicGAN import BasicGAN

__all__ = ['basic','basicgan','mmaskedgan','maskedganplus']

def basic(**kwargs):
	return BasicMachine(**kwargs)

def mmaskedgan(**kwargs):
	return MMaskedRASCGAN(pixelloss=nn.MSELoss, **kwargs)

def maskedganplus(**kwargs):
	return MMaskedRASCGAN(pixelloss=nn.MSELoss, **kwargs)

def basicgan(**kwargs):
	return BasicGAN( **kwargs)
