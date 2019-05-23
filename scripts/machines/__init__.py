import torch
import torch.nn as nn

from .BasicMachine import BasicMachine
from .MaskedRASC import MaskedRASC
from .MaskedRASCGAN import MaskedRASCGAN
from .MaskedRASCGAAN import MaskedRASCGAAN
from .MaskedSplicedGAN import MaskedSplicedGAN
from .MultiMaskedRASC import MultiMaskedRASC
from .MMaskedRASCGAN import MMaskedRASCGAN
from .BasicGAN import BasicGAN


__all__ = ['basic','multimaskedpixel','maskedpixel','maskedbce',
'gan','semirasc','splicing','maskedgan','mmaskedgan'
'maskedganplus','maskedpixell1']

def basic(**kwargs):
	return BasicMachine(**kwargs)

def maskedpixel(**kwargs):
	return MaskedRASC(pixelloss=nn.MSELoss,**kwargs)

def multimaskedpixel(**kwargs):
	return MultiMaskedRASC(pixelloss=nn.MSELoss,**kwargs)

def maskedpixell1(**kwargs):
	return MaskedRASC(pixelloss=nn.L1Loss,**kwargs)

def maskedbce(**kwargs):
	return MaskedRASC(pixelloss=nn.BCELoss,**kwargs)

def maskedgan(**kwargs):
	return MaskedRASCGAN(pixelloss=nn.MSELoss, **kwargs)

def mmaskedgan(**kwargs):
	return MMaskedRASCGAN(pixelloss=nn.MSELoss, **kwargs)

def splicedgan(**kwargs):
	return MaskedSplicedGAN(pixelloss=nn.MSELoss, **kwargs)

def maskedganplus(**kwargs):
	return MaskedRASCGAAN(pixelloss=nn.MSELoss, **kwargs)


def basicgan(**kwargs):
	return BasicGAN( **kwargs)
