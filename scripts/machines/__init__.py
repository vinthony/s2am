
from .BasicMachine import BasicMachine
from .MaskedRASC import MaskedRASC
from .SemiRASC import SemiRASC


__all__ = ['basic','maskedpixel','maskedbce','gan','semirasc']

def basic(**kwargs):
	return BasicMachine(**kwargs)

def semi(**kwargs):
	return SemiRASC(**kwargs)

def maskedpixel(**kwargs):
	return MaskedRASC(pixelloss=True,**kwargs)

def maskedbce(**kwargs):
	return MaskedRASC(pixelloss=False,**kwargs)


def ganbased(**kwargs):
	return GANRASC(**kwargs)
