

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math

from scripts.utils.model_init import *
from scripts.models.vgg import Vgg16
from scripts.models.rasc import *
from scripts.models.urasc import *
from scripts.models.unet import oUnetGenerator,UnetGenerator,MaskedMinimalUnetV2,MinimalUnetV2

# radhn with unet 

__all__ = ['unet','rascv1','rascv2','senet','cbam','rascpc','urasc','maskedurasc','uno','rascv2_adain']


def uno(**kwargs):
    # unet without mask.
    model = UnetGenerator(3,3,is_attention_layer=True,attention_model=UNO)
    model.apply(weights_init_kaiming)
    return model

def urasc(**kwargs):
    # learning without mask based on RASCV2.
    model = UnetGenerator(3,3,is_attention_layer=True,attention_model=URASC,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model

def maskedurasc(**kwargs):
    # learning without mask based on RASCV2.
    model = UnetGenerator(3,3,is_attention_layer=True,attention_model=MaskedURASC,basicblock=MaskedMinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model

def rascv2(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=RASC,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model

def rascv2_adain(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=RASCXADAIN,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model

def rascv1(**kwargs):
    # Splicing Region: features -> GlobalAttentionModule -> CNN -> * SplicingSmoothMask ->
    # mixed Region: faetures -> GlobalAttentionModule ----------⬆ 
    # Background Region: faetures -> GlobalAttentionModule -> * ReversedSmoothMask ->
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=RASC)
    model.apply(weights_init_kaiming)
    return model

def unet(**kwargs):
    # just original unet
    model = UnetGenerator(4,3)
    model.apply(weights_init_kaiming)
    return model


def cbam(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=CBAMConnect)
    model.apply(weights_init_kaiming)
    return model

def rascpc(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=RASC_PC)
    model.apply(weights_init_kaiming)
    return model

def senet(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=SENet)
    model.apply(weights_init_kaiming)
    return model


