

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math

from scripts.utils.model_init import *
from scripts.models.rasc import *
from scripts.models.urasc import *
from scripts.models.unet import *

# radhn with unet 

__all__ = [
'unet','s2asc','s2ad',
'senet','cbam',
'urasc','maskedurasc','maskedurascgan','uno',
'naiveuno','naivemultimaskedurasc']


def naiveuno(**kwargs):
    # unet without mask.
    model = NaiveUnetGenerator(3,3)
    model.apply(weights_init_kaiming)
    return model
    
def naivemultimaskedurasc(**kwargs):
    # unet without mask.
    model = NaiveUnetGenerator(3,3,attention_model=MaskedURASC,basicblock=NaiveMultiURASC,learning_model=ModifyBasicLearningBlock)
    model.apply(weights_init_kaiming)
    return model

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

def maskedurascgan(**kwargs):
    # learning without mask based on RASCV2.
    model = UnetGenerator(3,3,is_attention_layer=True,attention_model=MaskedURASC,basicblock=MaskedMinimalUnetV2,final_layer=True, outputmask=True)
    model.apply(weights_init_kaiming)
    return model

def s2ad(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=RASC,basicblock=MinimalUnetV2)
    model.apply(weights_init_kaiming)
    return model

def s2asc(**kwargs):
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

def senet(**kwargs):
    model = UnetGenerator(4,3,is_attention_layer=True,attention_model=SENet)
    model.apply(weights_init_kaiming)
    return model


