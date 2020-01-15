

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scripts.utils.model_init import *
from scripts.models.blocks import *

class UNO(nn.Module):
    def __init__(self,channel):
        super(UNO, self).__init__()

    def forward(self,feature,_m):
        return feature 

class SENet(nn.Module):
    """docstring for SENet"""
    def __init__(self,channel,type_of_connection=None):
        super(SENet, self).__init__()
        self.attention = SEBlock(channel,16)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))

        result = self.attention(feature) 
        
        return result

class CBAMConnect(nn.Module):
    def __init__(self,channel):
        super(CBAMConnect, self).__init__()
        self.attention = CBAM(channel)

    def forward(self,feature,mask):
        results = self.attention(feature)
        return results

class RASC(nn.Module):
    def __init__(self,channel,type_of_connection=BasicLearningBlock):
        super(RASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.gaussianMask = GaussianSmoothing(1,5,1)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        if w != mw:
            mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        # here we add gaussin filter to mask and reverse_mask for better harimoization of edges.

        mask = self.gaussianMask(F.pad(mask,(2,2,2,2),mode='reflect'))
        reverse_mask = self.gaussianMask(F.pad(reverse_mask,(2,2,2,2),mode='reflect'))

        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced 