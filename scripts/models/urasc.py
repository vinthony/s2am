

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


class URASC(nn.Module):
    def __init__(self,channel,learning_model=BasicLearningBlock):
        super(URASC, self).__init__()
        self.connection = learning_model(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.mask_attention = SpatialAttentionModule(channel,16)

    def forward(self,feature,_m):
        _,_,w,_ = feature.size()
      
        mask, reverse_mask = self.mask_attention(feature)

        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced  

class MaskedURASC(nn.Module):
    def __init__(self,channel,learning_model=BasicLearningBlock):
        super(MaskedURASC, self).__init__()
        self.connection = learning_model(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.mask_attention = SpatialAttentionModule(channel,16)

    def forward(self,feature,_m):
      
        mask, reverse_mask = self.mask_attention(feature)

        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced, mask

 
