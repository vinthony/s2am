

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from guided_filter_pytorch.guided_filter import FastGuidedFilter
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

class URASCG(nn.Module):
    def __init__(self,channel,learning_model=BasicLearningBlock):
        super(URASCG, self).__init__()
        self.connection = learning_model(channel)
        self.background_attention = ChannelAttentionModuleGlobal(channel,16)
        self.mixed_attention = ChannelAttentionModuleGlobal(channel,16)
        self.spliced_attention = ChannelAttentionModuleGlobal(channel,16)
        self.mask_attention = SpatialAttentionModuleGlobal(channel,16)

    def forward(self,feature,_m,global_feature):
      
        mask, reverse_mask = self.mask_attention(feature,global_feature)

        background = self.background_attention(feature,global_feature) * reverse_mask
        selected_feature = self.mixed_attention(feature,global_feature)
        spliced_feature = self.spliced_attention(feature,global_feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced, mask


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

class CrossMaskedURASC(nn.Module):
    def __init__(self,channel,learning_model=adainLearningBlock):
        super(CrossMaskedURASC, self).__init__()
        self.connection = learning_model(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.style_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.mask_attention = SpatialAttentionModule(channel,16)

    def forward(self,feature,_m):
      
        mask, reverse_mask = self.mask_attention(feature)

        background_style = self.style_attention(feature) * reverse_mask

        background = self.background_attention(feature) * reverse_mask

        selected_feature = self.mixed_attention(feature)
        
        spliced_feature = self.spliced_attention(feature) 
        
        spliced = ( self.connection(spliced_feature,background_style) + selected_feature ) * mask
        return background + spliced, mask    

class PreCrossMaskedURASC(nn.Module):
    def __init__(self,channel,learning_model=adainLearningBlock):
        super(PreCrossMaskedURASC, self).__init__()
        self.connection = learning_model(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.style_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.mask_attention = SpatialAttentionModule(channel,16)

    def forward(self,feature,_m):
      
        mask, reverse_mask = self.mask_attention(feature)

        background_style = self.style_attention(feature) * reverse_mask

        background = self.background_attention(feature) * reverse_mask

        selected_feature = self.mixed_attention(feature) * mask
        
        spliced_feature = self.spliced_attention(feature) * mask
        
        spliced =  self.connection(spliced_feature, background_style) + selected_feature 
        return background + spliced, mask     
 
