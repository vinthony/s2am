import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math
import numbers

from scripts.utils.model_init import *
from scripts.models.vgg import Vgg16
from scripts.models.blocks import *


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
        

# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class ChannelPool(nn.Module):
    def __init__(self,types):
        super(ChannelPool, self).__init__()
        if types == 'avg': 
            self.poolingx = nn.AdaptiveAvgPool1d(1)
        elif types == 'max':
            self.poolingx = nn.AdaptiveMaxPool1d(1)
        else:
            raise 'inner error'

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1) 
        pooled =  self.poolingx(input)# b,w*h,c ->  b,w*h,1
        _, _, c = pooled.size()
        return pooled.view(n,c,w,h)

class RegionalSkipConnect(nn.Module):
    """docstring for RegionalSkipConnect"""
    def __init__(self,channel):
        super(RegionalSkipConnect, self).__init__()
        self.rconv1 = nn.Conv2d(channel,channel*2,3,padding=1,bias=False)
        self.rbn1 = nn.BatchNorm2d(channel*2)
        self.rconv2 = nn.Conv2d(channel*2,channel,3,padding=1,bias=False)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self,feature):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature)))))) 

class NNSkipConnect(nn.Module):
    """docstring for RegionalSkipConnect"""
    def __init__(self,channel):
        super(NNSkipConnect, self).__init__()
        self.rconv1 = nn.Conv2d(channel,channel*2,3,padding=1,bias=False)
        self.rbn1 = nn.BatchNorm2d(channel*2)
        self.rconv2 = nn.Conv2d(channel*2,channel,3,padding=1,bias=False)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self,feature,mask=None):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature)))))) 


class GlobalAttentionModule(nn.Module):
    """docstring for GlobalAttentionModule"""
    def __init__(self, channel,reducation=16):
        super(GlobalAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*2,channel//reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reducation,channel),
            nn.Sigmoid())
        
    def forward(self,x):
        b,c,w,h = x.size()
        y1 = self.avg_pool(x).view(b,c)
        y2 = self.max_pool(x).view(b,c)
        y = self.fc(torch.cat([y1,y2],1)).view(b,c,1,1)
        return x*y

class SpatialAttentionModule(nn.Module):
    """docstring for SpatialAttentionModule"""
    def __init__(self, channel,reducation=16):
        super(SpatialAttentionModule, self).__init__()
        self.avg_pool = ChannelPool('avg')
        self.max_pool = ChannelPool('max')
        self.fc = nn.Sequential(
            nn.Conv2d(2,reducation,7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reducation,1,7,stride=1,padding=3),
            nn.Sigmoid())
        
    def forward(self,x):
        b,c,w,h = x.size()
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = self.fc(torch.cat([y1,y2],1))
        return x*y




class GlobalAttentionModuleJustSigmoid(nn.Module):
    """docstring for GlobalAttentionModule"""
    def __init__(self, channel,reducation=16):
        super(GlobalAttentionModuleJustSigmoid, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*2,channel//reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reducation,channel),
            nn.Sigmoid())
        
    def forward(self,x):
        b,c,w,h = x.size()
        y1 = self.avg_pool(x).view(b,c)
        y2 = self.max_pool(x).view(b,c)
        y = self.fc(torch.cat([y1,y2],1)).view(b,c,1,1)
        return y

class RegionalAttentionMask(nn.Module):
    """docstring for RegionalAttentionConnect"""
    def __init__(self,channel,type_of_connection=RegionalSkipConnect):
        super(RegionalAttentionMask, self).__init__()
        self.connection = type_of_connection(channel)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        background = feature * reverse_mask
        spliced = self.connection(feature) * mask
        return background + spliced  


class RegionalAttentionConnect(nn.Module):
    """docstring for RegionalAttentionConnect"""
    def __init__(self,channel,type_of_connection=RegionalSkipConnect):
        super(RegionalAttentionConnect, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        background = self.background_attention(feature)* reverse_mask
        spliced_feature = self.connection(feature)
        spliced = self.spliced_attention(spliced_feature) * mask
        return background + spliced  

class RegionalAttentionConnectv2(nn.Module):
    """docstring for RegionalAttentionConnectv2"""
    def __init__(self,channel,type_of_connection=RegionalSkipConnect):
        super(RegionalAttentionConnectv2, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced 

class RegionalAttentionConnectWithoutMask(nn.Module):
    """docstring for RegionalAttentionConnectv2"""
    def __init__(self,channel,type_of_connection=RegionalSkipConnect):
        super(RegionalAttentionConnectWithoutMask, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.mask_attention = SpatialAttentionModule(channel,16)
        self.reverse_mask_attention = SpatialAttentionModule(channel,16)

    def forward(self,feature,m):
        _,_,w,_ = feature.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        
        # here we build the mask by the input feature

        mask = self.mask_attention(feature)
        reverse_mask = self.reverse_mask_attention(feature)

        reverse_mask = -1*(mask-1)
        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced 


class PreRegionalAttentionConnectv2(nn.Module):
    """docstring for RegionalAttentionConnectv2"""
    def __init__(self,channel,type_of_connection=RegionalSkipConnect):
        super(PreRegionalAttentionConnectv2, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        background = self.background_attention(feature * reverse_mask)
        selected_feature = self.mixed_attention(feature * mask)
        spliced_feature = self.spliced_attention(feature * mask) 
        spliced = ( self.connection(spliced_feature) + selected_feature )
        return background + spliced   


class PreRegionalAttentionConnectADAIN(nn.Module):
    def __init__(self,channel):
        super(PreRegionalAttentionConnectADAIN, self).__init__()
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        background = self.background_attention(feature * reverse_mask)
        selected_feature = self.mixed_attention(feature * mask)
        spliced_feature = self.spliced_attention(feature * mask) 
        spliced = ( adaptive_instance_normalization(spliced_feature, background) + selected_feature )
        return background + spliced   


class RegionalAttentionConnectv3(nn.Module):
    """docstring for RegionalAttentionConnectv3"""
    def __init__(self,channel,type_of_connection=RegionalSkipConnect):
        super(RegionalAttentionConnectv3, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModuleJustSigmoid(channel,16)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        background = self.background_attention(feature) * reverse_mask
        choosen_channel = self.spliced_attention(feature)
        spliced_feature = choosen_channel * feature
        selected_feature = (1 - choosen_channel) * feature
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced     




class RegionalAttentionConnectGaussianMask(nn.Module):
    def __init__(self,channel,type_of_connection=RegionalSkipConnect):
        super(RegionalAttentionConnectGaussianMask, self).__init__()
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



class MinimalUnet(nn.Module):
    """docstring for MinimalUnet"""
    def __init__(self, down=None,up=None,submodule=None,attention=None,withoutskip=False):
        super(MinimalUnet, self).__init__()
        
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up) 
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None 
        self.is_sub = not submodule == None 

    
    def forward(self,x,mask=None):
        if self.is_sub: 
            x_up,_ = self.sub(self.down(x),mask)
        else:
            x_up = self.down(x)

        if self.is_attention:
            x = self.attention(x,mask)
        
        if self.withoutskip: #outer or inner.
            x_out = self.up(x_up)
        else:
            x_out = (torch.cat([x,self.up(x_up)],1),mask)

        return x_out

        

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,is_attention_layer=False,attention_model=RegionalAttentionConnect):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = MinimalUnet(down,up,submodule,withoutskip=outermost)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = MinimalUnet(down,up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if is_attention_layer:
                attention_model = attention_model(input_nc)
            else:
                attention_model = None
            if use_dropout:
                model = MinimalUnet(down,up.append(nn.Dropout(0.5)),submodule,attention_model)
            else:
                model = MinimalUnet(down,up,submodule,attention_model)

        self.model = model


    def forward(self, x,mask):
        # build the mask for attention use
        return self.model(x,mask)
            
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,is_attention_layer=False,attention_model=RegionalAttentionConnect,use_inner_attention=False):
        super(UnetGenerator, self).__init__()

        # 8 for 256x256
        # 9 for 512x512
        # construct unet structure
        self.need_mask = not input_nc == output_nc

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True) # 1
        for i in range(num_downs - 5): #3 times
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,is_attention_layer=use_inner_attention,attention_model=attention_model) # 8,4,2
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model) #16
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model) #32
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model) #64 
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer) # 128

        self.model = unet_block

    def forward(self, input):
        if self.need_mask:
            return self.model(input,input[:,3:4,:,:])
        else:
            return self.model(input[:,0:3,:,:],input[:,3:4,:,:])