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
from torch import nn, cuda
from torch.autograd import Variable


class BasicLearningBlock(nn.Module):
    """docstring for BasicLearningBlock"""
    def __init__(self,channel):
        super(BasicLearningBlock, self).__init__()
        self.rconv1 = nn.Conv2d(channel,channel*2,3,padding=1,bias=False)
        self.rbn1 = nn.BatchNorm2d(channel*2)
        self.rconv2 = nn.Conv2d(channel*2,channel,3,padding=1,bias=False)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self,feature):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature)))))) 
        

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        
        if mask_in is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output

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




class PCBlock(nn.Module):
    """docstring for RegionalSkipConnect"""
    def __init__(self,channel):
        super(PCBlock, self).__init__()
        self.rconv1 = PartialConv2d(channel,channel*2,3,padding=1,bias=False,stride=1)
        self.rbn1 = nn.BatchNorm2d(channel*2)
        self.rconv2 = PartialConv2d(channel*2,channel,3,padding=1,bias=False,stride=1)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self,feature,mask_in):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature,mask_in=mask_in))),mask_in=mask_in)))

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


class SEBlock(nn.Module):
    """docstring for SEBlock"""
    def __init__(self, channel,reducation=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reducation,channel),
            nn.Sigmoid())
        
    def forward(self,x):
        b,c,w,h = x.size()
        y1 = self.avg_pool(x).view(b,c)
        y = self.fc(y1).view(b,c,1,1)
        return x*y



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
        yr = 1-y
        return y,yr



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



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPoolX(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPoolX()
        self.spatial = BasicBlock(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


