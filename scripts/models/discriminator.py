

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math

import torchvision
from torchvision import datasets, models, transforms

from scripts.utils.model_init import *
from scripts.models.unet import oUnetGenerator,UnetGenerator

# radhn with unet 

__all__ = ['patchgan','patchganwithoutnorm','pixelgan','compared']


class CompareDiscriminator(nn.Module):
    def __init__(self, in_channels=3, normalization=True):
        super(CompareDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        # # patchgan(16)
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=normalization), # 256->128
            *discriminator_block(64, 128, normalization=normalization),# 128->64
            *discriminator_block(128, 256, normalization=normalization), # 64->32
            *discriminator_block(256, 256, normalization=normalization), # 32->16 (16*16*512)
        )

        self.classify = nn.Sequential(
                *discriminator_block(512, 256, normalization=normalization), # 8x8x256
                *discriminator_block(256, 128, normalization=normalization), # 4x4x128
                *discriminator_block(128, 64, normalization=normalization), # 2x2x64
                *discriminator_block(64, 64, normalization=normalization), # 1x1x64
            )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # normalized
        # trans = transforms.normalize(mean=mean,std=std)
        # trans(img_A)
        # trans(img_B)

        fA = self.model(img_A)
        fB = self.model(img_B)

        return self.classify(torch.cat((fA, fB), 1)).view(-1,8,8)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, normalization=True):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # patchgan(16)
        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 64, normalization=False), # 256->128
            *discriminator_block(64, 128, normalization=normalization),# 128->64
            *discriminator_block(128, 256, normalization=normalization), # 64->32
            *discriminator_block(256, 512, normalization=normalization), # 32->16
            nn.ZeroPad2d((1, 0, 1, 0)), 
            nn.Conv2d(512, 1, 4, padding=1, bias=False) 
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class ImageDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(ImageDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=3, stride=1, padding=1, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc=6, ndf=64):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func != nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=True)]

        self.model = nn.Sequential(*self.net)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

def patchgan():
    model = Discriminator()
    model.apply(weights_init_normal)
    return model

def patchganwithoutnorm():
    model = Discriminator(normalization=False)
    model.apply(weights_init_normal)
    return model

def pixelgan():
    model = PixelDiscriminator()
    model.apply(weights_init_normal)
    return model

def compared():
    model = CompareDiscriminator()
    model.apply(weights_init_normal)
    return model