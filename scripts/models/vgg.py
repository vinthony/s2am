from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models
from scripts.models.rasc import *
from scripts.models.urasc import *

class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23,30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
                
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return (h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class DecoderBlock(nn.Module):
    """docstring for decoderBlock"""
    def __init__(self, inchannels,outchannels,upsample=True):
        super(DecoderBlock, self).__init__()
        
                                        
        if upsample :
            self.deconv = nn.ConvTranspose2d(inchannels, outchannels,kernel_size=4, stride=2, padding=1)
        else:
            self.deconv = nn.Conv2d(inchannels,outchannels,kernel_size=3,stride=1,padding=1)


        self.relu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm2d(outchannels)

    def forward(self,feature):
        return self.relu(self.bn(self.deconv(feature)))

class VGGGenerator(nn.Module):
    """docstring for VGGGenerator"""
    def __init__(self, requires_grad=False, attention_module=URASC, output=3):
        super(VGGGenerator, self).__init__()

        self.encoder = Vgg16(requires_grad=requires_grad)
        self.output = DecoderBlock(64,output,upsample=False)
        self.decoder1 = DecoderBlock(64*2,64,upsample=False)
        self.decoder2 = DecoderBlock(128*2,64)
        self.decoder3 = DecoderBlock(256*2,128)
        self.decoder4 = DecoderBlock(512+256,256)
        self.decoder5 = DecoderBlock(512,256)

        self.rasc1 = attention_module(64*2)
        self.rasc2 = attention_module(128*2)
        self.rasc3 = attention_module(256*2)

    def forward(self,feature):
        rgb = feature[:,0:3,:,:]
        mask = feature[:,3:4,:,:]

        # 256*64, 128*128, 64*256, 32*512, 16*512 
        relu1, relu2, relu3, relu4, relu5 = self.encoder(rgb)

        drelu5 = self.decoder5(relu5) # 512
        drelu4 = self.decoder4(torch.cat((drelu5,relu4),1)) # 256
        drelu3 = self.decoder3(self.rasc3(torch.cat((drelu4,relu3),dim=1),mask)) # 256
        drelu2 = self.decoder2(self.rasc2(torch.cat((drelu3,relu2),dim=1),mask))
        drelu1 = self.decoder1(self.rasc1(torch.cat((drelu2,relu1),dim=1),mask))

        return self.output(drelu1)

        # here we use the faetures from the decoder and skip-connection to rasc, just like rasc-v2


        
        