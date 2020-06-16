

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


from scripts.utils.model_init import *
from scripts.models.vgg import Vgg16

__all__ = ['dhn','dhn256','dih256','dih256seg']

class DHN(nn.Module):

    def __init__(self,):
        super(DHN, self).__init__()
        self.conv1 = nn.Conv2d(4,64,4,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,64,4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,128,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128,128,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128,256,4,stride=2,padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256,256,4,stride=2,padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256,512,4,stride=2,padding=1,bias=False)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512,1024,4,stride=2,bias=False)
        # self.bn8 = nn.BatchNorm2d(1024)

        self.dev1 = nn.ConvTranspose2d(1024,512,4,stride=2,bias=False)
        self.dbn1 = nn.BatchNorm2d(512)

        self.dev2 = nn.ConvTranspose2d(512,256,4,padding=1,stride=2,bias=False)
        self.dbn2 = nn.BatchNorm2d(256)

        self.dev3 = nn.ConvTranspose2d(256,256,4,padding=1,stride=2,bias=False)
        self.dbn3 = nn.BatchNorm2d(256)

        self.dev4 = nn.ConvTranspose2d(256,128,4,padding=1,stride=2,bias=False)
        self.dbn4 = nn.BatchNorm2d(128)

        self.dev5 = nn.ConvTranspose2d(128,128,4,padding=1,stride=2,bias=False)
        self.dbn5 = nn.BatchNorm2d(128)

        self.dev6 = nn.ConvTranspose2d(128,64,4,padding=1,stride=2,bias=False)
        self.dbn6 = nn.BatchNorm2d(64)

        self.dev7 = nn.ConvTranspose2d(64,64,4,padding=1,stride=2,bias=False)
        self.dbn7 = nn.BatchNorm2d(64)

        self.dev8 = nn.ConvTranspose2d(64,3,4,padding=1,stride=2,bias=False)



    def forward(self, x):
        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))
        x6 = F.elu(self.bn6(self.conv6(x5)))
        x7 = F.elu(self.bn7(self.conv7(x6)))
        x8 = F.elu(self.conv8(x7))

        d1 = F.elu(self.dbn1(self.dev1(x8))) + x7
        d2 = F.elu(self.dbn2(self.dev2(d1))) + x6
        d3 = F.elu(self.dbn3(self.dev3(d2))) + x5
        d4 = F.elu(self.dbn4(self.dev4(d3))) + x4
        d5 = F.elu(self.dbn5(self.dev5(d4))) + x3
        d6 = F.elu(self.dbn6(self.dev6(d5))) + x2
        d7 = F.elu(self.dbn7(self.dev7(d6))) + x1
        d8 = self.dev8(d7) 

        return d8

class DHN256(nn.Module):

    def __init__(self,):
        super(DHN256, self).__init__()
        self.conv1 = nn.Conv2d(4,64,4,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,64,4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,128,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128,128,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128,256,4,stride=2,padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256,512,4,stride=2,padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512,1024,4,stride=2,bias=False)

        self.dev1 = nn.ConvTranspose2d(1024,512,4,stride=2,bias=False)
        self.dbn1 = nn.BatchNorm2d(512)

        self.dev3 = nn.ConvTranspose2d(512,256,4,padding=1,stride=2,bias=False)
        self.dbn3 = nn.BatchNorm2d(256)

        self.dev4 = nn.ConvTranspose2d(256,128,4,padding=1,stride=2,bias=False)
        self.dbn4 = nn.BatchNorm2d(128)

        self.dev5 = nn.ConvTranspose2d(128,128,4,padding=1,stride=2,bias=False)
        self.dbn5 = nn.BatchNorm2d(128)

        self.dev6 = nn.ConvTranspose2d(128,64,4,padding=1,stride=2,bias=False)
        self.dbn6 = nn.BatchNorm2d(64)

        self.dev7 = nn.ConvTranspose2d(64,64,4,padding=1,stride=2,bias=False)
        self.dbn7 = nn.BatchNorm2d(64)

        self.dev8 = nn.ConvTranspose2d(64,3,4,padding=1,stride=2,bias=False)


    def forward(self, x):

        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))
        x6 = F.elu(self.bn6(self.conv6(x5)))
        x8 = F.elu(self.conv8(x6))

        d1 = F.elu(self.dbn1(self.dev1(x8))) + x6
        d3 = F.elu(self.dbn3(self.dev3(d1))) + x5
        d4 = F.elu(self.dbn4(self.dev4(d3))) + x4
        d5 = F.elu(self.dbn5(self.dev5(d4))) + x3
        d6 = F.elu(self.dbn6(self.dev6(d5))) + x2
        d7 = F.elu(self.dbn7(self.dev7(d6))) + x1
        d8 = self.dev8(d7) 

        return d8

class DIH256(nn.Module):

    def __init__(self,):
        super(DIH256, self).__init__()
        self.conv1 = nn.Conv2d(4,64,4,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,64,4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,128,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128,128,4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128,256,4,stride=2,padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256,512,4,stride=2,padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512*4*4,1024)
        
        self.dev1 = nn.ConvTranspose2d(1024,512,4,stride=2,bias=False)
        self.dbn1 = nn.BatchNorm2d(512)

        self.dev3 = nn.ConvTranspose2d(512,256,4,padding=1,stride=2,bias=False)
        self.dbn3 = nn.BatchNorm2d(256)

        self.dev4 = nn.ConvTranspose2d(256,128,4,padding=1,stride=2,bias=False)
        self.dbn4 = nn.BatchNorm2d(128)

        self.dev5 = nn.ConvTranspose2d(128,128,4,padding=1,stride=2,bias=False)
        self.dbn5 = nn.BatchNorm2d(128)

        self.dev6 = nn.ConvTranspose2d(128,64,4,padding=1,stride=2,bias=False)
        self.dbn6 = nn.BatchNorm2d(64)

        self.dev7 = nn.ConvTranspose2d(64,64,4,padding=1,stride=2,bias=False)
        self.dbn7 = nn.BatchNorm2d(64)

        self.dev8 = nn.ConvTranspose2d(64,3,4,padding=1,stride=2,bias=False)


    def forward(self, x):

        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))
        x6 = F.elu(self.bn6(self.conv6(x5)))
        
        o0 = self.fc(x6.view(x.size(0),-1))
        o0 = o0.view(x.size(0),-1,1,1)

        d1 = F.elu(self.dbn1(self.dev1(o0))) + x6
        d3 = F.elu(self.dbn3(self.dev3(d1))) + x5
        d4 = F.elu(self.dbn4(self.dev4(d3))) + x4
        d5 = F.elu(self.dbn5(self.dev5(d4))) + x3
        d6 = F.elu(self.dbn6(self.dev6(d5))) + x2
        d7 = F.elu(self.dbn7(self.dev7(d6))) + x1
        d8 = self.dev8(d7) 

        return d8

class DIH256SEG(nn.Module):
    def __init__(self,):
        super(DIH256SEG, self).__init__()
        self.conv1 = nn.Conv2d(4,64,4,stride=2,padding=1) 
        self.bn1 = nn.BatchNorm2d(64) #128

        self.conv2 = nn.Conv2d(64,64,4,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(64) #64

        self.conv3 = nn.Conv2d(64,128,4,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(128) #32

        self.conv4 = nn.Conv2d(128,128,4,stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(128) #16

        self.conv5 = nn.Conv2d(128,256,4,stride=2,padding=1)
        self.bn5 = nn.BatchNorm2d(256) #8

        self.conv6 = nn.Conv2d(256,512,4,stride=2,padding=1)
        self.bn6 = nn.BatchNorm2d(512) #4

        self.fc = nn.Linear(512*4*4,1024)

        self.dev1 = nn.ConvTranspose2d(1024,512,4,stride=2)
        self.dbn1 = nn.BatchNorm2d(512)

        self.dev3 = nn.ConvTranspose2d(1024,256,4,padding=1,stride=2)
        self.dbn3 = nn.BatchNorm2d(256)

        self.dev4 = nn.ConvTranspose2d(256*2,128,4,padding=1,stride=2)
        self.dbn4 = nn.BatchNorm2d(128)

        self.dev5 = nn.ConvTranspose2d(128*2,128,4,padding=1,stride=2)
        self.dbn5 = nn.BatchNorm2d(128)

        self.dev6 = nn.ConvTranspose2d(128*2,64,4,padding=1,stride=2)
        self.dbn6 = nn.BatchNorm2d(64)

        self.dev7 = nn.ConvTranspose2d(64*2,64,4,padding=1,stride=2)
        self.dbn7 = nn.BatchNorm2d(64)

        self.dev8 = nn.ConvTranspose2d(64,3,4,padding=1,stride=2)
    #       seg branch
        self.sdev1 = nn.ConvTranspose2d(1024,512,4,stride=2)
        self.sdbn1 = nn.BatchNorm2d(512)

        self.sdev3 = nn.ConvTranspose2d(512,256,4,padding=1,stride=2)
        self.sdbn3 = nn.BatchNorm2d(256)

        self.sdev4 = nn.ConvTranspose2d(256,128,4,padding=1,stride=2)
        self.sdbn4 = nn.BatchNorm2d(128)

        self.sdev5 = nn.ConvTranspose2d(128,128,4,padding=1,stride=2)
        self.sdbn5 = nn.BatchNorm2d(128)

        self.sdev6 = nn.ConvTranspose2d(128,64,4,padding=1,stride=2)
        self.sdbn6 = nn.BatchNorm2d(64)

        self.sdev7 = nn.ConvTranspose2d(64,64,4,padding=1,stride=2)
        self.sdbn7 = nn.BatchNorm2d(64)

        self.sdev8 = nn.ConvTranspose2d(64,64,4,padding=1,stride=2)
        self.sdbn8 = nn.BatchNorm2d(64)

        self.sdev9 = nn.Conv2d(64,26,1)


    def forward(self, x):

        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))
        x6 = F.elu(self.bn6(self.conv6(x5)))
        
        o0 = self.fc(x6.view(x.size(0),-1))

        o0 = o0.view(x.size(0),-1,1,1)

        d1 = F.elu(self.dbn1(self.dev1(o0))) + x6
        sd1 = F.elu(self.sdbn1(self.sdev1(o0))) + x6

        o1 = torch.cat((d1,sd1),1)

        d3 = F.elu(self.dbn3(self.dev3(o1))) + x5
        sd3 = F.elu(self.sdbn3(self.sdev3(d1))) + x5

        o3 = torch.cat((d3,sd3),1)

        d4 = F.elu(self.dbn4(self.dev4(o3))) + x4
        sd4 = F.elu(self.sdbn4(self.sdev4(sd3))) + x4

        o4 = torch.cat((d4,sd4),1)

        d5 = F.elu(self.dbn5(self.dev5(o4))) + x3
        sd5 = F.elu(self.sdbn5(self.sdev5(sd4))) + x3
        
        o5 = torch.cat((d5,sd5),1)

        d6 = F.elu(self.dbn6(self.dev6(o5))) + x2
        sd6 = F.elu(self.sdbn6(self.sdev6(sd5))) + x2

        o6 = torch.cat((d6,sd6),1)
        
        d7 = F.elu(self.dbn7(self.dev7(o6))) + x1
        sd7 = F.elu(self.sdbn7(self.sdev7(sd6))) + x1 #128

        sd8 = F.elu(self.sdbn8(self.sdev8(sd7)))

        sd9 = self.sdev9(sd8)

        d8 = self.dev8(d7) 

        return d8,sd9


def dhn(**kwargs):
    model = DHN()
    model.apply(weights_init_kaiming)
    return model

def dhn256(**kwargs):
    model = DHN256()
    model.apply(weights_init_kaiming)
    return model

def dih256(**kwargs):
    model = DIH256()
    model.apply(weights_init_kaiming)
    return model

def dih256seg(**kwargs):
    model = DIH256SEG()
    model.apply(weights_init_kaiming)
    return model


