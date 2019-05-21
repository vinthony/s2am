

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from guided_filter_pytorch.guided_filter import FastGuidedFilter


from scripts.utils.model_init import *
from scripts.models.blocks import *

__all__ = ['can','racan','msdgf']

class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.weight = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.bias = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.weight * x + self.bias * self.bn(x)

class CAN(nn.Module):

    def __init__(self,):
        super(CAN, self).__init__()
        self.conv1 = nn.Conv2d(4,24,3,stride=1,padding=1,dilation=1,bias=False)
        self.bn1 = AdaptiveBatchNorm2d(24)

        self.conv2 = nn.Conv2d(24,24,3,stride=1,padding=2,dilation=2,bias=False)
        self.bn2 = AdaptiveBatchNorm2d(24)

        self.conv3 = nn.Conv2d(24,24,3,stride=1,padding=4,dilation=4,bias=False)
        self.bn3 = AdaptiveBatchNorm2d(24)

        self.conv4 = nn.Conv2d(24,24,3,stride=1,padding=8,dilation=8,bias=False)
        self.bn4 = AdaptiveBatchNorm2d(24)

        self.conv5 = nn.Conv2d(24,24,3,stride=1,padding=16,dilation=16,bias=False)
        self.bn5 = AdaptiveBatchNorm2d(24)

        self.conv6 = nn.Conv2d(24,24,3,stride=1,padding=32,dilation=32,bias=False)
        self.bn6 = AdaptiveBatchNorm2d(24)

        self.conv7 = nn.Conv2d(24,24,3,stride=1,padding=64,dilation=64,bias=False)
        self.bn7 = AdaptiveBatchNorm2d(24)

        self.conv8 = nn.Conv2d(24,3,1,stride=1,dilation=1,bias=False)

       

    def forward(self, x):
        # every two blocks we add the block of radhn-v3
        x1 = F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)),0.2) 
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)),0.2)
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)),0.2)
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)),0.2)
        x6 = F.leaky_relu(self.bn6(self.conv6(x5)),0.2)
        x7 = F.leaky_relu(self.bn7(self.conv7(x6)),0.2)
        x10 = self.conv8(x7)

        return x10

def can(**kwargs):
    model = CAN()
    model.apply(weights_init_kaiming)
    return model


class RACAN(nn.Module):

    def __init__(self,):
        super(RACAN, self).__init__()
        self.conv1 = nn.Conv2d(4,24,3,stride=1,padding=1,dilation=1,bias=False)
        self.bn1 = AdaptiveBatchNorm2d(24)

        self.conv2 = nn.Conv2d(24,24,3,stride=1,padding=2,dilation=2,bias=False)
        self.bn2 = AdaptiveBatchNorm2d(24)

        self.conv3 = nn.Conv2d(24,24,3,stride=1,padding=4,dilation=4,bias=False)
        self.bn3 = AdaptiveBatchNorm2d(24)

        self.conv4 = nn.Conv2d(24,24,3,stride=1,padding=8,dilation=8,bias=False)
        self.bn4 = AdaptiveBatchNorm2d(24)

        self.conv5 = nn.Conv2d(24,24,3,stride=1,padding=16,dilation=16,bias=False)
        self.bn5 = AdaptiveBatchNorm2d(24)

        self.conv6 = nn.Conv2d(24,24,3,stride=1,padding=32,dilation=32,bias=False)
        self.bn6 = AdaptiveBatchNorm2d(24)

        self.conv7 = nn.Conv2d(24,24,3,stride=1,padding=64,dilation=64,bias=False)
        self.bn7 = AdaptiveBatchNorm2d(24)

        self.conv8 = nn.Conv2d(24,3,1,stride=1,dilation=1,bias=False)

    
        self.regional_attention1 = RegionalAttentionConnectGaussianMask(24)
        self.regional_attention2 = RegionalAttentionConnectGaussianMask(24)
        self.regional_attention3 = RegionalAttentionConnectGaussianMask(24)


    def forward(self, x):

        img = x[:,0:3,:,:]
        mask  = x[:,3:4,:,:]
        
        # every two blocks we add the block of radhn-v3
        x1 = F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)),0.2) 
        x2 = self.regional_attention1(x2,mask) + x2
        # 
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)),0.2)
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)),0.2)
        x4 = self.regional_attention1(x4,mask) + x4

        x5 = F.leaky_relu(self.bn5(self.conv5(x4)),0.2)
        x6 = F.leaky_relu(self.bn6(self.conv6(x5)),0.2) 
        x7 = F.leaky_relu(self.bn7(self.conv7(x6)),0.2)
        
        x10 = self.conv8(x7)

        return x10

def racan(**kwargs):
    # can with radhn-v3-gaussian
    model = RACAN()
    model.apply(weights_init_kaiming)
    return model


class MSDGF(nn.Module):

    def __init__(self,):
        super(MSDGF, self).__init__()

        radius = 1
        eps=1e-8

        self.pretrained = Vgg16(requires_grad=False)
        self.gf2 = FastGuidedFilter(radius, eps)
        self.gf3 = FastGuidedFilter(radius, eps)
        self.gf4 = FastGuidedFilter(radius, eps)
        self.gf5 = FastGuidedFilter(radius, eps)

        self.guided_f1 = nn.Sequential(
            nn.Conv2d(64,32,1,bias=False),
            AdaptiveBatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,4,1,bias=False)
            )

        self.guided_f2 = nn.Sequential(
            nn.Conv2d(129,32,1,bias=False),
            AdaptiveBatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,3,1,bias=False)
            )
        self.guided_f3 = nn.Sequential(
            nn.Conv2d(257,64,1,bias=False),
            AdaptiveBatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,3,1,bias=False)
            )
        self.guided_f4 = nn.Sequential(
            nn.Conv2d(513,128,1,bias=False),
            AdaptiveBatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,3,1,bias=False)
            )


        self.guided_f5 = nn.Sequential(
            nn.Conv2d(513,256,1,bias=False),
            AdaptiveBatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,3,1,bias=False)
            )


    def forward(self, x):
        img = x[:,0:3,:,:]
        msk = x[:,3:4,:,:]
        v16 = self.pretrained(img)

        vgg16_mask = F.softmax(self.guided_f1(v16.relu1_2),dim=1)
        vgg16_l2 = self.gf2(F.avg_pool2d(img,2),self.guided_f2(torch.cat([v16.relu2_2,F.avg_pool2d(msk,2)],dim=1)),img).clamp(0, 1)
        vgg16_l3 = self.gf3(F.avg_pool2d(img,4),self.guided_f3(torch.cat([v16.relu3_3,F.avg_pool2d(msk,4)],dim=1)),img).clamp(0, 1)
        vgg16_l4 = self.gf4(F.avg_pool2d(img,8),self.guided_f4(torch.cat([v16.relu4_3,F.avg_pool2d(msk,8)],dim=1)),img).clamp(0, 1)
        vgg16_l5 = self.gf5(F.avg_pool2d(img,16),self.guided_f5(torch.cat([v16.relu5_3,F.avg_pool2d(msk,16)],dim=1)),img).clamp(0, 1)
        
        output =  vgg16_mask[:,0:1,:,:] * vgg16_l2 + vgg16_mask[:,1:2,:,:] * vgg16_l3 + vgg16_mask[:,2:3,:,:] * vgg16_l4 + vgg16_l4 * vgg16_mask[:,3:4,:,:] 
        return output

def msdgf(**kwargs):
    model = MSDGF()
    model.apply(weights_init_kaiming)
    return model


