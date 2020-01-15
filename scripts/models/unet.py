import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import functools
from scripts.models.blocks import *
from scripts.models.rasc import *
from scripts.models.urasc import *


class MaskedMinimalUnetV2(nn.Module):
    """docstring for MinimalUnet"""
    def __init__(self, down=None,up=None,submodule=None,attention=None,withoutskip=False,outermostattention=False,**kwags):
        super(MaskedMinimalUnetV2, self).__init__()
        
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up) 
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None 
        self.is_sub = not submodule == None 
        self.outermostattention = outermostattention
    
    def forward(self,x,mask=None,m_out=None):
        if self.is_sub: 
            if self.withoutskip:
                x_up,_,m_out = self.sub(self.down(x),mask)
            else:
                x_up,_ = self.sub(self.down(x),mask) 
        else:
            x_up = self.down(x)

        if self.withoutskip: #outer or inner.
            x_out = (self.up(x_up), nn.functional.interpolate(m_out,scale_factor=2,mode='bilinear',align_corners=True))
        else:
            if self.is_attention:
                x_, m_out = self.attention(torch.cat([x,self.up(x_up)],1),mask)
                if self.outermostattention:
                    x_out = (x_,mask,m_out)
                else:
                    x_out = (x_,mask)
            else:
                x_out = (torch.cat([x,self.up(x_up)],1),mask)

        return x_out


class MinimalUnetV2(nn.Module):
    """docstring for MinimalUnet"""
    def __init__(self, down=None,up=None,submodule=None,attention=None,withoutskip=False,**kwags):
        super(MinimalUnetV2, self).__init__()
        
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

        if self.withoutskip: #outer or inner.
            x_out = self.up(x_up)
        else:
            if self.is_attention:
                x_out = (self.attention(torch.cat([x,self.up(x_up)],1),mask),mask)
            else:
                x_out = (torch.cat([x,self.up(x_up)],1),mask)

        return x_out


class MinimalUnet(nn.Module):
    """docstring for MinimalUnet"""
    def __init__(self, down=None,up=None,submodule=None,attention=None,withoutskip=False,**kwags):
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
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,is_attention_layer=False,
                 attention_model=RASC,basicblock=MinimalUnet,outermostattention=False,conv_block=nn.Conv2d):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = conv_block(input_nc, inner_nc, kernel_size=4,
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
            model = basicblock(down,up,submodule,withoutskip=outermost)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = basicblock(down,up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if is_attention_layer:
                if MinimalUnetV2.__qualname__ in basicblock.__qualname__  :
                    attention_model = attention_model(input_nc*2)
                elif MinimalUnetInpainting.__qualname__ in basicblock.__qualname__:
                    attention_model = attention_model(inner_nc)   
                else:
                    attention_model = attention_model(input_nc)   
            else:
                attention_model = None
                
            if use_dropout:
                model = basicblock(down,up.append(nn.Dropout(0.5)),submodule,attention_model,outermostattention=outermostattention)
            else:
                model = basicblock(down,up,submodule,attention_model,outermostattention=outermostattention)

        self.model = model


    def forward(self, x,mask=None):
        # build the mask for attention use
        return self.model(x,mask)
            
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,basicblock=MinimalUnet,final_layer=None,outputmask=False,conv_block=nn.Conv2d):
        super(UnetGenerator, self).__init__()

        # 8 for 256x256
        # 9 for 512x512
        # construct unet structure
        self.need_mask = ( (not input_nc == output_nc) and input_nc == 4 )

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,basicblock=basicblock,conv_block=conv_block) # 1
        for i in range(num_downs - 5): #3 times
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,is_attention_layer=use_inner_attention,attention_model=attention_model,basicblock=basicblock,conv_block=conv_block) # 8,4,2
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock,conv_block=conv_block) #16
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock,conv_block=conv_block) #32
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock, outermostattention=True,conv_block=conv_block) #64 
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, basicblock=basicblock, norm_layer=norm_layer) # 128

        self.model = unet_block
        self.final_layer = final_layer
        self.outputmask = outputmask

    def forward(self, input):
        if self.need_mask:
            return torch.tanh(self.model(input,input[:,3:4,:,:]))
        else:
            output = self.model(input[:,0:3,:,:],input[:,3:4,:,:])
            if self.outputmask and self.final_layer:
                return (torch.tanh(output[0]),output[1])
            elif self.final_layer:
                return torch.tanh(output)
            else:
                return output


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class oUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(oUnetGenerator, self).__init__()

        # construct unet structure
        unet_block = oUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = oUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = oUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = oUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = oUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = oUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class oUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(oUnetSkipConnectionBlock, self).__init__()
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
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class NaiveMinimalUnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,learning_model=None):
        super(NaiveMinimalUnet, self).__init__()

        # construct unet structure

        self.dropout = 0.5 if use_dropout else 0

        self.down1 = UNetDown(input_nc, ngf, normalize=False) # 2
        self.down2 = UNetDown(ngf, ngf*2) # 4
        self.down3 = UNetDown(ngf*2, ngf*4) # 8
        self.down4 = UNetDown(ngf*4, ngf*8, dropout=self.dropout) # 16
        self.down5 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 32
        self.down6 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 64
        self.down7 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 128
        self.down8 = UNetDown(ngf*8, ngf*8, normalize=False, dropout=self.dropout) #256

        self.up1 = UNetUp(ngf*8, ngf*8, dropout=self.dropout) #128
        self.up2 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #64
        self.up3 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #32
        self.up4 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #16
        self.up5 = UNetUp(ngf*16, ngf*4) #8
        self.up6 = UNetUp(ngf*8, ngf*2) #4
        self.up7 = UNetUp(ngf*4, ngf) #2

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, output_nc, 4, padding=1)
        )

    def forward(self, x, mask):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7) # the connected features.
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3) #
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class NaiveRASC(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,learning_model=BasicLearningBlockLite):
        super(NaiveRASC, self).__init__()

        # construct unet structure

        self.dropout = 0.5 if use_dropout else 0

        self.down1 = UNetDown(input_nc, ngf, normalize=False) # 2
        self.down2 = UNetDown(ngf, ngf*2) # 4
        self.down3 = UNetDown(ngf*2, ngf*4) # 8
        self.down4 = UNetDown(ngf*4, ngf*8, dropout=self.dropout) # 16
        self.down5 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 32
        self.down6 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 64
        self.down7 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 128
        self.down8 = UNetDown(ngf*8, ngf*8, normalize=False, dropout=self.dropout) #256

        self.up1 = UNetUp(ngf*8, ngf*8, dropout=self.dropout) #128
        self.up2 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #64
        self.up3 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #32
        self.up4 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #16
        self.up5 = UNetUp(ngf*16, ngf*4) #8
        self.up6 = UNetUp(ngf*8, ngf*2) #4
        self.up7 = UNetUp(ngf*4, ngf) #2

        self.rasc5 = attention_model(ngf*8,learning_model=learning_model)
        self.rasc6 = attention_model(ngf*4,learning_model=learning_model) 
        self.rasc7 = attention_model(ngf*2,learning_model=learning_model)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, output_nc, 4, padding=1)
        )

    def forward(self, x, mask):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7) # the connected features.
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        
        u5 = self.up5(u4, d3) #
        u5,m5 = self.rasc5(u5,mask) 

        u6 = self.up6(u5, d2)
        u6,m6 = self.rasc6(u6,mask) # 32*32
        
        u7 = self.up7(u6, d1)       # 64*64
        u7,m7 = self.rasc7(u7,mask) # 128*128

        return self.final(u7),m5,m6,m7


class NaiveMultiURASC(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,learning_model=BasicLearningBlockLite):
        super(NaiveMultiURASC, self).__init__()

        # construct unet structure

        self.dropout = 0.5 if use_dropout else 0

        self.down1 = UNetDown(input_nc, ngf, normalize=False) # 2
        self.down2 = UNetDown(ngf, ngf*2) # 4
        self.down3 = UNetDown(ngf*2, ngf*4) # 8
        self.down4 = UNetDown(ngf*4, ngf*8, dropout=self.dropout) # 16
        self.down5 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 32
        self.down6 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 64
        self.down7 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 128
        self.down8 = UNetDown(ngf*8, ngf*8, normalize=False, dropout=self.dropout) #256

        self.up1 = UNetUp(ngf*8, ngf*8, dropout=self.dropout) #128
        self.up2 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #64
        self.up3 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #32
        self.up4 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #16
        self.up5 = UNetUp(ngf*16, ngf*4) #8
        self.up6 = UNetUp(ngf*8, ngf*2) #4
        self.up7 = UNetUp(ngf*4, ngf) #2

        self.rasc5 = attention_model(ngf*8,learning_model=learning_model)
        self.rasc6 = attention_model(ngf*4,learning_model=learning_model) 
        self.rasc7 = attention_model(ngf*2,learning_model=learning_model)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, output_nc, 4, padding=1)
        )

    def forward(self, x, mask):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7) # the connected features.
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        
        u5 = self.up5(u4, d3) #
        u5,m5 = self.rasc5(u5,mask) 

        u6 = self.up6(u5, d2)
        u6,m6 = self.rasc6(u6,mask) # 32*32
        
        u7 = self.up7(u6, d1)       # 64*64
        u7,m7 = self.rasc7(u7,mask) # 128*128

        return self.final(u7),m5,m6,m7


class NaiveMultiURASCG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,learning_model=BasicLearningBlockLite):
        super(NaiveMultiURASCG, self).__init__()

        # construct unet structure

        self.dropout = 0.5 if use_dropout else 0

        self.down1 = UNetDown(input_nc, ngf, normalize=False) # 2
        self.down2 = UNetDown(ngf, ngf*2) # 4
        self.down3 = UNetDown(ngf*2, ngf*4) # 8
        self.down4 = UNetDown(ngf*4, ngf*8, dropout=self.dropout) # 16
        self.down5 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 32
        self.down6 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 64
        self.down7 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 128
        self.down8 = UNetDown(ngf*8, ngf*8, normalize=False, dropout=self.dropout) #256

        self.up1 = UNetUp(ngf*8, ngf*8, dropout=self.dropout) #128
        self.up2 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #64
        self.up3 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #32
        self.up4 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #16
        self.up5 = UNetUp(ngf*16, ngf*4) #8
        self.up6 = UNetUp(ngf*8, ngf*2) #4
        self.up7 = UNetUp(ngf*4, ngf) #2

        self.rasc5 = attention_model(ngf*8,learning_model=learning_model)
        self.rasc6 = attention_model(ngf*4,learning_model=learning_model) 
        self.rasc7 = attention_model(ngf*2,learning_model=learning_model)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, output_nc, 4, padding=1)
        )

    def forward(self, x, mask):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7) # the connected features.
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        
        u5 = self.up5(u4, d3) #
        u5,m5 = self.rasc5(u5,mask,d8) 

        u6 = self.up6(u5, d2)
        u6,m6 = self.rasc6(u6,mask,d8) # 32*32
        
        u7 = self.up7(u6, d1)       # 64*64
        u7,m7 = self.rasc7(u7,mask,d8) # 128*128

        return self.final(u7),m5,m6,m7

class NaiveMultiURASCX(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,learning_model=BasicLearningBlockLite):
        super(NaiveMultiURASCX, self).__init__()

        # construct unet structure

        self.dropout = 0.5 if use_dropout else 0

        self.down1 = UNetDown(input_nc, ngf, normalize=False) # 2
        self.down2 = UNetDown(ngf, ngf*2) # 4
        self.down3 = UNetDown(ngf*2, ngf*4) # 8
        self.down4 = UNetDown(ngf*4, ngf*8, dropout=self.dropout) # 16
        self.down5 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 32
        self.down6 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 64
        self.down7 = UNetDown(ngf*8, ngf*8, dropout=self.dropout) # 128
        self.down8 = UNetDown(ngf*8, ngf*8, normalize=False, dropout=self.dropout) #256

        self.up1 = UNetUp(ngf*8, ngf*8, dropout=self.dropout) #128
        self.up2 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #64
        self.up3 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #32
        self.up4 = UNetUp(ngf*16, ngf*8, dropout=self.dropout) #16
        self.up5 = UNetUp(ngf*16, ngf*4) #8
        self.up6 = UNetUp(ngf*8, ngf*2) #4
        self.up7 = UNetUp(ngf*4, ngf) #2

        self.rasc5 = attention_model(ngf*8,learning_model=learning_model)
        self.rasc6 = attention_model(ngf*4,learning_model=learning_model) 
        self.rasc7 = attention_model(ngf*2,learning_model=learning_model)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, output_nc, 4, padding=1)
        )

    def forward(self, x, mask):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7) # the connected features.
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        
        u5 = self.up5(u4, d3) #
        u5,m5 = self.rasc5(u5,mask) 

        u6 = self.up6(u5, d2)
        u6,m6 = self.rasc6(u6,mask) # 32*32
        
        u7 = self.up7(u6, d1)       # 64*64
        u7,m7 = self.rasc7(u7,mask) # 128*128

        mx = F.interpolate(m7,scale_factor=2,mode='bilinear',align_corners=True)

        output = mx * self.final(u7) + (1-mx) * x

        return output,m5,m6,m7

class NaiveUnetGenerator(nn.Module):
    """docstring for NaiveUnetGenerator"""
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,basicblock=NaiveMinimalUnet,learning_model=BasicLearningBlockLite,final_layer=False,outputmask=False):
        super(NaiveUnetGenerator, self).__init__()
        
        self.need_mask = ( (not input_nc == output_nc) and input_nc == 4 )
        
        self.model = basicblock(input_nc,output_nc,ngf=64,use_dropout=use_dropout,attention_model=attention_model,learning_model=learning_model)

        self.final_layer = final_layer
        self.outputmask = outputmask


    def forward(self, input):
        if self.need_mask:
            return torch.tanh(self.model(input,input[:,3:4,:,:]))
        else:
            output = self.model(input[:,0:3,:,:],input[:,3:4,:,:])
            if self.outputmask and self.final_layer:
                return (torch.tanh(output[0]),output[1])
            elif self.final_layer:
                return torch.tanh(output)
            else:
                return output

    def freeze_weighting_of_rasc(self):
        for p in self.model.rasc5.mask_attention.parameters():
            p.requires_grad = False
        for p in self.model.rasc6.mask_attention.parameters():
            p.requires_grad = False
        for p in self.model.rasc7.mask_attention.parameters():
            p.requires_grad = False
                 
        
        