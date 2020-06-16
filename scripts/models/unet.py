import torch
import torch.nn as nn
from torch.nn import init
import functools
from scripts.models.blocks import *
from scripts.models.rasc import *


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
                 attention_model=RASC,basicblock=MinimalUnet,outermostattention=False):
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
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,basicblock=MinimalUnet):
        super(UnetGenerator, self).__init__()

        # 8 for 256x256
        # 9 for 512x512
        # construct unet structure
        self.need_mask = not input_nc == output_nc

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,basicblock=basicblock) # 1
        for i in range(num_downs - 5): #3 times
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,is_attention_layer=use_inner_attention,attention_model=attention_model,basicblock=basicblock) # 8,4,2
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock) #16
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock) #32
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock, outermostattention=True) #64 
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, basicblock=basicblock, norm_layer=norm_layer) # 128

        self.model = unet_block

    def forward(self, input):
        if self.need_mask:
            return self.model(input,input[:,3:4,:,:])
        else:
            return self.model(input[:,0:3,:,:],input[:,3:4,:,:])

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

class DefaultLayerDown(nn.Module):
    def __init__(self,in_size,out_size,normalize=True, dropout=0.0):
        super(DefaultLayerDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        # if normalize:
        #     layers.append(nn.InstanceNorm2d(out_size))
        # layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)


class DefaultLayerUp(nn.Module):
    def __init__(self,in_size,out_size, dropout=0.0):
        super(DefaultLayerUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(out_size),
            # nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

class GatedDownX(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(GatedDownX, self).__init__()
        self.front = nn.Conv2d(in_size, out_size*2, 4, 2, 1, bias=False)

    def forward(self, input):
        x = self.front(input)
        x,y = torch.chunk(x,2,dim=1)
        foreground_feature = F.leaky_relu(x,0.2)
        attention = F.sigmoid(y)
        x = foreground_feature * attention 
        return x

class GatedUpX(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(GatedUpX, self).__init__()
        self.front = nn.ConvTranspose2d(in_size, out_size*2, 4, 2, 1, bias=False)

    def forward(self, input, skip_input):
        x = self.front(input)
        x, y = torch.chunk(x, 2, dim=1)
        foreground_feature = F.leaky_relu(x, 0.2)
        attention = F.sigmoid(y)
        x = foreground_feature * attention 
        x = torch.cat((x, skip_input), 1)
        return x

class GatedUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, down=GatedDownX, up=GatedUpX, ngf=48):
        super(GatedUnet, self).__init__()

        self.down1 = down(in_channels, ngf, normalize=False)
        self.down2 = down(ngf, ngf*2)
        self.down3 = down(ngf*2, ngf*4)
        self.down4 = down(ngf*4, ngf*8, dropout=0.5)
        self.down5 = down(ngf*8, ngf*8, dropout=0.5)
        self.down6 = down(ngf*8, ngf*8, dropout=0.5)
        self.down7 = down(ngf*8, ngf*8, dropout=0.5)
        self.down8 = down(ngf*8, ngf*8, normalize=False, dropout=0.5)

        self.up1 = up(ngf*8, ngf*8, dropout=0.5)
        self.up2 = up(ngf*16, ngf*8, dropout=0.5)
        self.up3 = up(ngf*16, ngf*8, dropout=0.5)
        self.up4 = up(ngf*16, ngf*8, dropout=0.5)
        self.up5 = up(ngf*16, ngf*4)
        self.up6 = up(ngf*8, ngf*2)
        self.up7 = up(ngf*4, ngf)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ngf*2, out_channels, 4, padding=1),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

class PartialCoXv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        # self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialcoXv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialCoXv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialCoXv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialCoXv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialCoXv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PCoXvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_4 = PCBActiv(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input):
        input_mask = 1 - input[:, 3:4, :, :].repeat(1,3,1,1)
        input = input[:,0:3,:,:]

        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()
