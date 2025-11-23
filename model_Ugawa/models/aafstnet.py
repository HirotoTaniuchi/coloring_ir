import torch
import torch.nn as nn
from torchvision.transforms import Grayscale
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

import kornia as K

def set_normalize(norm_type:str):
    """
    select normalization type from 'batch' or 'instance'
    norm_type : str
    """
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError(f'{norm_type} normalization is not implemented')


class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, norm_type='instance'):
        super(UNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose

        self.inc = DoubleConv(in_channels=input_nc, out_channels=64, norm_type=norm_type)
        self.down1 = Down(in_channels=64, out_channels=128, norm_type=norm_type)
        self.down2 = Down(in_channels=128, out_channels=256, norm_type=norm_type)
        self.down3 = Down(in_channels=256, out_channels=512, norm_type=norm_type)
        factor = 2 if bilinear else 1
        self.down4 = Down(in_channels=512, out_channels=1024 // factor)
        self.up1 = Up(in_channels=1024, out_channels=512 // factor, bilinear=bilinear)
        self.up2 = Up(in_channels=512, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type)
        self.up3 = Up(in_channels=256, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type)
        self.up4 = Up(in_channels=128, out_channels=64, bilinear=bilinear, norm_type=norm_type)
        self.outc = OutConv(in_channels=64, out_channels=output_nc)

    def forward(self, x):
        # downsampling
            # self.feam_onでFEAModuleを使用
            # self.verbose==Trueで詳細表示
        x1 = self.inc(x)
        if self.verbose: print(f"x1.shape = {x1.shape}")
        x2 = self.down1(x1)
        if self.verbose: print(f"x2.shape = {x2.shape}")
        x3 = self.down2(x2)
        if self.verbose: print(f"x3.shape = {x3.shape}")
        x4 = self.down3(x3)
        if self.verbose: print(f"x4.shape = {x4.shape}")
        x5 = self.down4(x4)
        if self.verbose: print(f"x5.shape = {x5.shape}")

        # upsampling
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNet_Discriminator3(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, norm_type='instance'):
        super(UNet_Discriminator3, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose

        self.inc = DoubleConv(in_channels=input_nc, out_channels=64, norm_type=norm_type)
        self.down1 = Down(in_channels=64, out_channels=128, norm_type=norm_type)
        self.down2 = Down(in_channels=128, out_channels=256, norm_type=norm_type)
        self.down3 = Down(in_channels=256, out_channels=512, norm_type=norm_type)
        factor = 2 if bilinear else 1
        self.down4 = Down(in_channels=512, out_channels=1024 // factor)
        self.up1 = Up(in_channels=1024, out_channels=512 // factor, bilinear=bilinear)
        self.up2 = Up(in_channels=512, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type)
        self.up3 = Up(in_channels=256, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type)
        self.up4 = Up(in_channels=128, out_channels=64, bilinear=bilinear, norm_type=norm_type)
        self.outc = OutConv2(in_channels=64, out_channels=1)
        self.b_outc = OutConv2(in_channels=512, out_channels=1)


    def forward(self, x):
        # downsampling
            # self.feam_onでFEAModuleを使用
            # self.verbose==Trueで詳細表示
        x1 = self.inc(x)
        if self.verbose: print(f"x1.shape = {x1.shape}")
        x2 = self.down1(x1)
        if self.verbose: print(f"x2.shape = {x2.shape}")
        x3 = self.down2(x2)
        if self.verbose: print(f"x3.shape = {x3.shape}")
        x4 = self.down3(x3)
        if self.verbose: print(f"x4.shape = {x4.shape}")
        x5 = self.down4(x4)
        if self.verbose: print(f"x5.shape = {x5.shape}")
        bottleneck_out = self.b_outc(x5)
        # upsampling
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out, bottleneck_out

""" Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type='instance'):
        super().__init__()
        norm_layer = set_normalize(norm_type)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        super().__init__()
        norm_layer = set_normalize(norm_type)
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        return self.single_conv(x)

class SingleConvProjection(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        super().__init__()
        norm_layer = set_normalize(norm_type)
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        return self.single_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4, stride=2, padding=1),
            DoubleConv(in_channels, out_channels, norm_type=norm_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type='instance'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, \
                                    mid_channels=in_channels // 2, norm_type=norm_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# SRBのEncoderブロック
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rcab_on=False, norm_type='instance'):
        super(EncoderBlock, self).__init__()
        model = []
        model += [nn.LeakyReLU(0.2)]
        
        # Enc1~3はRCABを使用、Enc4~7はRCAB無し
        if rcab_on:
            model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                      RCAB(input_nc=out_channels, output_nc=out_channels)]
        else:
            out_channels = 512
            model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)]
        norm_layer = set_normalize(norm_type=norm_type)
        model += [norm_layer(out_channels)]

        self.e_block = nn.Sequential(*model)

    def forward(self, input):
        out = self.e_block(input)
        return out

# SRBのDecoderブロック
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rcab_on=False, norm_type='instance'):
        super(DecoderBlock, self).__init__()
        model = []
        model += [nn.ReLU(inplace=False)]

        # Dec1~3はRCABを使用、Dec4~7はRCAB無し
        if rcab_on:
            model += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)]
            model += [RCAB(input_nc=out_channels, output_nc=out_channels)]
        else:
            in_channels = 1024
            model += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)]
        norm_layer = set_normalize(norm_type=norm_type)
        model += [norm_layer(out_channels)]
        
        self.d_block = nn.Sequential(*model)
    
    def forward(self, x1, x2):
        for i in range(len(x2)):
            cat = torch.cat((x1, x2[i]), dim=1)
            x1 = cat.clone()
        out = self.d_block(cat)
        return out

# SRBの最下層ブロック
class BottomBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, norm_type='instance'):
        super(BottomBlock, self).__init__()
        self.l_relu = nn.LeakyReLU(0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        norm_layer = set_normalize(norm_type=norm_type)
        self.norm = norm_layer(out_channels)
    
    def forward(self, input):
        out = self.l_relu(input)
        out = self.conv(out)
        out = self.norm(out)
        return out

# SRBをそのまま出力として用いる場合に使う
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                  nn.Tanh())

    def forward(self, x):
        return self.conv(x)

class OutConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                  nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)



######################################################################
#                            RCAB                                    #
######################################################################
class RCAB(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(RCAB, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.conv_1 = nn.Conv2d(in_channels=input_nc, out_channels=input_nc, kernel_size=3, stride=1, padding=1)
        self.l_relu = nn.LeakyReLU(0.2)
        self.conv_2 = nn.Conv2d(in_channels=input_nc, out_channels=input_nc, kernel_size=3, stride=1, padding=1)
        self.cab = CAB(input_nc=input_nc, output_nc=output_nc, reduction=32)

    def forward(self, input):
        verbose = False
        if verbose: print("---------------RCAB START---------------")
        identity = input
        if verbose: print(f"input : {input.shape}")
        cab = self.conv_1(input)
        if verbose: print(f"conv_1 : {cab.shape}")
        cab = self.l_relu(cab)
        if verbose: print(f"leaky relu : {cab.shape}")
        cab = self.conv_2(cab)
        if verbose: print(f"conv_2 : {cab.shape}")
        cab = self.cab(cab)
        if verbose: print(f"CAB out : {cab.shape}")
        out = identity * cab
        if verbose: print(f"RCAB out : {out.shape}")
        if verbose: print("---------------RCAB END---------------")
        
        return out

######################################################################
#                            CAB                                     #
######################################################################
# https://github.com/Andrew-Qibin/CoordAttention/blob/main/coordatt.py
class CAB(nn.Module):
    def __init__(self, input_nc, output_nc, reduction=32):
        super(CAB, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        middle_nc = max(8, input_nc // reduction)

        self.conv1 = nn.Conv2d(input_nc, middle_nc, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_nc)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(middle_nc, output_nc, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(middle_nc, output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        verbose = False
        if verbose: print("---------------CAB START---------------")
        identity = x
        if verbose: print(f"input : {x.shape}")
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        if verbose: print(f"pool_h : {x_h.shape}")
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        if verbose: print(f"pool_w : {x_w.shape}")

        y = torch.cat([x_h, x_w], dim=2)
        if verbose: print(f"concat : {y.shape}")
        y = self.conv1(y)
        if verbose: print(f"conv : {y.shape}")
        y = self.bn1(y)
        if verbose: print(f"bn1 : {y.shape}")
        y = self.act(y) 
        if verbose: print(f"non-linear : {y.shape}")
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        if verbose: print(f"split_h : {x_h.shape}")
        x_w = x_w.permute(0, 1, 3, 2)
        if verbose: print(f"split_w : {x_w.shape}")

        a_h = self.conv_h(x_h).sigmoid()
        if verbose: print(f"conv & sigmoid h : {a_h.shape}")
        a_w = self.conv_w(x_w).sigmoid()
        if verbose: print(f"conv & sigmoid w : {a_w.shape}")

        out = identity * a_w * a_h
        if verbose: print(f"CAB out : {out.shape}")
        if verbose: print("---------------CAB END---------------")
        
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CAB2(nn.Module):
    def __init__(self, input_nc, output_nc, reduction=32):
        super(CAB2, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        middle_nc = max(8, input_nc // reduction)

        self.conv1 = nn.Conv2d(input_nc, middle_nc, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_nc)
        self.act = h_swish2()
        
        self.conv_h = nn.Conv2d(middle_nc, output_nc, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(middle_nc, output_nc, kernel_size=1, stride=1, padding=0)
        self.l_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        verbose = False
        if verbose: print("---------------CAB START---------------")
        identity = x
        if verbose: print(f"input : {x.shape}")
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        if verbose: print(f"pool_h : {x_h.shape}")
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        if verbose: print(f"pool_w : {x_w.shape}")

        y = torch.cat([x_h, x_w], dim=2)
        if verbose: print(f"concat : {y.shape}")
        y = self.conv1(y)
        if verbose: print(f"conv : {y.shape}")
        y = self.bn1(y)
        if verbose: print(f"bn1 : {y.shape}")
        y = self.act(y) 
        if verbose: print(f"non-linear : {y.shape}")
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        if verbose: print(f"split_h : {x_h.shape}")
        x_w = x_w.permute(0, 1, 3, 2)
        if verbose: print(f"split_w : {x_w.shape}")

        a_h = self.l_relu(self.conv_h(x_h))
        if verbose: print(f"conv & relu h : {a_h.shape}")
        a_w = self.l_relu(self.conv_w(x_w))
        if verbose: print(f"conv & relu w : {a_w.shape}")

        out = identity * a_w * a_h
        if verbose: print(f"CAB out : {out.shape}")
        if verbose: print("---------------CAB END---------------")
        
        return out

class h_sigmoid2(nn.Module):
    def __init__(self, inplace=False):
        super(h_sigmoid2, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish2(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish2, self).__init__()
        self.sigmoid = h_sigmoid2(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


######################################################################
#                            AAFSTNet                                #
######################################################################
class AAFSTNet(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False, norm_type='instance'):
        super(AAFSTNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.verbose = verbose
        self.srb = SemanticReasoningBlock(input_nc=input_nc, output_nc=64, verbose=verbose, norm_type='instance')
        self.tex_transfar = TextureTransferBlock(input_nc=input_nc)
        self.tex_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        self.fab = FusionAttentionBlock(in_channels=128, out_channels=3)

    def forward(self, input):
        if self.verbose: print(f"input          : {input.shape}")
        texture = self.tex_transfar(input)
        texture = self.tex_conv(texture)
        if self.verbose: print(f"TTB            : {texture.shape}")
        srb = self.srb(input)
        if self.verbose: print(f"SRB            : {srb.shape}")
        out = torch.cat((texture, srb), dim=1)
        if self.verbose: print(f"TTB SRB concat : {out.shape}")
        out = self.fab(out)
        if self.verbose: print(f"FAB            : {out.shape}")

        return out

class TextureTransferBlock(nn.Module):
    def __init__(self, input_nc=3):
        super(TextureTransferBlock, self).__init__()
        self.input_nc = input_nc
        self.grayscale = Grayscale(num_output_channels=1)
        # ガウシアンフィルタ
        weight = [[1, 4,  6,  4,  1],
                  [4, 16, 24, 16, 4],
                  [6, 24, 36, 24, 6],
                  [4, 16, 24, 16, 4],
                  [1, 4,  6,  4,  1]]
        for i in range(5):
            weight[i][:] = [x / 256 for x in weight[i]]
        self.filter_weight = torch.cuda.FloatTensor(weight).view(1, 1, 5, 5)

    def forward(self, input):
        # Grayscale ch数が3でないなら一番上のchにエッジフィルタをかける
        if input.shape[1] != 3:
            gray = input[:, :1, ...]
        else:
            gray = self.grayscale(input)
        # apply filter
        output = gray - F.conv2d(gray, weight=Variable(self.filter_weight), stride=1, padding=2)
        return output


# https://github.com/milesial/Pytorch-UNet
class SemanticReasoningBlock(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False, srb_only=False, norm_type='instance'):
        super(SemanticReasoningBlock, self).__init__()
        self.verbose = verbose
        self.srb_only = srb_only

        self.in_conv = DoubleConv(in_channels=input_nc, out_channels=64)
        self.e_block_1 = EncoderBlock(in_channels=64, out_channels=64, rcab_on=True, norm_type=norm_type)
        self.e_block_2 = EncoderBlock(in_channels=64, out_channels=128, rcab_on=True, norm_type=norm_type)
        self.e_block_3 = EncoderBlock(in_channels=128, out_channels=256, rcab_on=True, norm_type=norm_type)
        self.e_block_4 = EncoderBlock(in_channels=256, out_channels=512, rcab_on=False, norm_type=norm_type)
        self.e_block_5 = EncoderBlock(in_channels=512, out_channels=512, rcab_on=False, norm_type=norm_type)
        self.e_block_6 = EncoderBlock(in_channels=512, out_channels=512, rcab_on=False, norm_type=norm_type)
        self.e_block_7 = EncoderBlock(in_channels=512, out_channels=512, rcab_on=False, norm_type=norm_type)
        
        self.bottom = BottomBlock()

        self.d_block_1 = DecoderBlock(in_channels=64*2 + 256 + 128, out_channels=output_nc, rcab_on=True, norm_type=norm_type)
        self.d_block_2 = DecoderBlock(in_channels=128*2 + 256, out_channels=64, rcab_on=True, norm_type=norm_type)        
        self.d_block_3 = DecoderBlock(in_channels=256*2, out_channels=128, rcab_on=True, norm_type=norm_type)
        self.d_block_4 = DecoderBlock(in_channels=512*2, out_channels=256, rcab_on=False, norm_type=norm_type)
        self.d_block_5 = DecoderBlock(in_channels=512*2, out_channels=512, rcab_on=False, norm_type=norm_type)
        self.d_block_6 = DecoderBlock(in_channels=512*2, out_channels=512, rcab_on=False, norm_type=norm_type)
        self.d_block_7 = DecoderBlock(in_channels=512*2, out_channels=512, rcab_on=False, norm_type=norm_type)

        self.out_conv = nn.Conv2d(in_channels=output_nc, out_channels=3, kernel_size=1, stride=1)

    def forward(self, input):
        # Encoder
        if self.verbose: print("---------------SRB START---------------")
        if self.verbose: print(f"input   : {input.shape}")
        out = self.in_conv(input)
        if self.verbose: print(f"in_conv : {out.shape}")
        enc1 = self.e_block_1(out)
        if self.verbose: print(f"Enc_1   : {enc1.shape}")
        enc2 = self.e_block_2(enc1)
        if self.verbose: print(f"Enc_2   : {enc2.shape}")
        enc3 = self.e_block_3(enc2)
        if self.verbose: print(f"Enc_3   : {enc3.shape}")
        enc4 = self.e_block_4(enc3)
        if self.verbose: print(f"Enc_4   : {enc4.shape}")
        enc5 = self.e_block_5(enc4)
        if self.verbose: print(f"Enc_5   : {enc5.shape}")
        enc6 = self.e_block_6(enc5)
        if self.verbose: print(f"Enc_6   : {enc6.shape}")
        enc7 = self.e_block_7(enc6)
        if self.verbose: print(f"Enc_7   : {enc7.shape}")
        bottom = self.bottom(enc7)
        if self.verbose: print(f"bottom  : {out.shape}")

        # Decoder
        dec7 = self.d_block_7(bottom, (enc7,))
        if self.verbose: print(f"Dec_7   : {dec7.shape}")
        dec6 = self.d_block_6(dec7, (enc6,))
        if self.verbose: print(f"Dec_6   : {dec6.shape}")
        dec5 = self.d_block_5(dec6, (enc5,))
        if self.verbose: print(f"Dec_5   : {dec5.shape}")
        dec4 = self.d_block_4(dec5, (enc4,))
        dec4_x4 = F.interpolate(input=dec4, scale_factor=2)
        dec4_x8 = F.interpolate(input=dec4, scale_factor=4)
        if self.verbose: print(f"Dec_4   : {dec4.shape}")
        if self.verbose: print(f"Dec_4_x4: {dec4_x4.shape}")
        if self.verbose: print(f"Dec_4_x8: {dec4_x8.shape}")
        dec3 = self.d_block_3(dec4, (enc3,))
        dec3_x4 = F.interpolate(input=dec3, scale_factor=2)
        if self.verbose: print(f"Dec_3   : {dec3.shape}")
        if self.verbose: print(f"Dec_3_x4: {dec3_x4.shape}")
        dec2 = self.d_block_2(dec3, (enc2, dec4_x4))
        if self.verbose: print(f"Dec_2   : {dec2.shape}")
        dec1 = self.d_block_1(dec2, (enc1, dec3_x4, dec4_x8))
        if self.verbose: print(f"Dec_1   : {dec1.shape}")
        if self.srb_only:
            out = self.out_conv(dec1)
        else:
            out = dec1
        if self.verbose: print(f"out_conv: {out.shape}")
        if self.verbose: print("---------------SRB END---------------")
        
        return out

class FusionAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rcab_on=True):
        super(FusionAttentionBlock, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)]
        model += [nn.LeakyReLU(0.2)]
        if rcab_on:
            model += [RCAB(input_nc=in_channels, output_nc=in_channels)]
        #model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)]
        #model += [nn.Sigmoid()]
        model += [OutConv(in_channels=in_channels, out_channels=out_channels)]
        self.fab = nn.Sequential(*model)
    
    def forward(self, input):
        out = self.fab(input)
        return out

class FusionAttentionBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, rcab_on=True):
        super(FusionAttentionBlock2, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)]
        model += [nn.LeakyReLU(0.2)]
        if rcab_on:
            model += [RCAB(input_nc=in_channels, output_nc=in_channels)]
        #model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)]
        #model += [nn.Sigmoid()]
        model += [OutConv2(in_channels=in_channels, out_channels=3)]
        self.fab = nn.Sequential(*model)
    
    def forward(self, input):
        out = self.fab(input)
        return out

class TextureTransferBlock2(nn.Module):
    def __init__(self, input_nc=3):
        super(TextureTransferBlock2, self).__init__()
        self.input_nc = input_nc
        self.grayscale = Grayscale(num_output_channels=1)
        # ラプラシアンフィルタ
        weight = [-1., -1., -1.,
                  -1., 8.,  -1.,
                  -1., -1., -1.]
        self.filter_weight = torch.FloatTensor(weight).view(1, 1, 3, 3)

    def forward(self, input):
        print(input.shape)
        # Grayscale
        gray = self.grayscale(input)
        # apply filter
        output = F.conv2d(gray, weight=Variable(self.filter_weight), stride=1, padding=1)
        return output

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_type='instance'):
        super(DoubleConv2, self).__init__()
        norm_layer = set_normalize(norm_type)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down2(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, norm_type='instance', rcab_on=True):
        super(Down2, self).__init__()
        norm_layer = set_normalize(norm_type=norm_type)
        model = [nn.MaxPool2d(4, stride=2, padding=1),
                 DoubleConv(in_channels, out_channels, norm_type=norm_type)]
        # rcab_onならRCABをモデルに追加
        if rcab_on:
            model += [RCAB(input_nc=out_channels, output_nc=out_channels)]
        model += [norm_layer(out_channels)]

        self.maxpool_conv = nn.Sequential(*model)

    def forward(self, x):
        return self.maxpool_conv(x)

class Down3(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, norm_type='instance', rcab_on=True):
        super(Down3, self).__init__()
        norm_layer = set_normalize(norm_type=norm_type)
        model = [nn.MaxPool2d(4, stride=2),
                 DoubleConv(in_channels, out_channels, norm_type=norm_type)]
        # rcab_onならRCABをモデルに追加
        if rcab_on:
            model += [RCAB(input_nc=out_channels, output_nc=out_channels)]
        model += [norm_layer(out_channels)]

        self.maxpool_conv = nn.Sequential(*model)

    def forward(self, x):
        return self.maxpool_conv(x)

class Up2(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type='instance', rcab_on=True):
        super(Up2, self).__init__()
        norm_layer = set_normalize(norm_type=norm_type)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, \
                                    mid_channels=in_channels // 2, norm_type=norm_type)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        if rcab_on:
            self.last = nn.Sequential(RCAB(input_nc=out_channels, output_nc=out_channels),
                                      norm_layer(out_channels))
        else:
            self.last = nn.Sequential(norm_layer(out_channels))


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #print(diffX, diffY)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        out = self.last(out) 
        return out


######################################################################
#                            AAFSTNet3                               #
######################################################################
class AAFSTNet3(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False, canny=False, TTB_on=True, norm_type='instance', rcab_on=False, tdga_on=True, use_resblock=True, use_global=False):
        super(AAFSTNet3, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.canny = canny
        self.verbose = verbose
        self.TTB_on = TTB_on
        self.srb = SemanticReasoningBlock3(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, tdga_on=tdga_on, use_resblock=use_resblock, norm_type=norm_type, use_global=use_global)

        self.tex_transfar = TextureTransferBlock(input_nc=input_nc)
        self.tex_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        if self.TTB_on:
            self.fab = FusionAttentionBlock(in_channels=128, out_channels=3, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=3, rcab_on=rcab_on)

    def forward(self, input):
        if self.verbose: print(f"input          : {input.shape}")
        texture = self.tex_transfar(input)
        texture = self.tex_conv(texture)
        if self.verbose: print(f"TTB            : {texture.shape}")
        srb = self.srb(input)
        if self.verbose: print(f"SRB            : {srb.shape}")
        if self.TTB_on:
            out = torch.cat((texture, srb), dim=1)
            if self.verbose: print(f"TTB SRB concat : {out.shape}")
        else:
            out = srb
        out = self.fab(out)
        if self.verbose: print(f"FAB            : {out.shape}")

        return out

class SemanticReasoningBlock3(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, rcab_on=False, tdga_on=True, norm_type='instance', use_resblock=True, n_blocks=4, use_global=False):
        super(SemanticReasoningBlock3, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose
        self.rcab_on = rcab_on 
        self.tdga_on = tdga_on
        self.use_resblock = use_resblock
        self.use_global = use_global
        
        if use_global:
            self.global_extractor = models.resnet50(pretrained=False)

        self.inc = DoubleConv2(in_channels=input_nc, out_channels=64, norm_type=norm_type)
        self.down1 = Down2(in_channels=64, out_channels=128, norm_type=norm_type, rcab_on=rcab_on)
        self.down2 = Down2(in_channels=128, out_channels=256, norm_type=norm_type, rcab_on=rcab_on)
        self.down3 = Down2(in_channels=256, out_channels=512, norm_type=norm_type, rcab_on=rcab_on)
        factor = 2 if bilinear else 1
        self.down4 = Down2(in_channels=512, out_channels=1024 // factor, norm_type=norm_type, rcab_on=rcab_on)
        self.down5 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.down6 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)

        if norm_type=='instance':
            norm_layer = nn.InstanceNorm2d
        elif norm_type=='batch':
            norm_layer = nn.BatchNorm2d

        if use_resblock:
            resblocks = []
            for _ in range(n_blocks):
                resblocks += [ResnetBlock(dim=512, norm_layer=norm_layer,
                                    use_dropout=False, use_bias=False, padding_type='reflect')]
            self.resblock = nn.Sequential(*resblocks)

        # TDGAモジュールを追加
        if tdga_on:
            self.tdga = PGAResBlockv4k3(in_dim=512, use_bias=False, use_map=False, norm_layer=norm_layer)

        self.up1 = Up2(in_channels=1024, out_channels=512 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up2 = Up2(in_channels=512, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up3 = Up2(in_channels=256, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up4 = Up2(in_channels=128, out_channels=64, bilinear=bilinear, norm_type=norm_type)
        self.outc = OutConv(in_channels=64, out_channels=output_nc)

    def forward(self, x):
        # downsampling
            # self.rcab_onでRCABを使用
            # self.tdga_onでTDGAを使用
            # self.verbose==Trueで詳細表示
        x1 = self.inc(x)
        x2 = self.down1(x1.clone())
        x3 = self.down2(x2.clone())
        x4 = self.down3(x3.clone())
        if self.tdga_on:
            x4 = self.tdga(x4)

        # ResnetBlockを使用
        if self.use_resblock:
            x5 = self.resblock(x4)
        else:
            # または普通にDownsample(TDGAと併用不可能)
            x5_ = self.down4(x4.clone())
            x5_ = self.down5(x5_)
            x5 = self.down6(x5_)
        
        if self.tdga_on:
            x5 = self.tdga(x5)
        if self.verbose: print(f"inc   : {x1.shape}")
        if self.verbose: print(f"down1 : {x2.shape}")
        if self.verbose: print(f"down2 : {x3.shape}")
        if self.verbose: print(f"down3 : {x4.shape}")
        if self.verbose: print(f"down4 : {x5.shape}")

        # upsampling
        x = self.up1(x5, x4)
        if self.verbose: print(f"up1   : {x.shape}")
        x = self.up2(x, x3)
        if self.verbose: print(f"up2   : {x.shape}")
        x = self.up3(x, x2)
        if self.verbose: print(f"up3   : {x.shape}")
        x = self.up4(x, x1)
        if self.verbose: print(f"up4   : {x.shape}")
        x = self.outc(x)
        if self.verbose: print(f"outc  : {x.shape}")
        return x


# Define a top-down guided attention module.
# Created by FuyaLuo : https://github.com/FuyaLuo/PearlGAN/
class PGAResBlockv4k3(nn.Module):
    def __init__(self, in_dim, use_bias, use_map, norm_layer=nn.BatchNorm2d):
        super(PGAResBlockv4k3, self).__init__()

        self.width = in_dim // 4
        self.bottlenec1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())

        self.ds1 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.ds2 = nn.AvgPool2d(kernel_size=3, stride=4)
        self.ds3 = nn.AvgPool2d(kernel_size=3, stride=8)
        self.ds4 = nn.AvgPool2d(kernel_size=3, stride=16)

        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=use_bias), norm_layer(in_dim), nn.PReLU())

        self.softmax  = nn.Softmax(dim=-1) #

        self.use_map = use_map

    def forward(self, input):
        #print(f"input.shape={input.shape}")
        b, c, h, w = input.size()

        input_fea = self.conv(input)
        spx = torch.split(input_fea, self.width, 1)
        #print(f"spx[0].shape:{spx[0].shape}, spx[1].shape:{spx[1].shape}, spx[2].shape:{spx[2].shape}, spx[3].shape:{spx[3].shape}")
        fea_ds1 = self.ds1(spx[0])
        fea_ds2 = self.ds2(spx[1])
        fea_ds3 = self.ds3(spx[2])
        fea_ds4 = self.ds4(spx[3])

        att_conv1 = self.bottlenec1(fea_ds4)
        att_map1_us = F.interpolate(att_conv1, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g1 = F.interpolate(att_conv1, size=(fea_ds3.size(2), fea_ds3.size(3)), mode='bilinear', align_corners=False)
        
        fea_att1 = att_map_g1.expand_as(fea_ds3).mul(fea_ds3) + fea_ds3
        att_conv2 = self.bottlenec2(fea_att1)
        att_map2_us = F.interpolate(att_conv2, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g2 = F.interpolate(att_conv2, size=(fea_ds2.size(2), fea_ds2.size(3)), mode='bilinear', align_corners=False)

        fea_att2 = att_map_g2.expand_as(fea_ds2).mul(fea_ds2) + fea_ds2
        att_conv3 = self.bottlenec3(fea_att2)
        att_map3_us = F.interpolate(att_conv3, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g3 = F.interpolate(att_conv3, size=(fea_ds1.size(2), fea_ds1.size(3)), mode='bilinear', align_corners=False)
        
        fea_att3 = att_map_g3.expand_as(fea_ds1).mul(fea_ds1) + fea_ds1
        att_conv4 = self.bottlenec4(fea_att3)
        att_map4_us = F.interpolate(att_conv4, size=(h, w), mode='bilinear', align_corners=False)
        
        y1 = att_map4_us.expand_as(spx[0]).mul(spx[0])
        y2 = att_map3_us.expand_as(spx[1]).mul(spx[1])
        y3 = att_map2_us.expand_as(spx[2]).mul(spx[2])
        y4 = att_map1_us.expand_as(spx[3]).mul(spx[3])

        out = torch.cat((y1, y2, y3, y4), 1) + input
        #print(f"output.shape={out.shape}")
        
        if self.use_map:
            return out, att_map1_us, att_map2_us, att_map3_us, att_map4_us
        else: 
            return out

# Define a resnet block
# Copied from networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                        norm_layer(dim),
                        nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                        norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class AAFSTNet4(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False, canny=False, TTB=True, norm_type='instance', rcab_on=True):
        super(AAFSTNet4, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.canny = canny
        self.verbose = verbose
        self.TTB = TTB
        self.srb = SemanticReasoningBlock4(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, norm_type='instance')
        self.tex_transfar = TextureTransferBlock(input_nc=input_nc)
        self.tex_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        if self.TTB:
            self.fab = FusionAttentionBlock(in_channels=128, out_channels=3, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=3, rcab_on=rcab_on)
            
    def forward(self, input):
        if self.verbose: print(f"input          : {input.shape}")
        texture = self.tex_transfar(input)
        texture = self.tex_conv(texture)
        if self.verbose: print(f"TTB            : {texture.shape}")
        srb, loss_reg = self.srb(input)
        if self.verbose: print(f"SRB            : {srb.shape}")
        if self.TTB:
            out = torch.cat((texture, srb), dim=1)
            if self.verbose: print(f"TTB SRB concat : {out.shape}")
        else:
            out = srb
        out = self.fab(out)
        if self.verbose: print(f"FAB            : {out.shape}")

        return out, loss_reg

class SemanticReasoningBlock4(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, rcab_on=True, norm_type='instance'):
        super(SemanticReasoningBlock4, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose
        self.rcab_on = rcab_on 
        self.edge = TextureTransferBlock(input_nc=input_nc)

        self.inc = DoubleConv2(in_channels=input_nc, out_channels=64, norm_type=norm_type)
        self.down1 = Down2(in_channels=64, out_channels=128, norm_type=norm_type, rcab_on=rcab_on)
        self.down2 = Down2(in_channels=128, out_channels=256, norm_type=norm_type, rcab_on=rcab_on)
        self.down3 = Down2(in_channels=256, out_channels=512, norm_type=norm_type, rcab_on=rcab_on)
        factor = 2 if bilinear else 1
        self.down4 = Down2(in_channels=512, out_channels=1024 // factor, norm_type=norm_type, rcab_on=rcab_on)
        self.down5 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.down6 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.up1 = Up2(in_channels=1024, out_channels=512 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up2 = Up2(in_channels=512, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up3 = Up2(in_channels=256, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up4 = Up2(in_channels=128, out_channels=64, bilinear=bilinear, norm_type=norm_type)
        self.outc = OutConv(in_channels=64, out_channels=output_nc)
        
         # エッジ強化ネットワーク(incとdown4の後に入れる)
        # 実装はresblock
        
        edge_enhance1 = []
        edge_enhance1 += [DoubleConv2(in_channels=65, out_channels=64, norm_type=norm_type), 
                          DoubleConv2(in_channels=64, out_channels=64, norm_type=norm_type)]
        self.edge_enhance1 = nn.Sequential(*edge_enhance1)

        edge_enhance2 = []
        edge_enhance2 += [DoubleConv2(in_channels=513, out_channels=512, norm_type=norm_type), 
                          DoubleConv2(in_channels=512, out_channels=512, norm_type=norm_type)]
        self.edge_enhance2 = nn.Sequential(*edge_enhance2)
        
    def forward(self, x):
        loss_reg = 0
        # downsampling
            # self.rcab_onでRCABを使用
            # self.verbose==Trueで詳細表示
        x1 = self.inc(x)

        # edge enhance network
        edge_map = self.edge(x.clone())
        e1 = torch.cat((x1.clone(), F.upsample_bilinear(edge_map, size=(x1.shape[2],x1.shape[3]))), dim=1)
        #e1 = torch.cat((x1, torch.zeros((1,1,x1.shape[2],x1.shape[3]), device='cuda:0')), dim=1)
        e1_ = self.edge_enhance1(e1)
        x1_ = x1 + e1_
        loss_reg += torch.norm(e1_)
        x2 = self.down1(x1_)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        e2 = torch.cat((x4.clone(), F.upsample_bilinear(edge_map, size=(x4.shape[2],x4.shape[3]))), dim=1)
        #e2 = torch.cat((x4, torch.zeros((1,1,x4.shape[2],x4.shape[3]), device='cuda:0')), dim=1)
        e2_ = self.edge_enhance2(e2)
        x4_ = x4 + e2_
        loss_reg += torch.norm(e2_)
        x5__ = self.down4(x4)
        x5_ = self.down5(x5__)
        x5 = self.down6(x5_)
        if self.verbose: print(f"inc   : {x1.shape}")
        if self.verbose: print(f"down1 : {x2.shape}")
        if self.verbose: print(f"down2 : {x3.shape}")
        if self.verbose: print(f"down3 : {x4.shape}")
        if self.verbose: print(f"down4 : {x5.shape}")

        # upsampling
        x = self.up1(x5, x4)
        if self.verbose: print(f"up1   : {x.shape}")
        x = self.up2(x, x3)
        if self.verbose: print(f"up2   : {x.shape}")
        x = self.up3(x, x2)
        if self.verbose: print(f"up3   : {x.shape}")
        x = self.up4(x, x1)
        if self.verbose: print(f"up4   : {x.shape}")
        x = self.outc(x)
        if self.verbose: print(f"outc  : {x.shape}")
        return x, loss_reg

class AAFSTNet2(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False, canny=False, TTB=True, norm_type='instance', rcab_on=True):
        super(AAFSTNet2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.canny = canny
        self.verbose = verbose
        self.TTB = TTB
        self.srb = SemanticReasoningBlock2(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, norm_type='instance')
        self.tex_transfar = TextureTransferBlock(input_nc=input_nc)
        self.tex_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        if self.TTB:
            self.fab = FusionAttentionBlock(in_channels=128, out_channels=output_nc, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)

    def forward(self, input):
        if self.verbose: print(f"input          : {input.shape}")
        edge = self.tex_transfar(input)
        texture = self.tex_conv(edge)
        if self.verbose: print(f"TTB            : {texture.shape}")
        srb = self.srb(input)
        if self.verbose: print(f"SRB            : {srb.shape}")
        if self.TTB:
            out = torch.cat((texture, srb), dim=1)
            if self.verbose: print(f"TTB SRB concat : {out.shape}")
        else:
            out = srb
        out = self.fab(out)
        if self.verbose: print(f"FAB            : {out.shape}")
        return out, edge

class SemanticReasoningBlock2(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, rcab_on=True, norm_type='instance'):
        super(SemanticReasoningBlock2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose
        self.rcab_on = rcab_on 
        
        self.inc = DoubleConv2(in_channels=input_nc, out_channels=64, norm_type=norm_type)
        self.down1 = Down3(in_channels=64, out_channels=128, norm_type=norm_type, rcab_on=rcab_on)
        self.down2 = Down3(in_channels=128, out_channels=256, norm_type=norm_type, rcab_on=rcab_on)
        self.down3 = Down3(in_channels=256, out_channels=512, norm_type=norm_type, rcab_on=rcab_on)
        factor = 2 if bilinear else 1
        self.down4 = Down3(in_channels=512, out_channels=1024 // factor, norm_type=norm_type, rcab_on=rcab_on)
        self.down5 = Down3(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.down6 = Down3(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.up1 = Up2(in_channels=1024, out_channels=512 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up2 = Up2(in_channels=512, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up3 = Up2(in_channels=256, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up4 = Up2(in_channels=128, out_channels=64, bilinear=bilinear, norm_type=norm_type)
        self.outc = OutConv(in_channels=64, out_channels=output_nc)

    def forward(self, x):
        # downsampling
            # self.rcab_onでRCABを使用
            # self.verbose==Trueで詳細表示

        x1 = self.inc(x)
        x2 = self.down1(x1.clone())
        x3 = self.down2(x2.clone())
        x4 = self.down3(x3.clone())
        x5_ = self.down4(x4.clone())
        x5_ = self.down5(x5_)
        x5 = self.down6(x5_)
        if self.verbose: print(f"inc   : {x1.shape}")
        if self.verbose: print(f"down1 : {x2.shape}")
        if self.verbose: print(f"down2 : {x3.shape}")
        if self.verbose: print(f"down3 : {x4.shape}")
        if self.verbose: print(f"down4 : {x5.shape}")

        # upsampling
        x = self.up1(x5, x4)
        if self.verbose: print(f"up1   : {x.shape}")
        x = self.up2(x, x3)
        if self.verbose: print(f"up2   : {x.shape}")
        x = self.up3(x, x2)
        if self.verbose: print(f"up3   : {x.shape}")
        x = self.up4(x, x1)
        if self.verbose: print(f"up4   : {x.shape}")
        x = self.outc(x)
        if self.verbose: print(f"outc  : {x.shape}")
        return x

class SemanticReasoningBlock2Sobel(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, rcab_on=True, norm_type='instance'):
        super(SemanticReasoningBlock2Sobel, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose
        self.rcab_on = rcab_on 
        
        n_edge_channels = 1
        self.resize = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.inc = DoubleConv2(in_channels=input_nc+n_edge_channels, out_channels=64, norm_type=norm_type)
        self.down1 = Down2(in_channels=64+n_edge_channels, out_channels=128, norm_type=norm_type, rcab_on=rcab_on)
        self.down2 = Down2(in_channels=128+n_edge_channels, out_channels=256, norm_type=norm_type, rcab_on=rcab_on)
        self.down3 = Down2(in_channels=256+n_edge_channels, out_channels=512, norm_type=norm_type, rcab_on=rcab_on)
        factor = 2 if bilinear else 1
        self.down4 = Down2(in_channels=512, out_channels=1024 // factor, norm_type=norm_type, rcab_on=rcab_on)
        self.down5 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.down6 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.up1 = Up2(in_channels=1024, out_channels=512 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up2 = Up2(in_channels=512, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up3 = Up2(in_channels=256, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up4 = Up2(in_channels=128, out_channels=64, bilinear=bilinear, norm_type=norm_type)
        self.outc = OutConv(in_channels=64, out_channels=output_nc)

    def forward(self, x, sod_map=None):
        # downsampling
            # self.rcab_onでRCABを使用
            # self.verbose==Trueで詳細表示
        if sod_map==None:
            edge_0 = K.color.rgb_to_grayscale((K.filters.sobel(x)))
        else:
            edge_0 = sod_map
        edge_1 = edge_0
        edge_2 = self.resize(edge_1)
        #print(edge_2.shape)
        edge_3 = self.resize(edge_2)
        #print(f"x={x.shape}, edge_0={edge_0.shape}")
        x1 = self.inc(torch.cat((x,edge_0), dim=1))
        x2 = self.down1(torch.cat((x1.clone(),edge_1), dim=1))
        #print(x2.shape)
        x3 = self.down2(torch.cat((x2.clone(),edge_2), dim=1))
        x4 = self.down3(torch.cat((x3.clone(),edge_3), dim=1))
        x5_ = self.down4(x4.clone())
        x5_ = self.down5(x5_)
        x5 = self.down6(x5_)
        if self.verbose: print(f"inc   : {x1.shape}")
        if self.verbose: print(f"down1 : {x2.shape}")
        if self.verbose: print(f"down2 : {x3.shape}")
        if self.verbose: print(f"down3 : {x4.shape}")
        if self.verbose: print(f"down4 : {x5.shape}")

        # upsampling
        x = self.up1(x5, x4)
        if self.verbose: print(f"up1   : {x.shape}")
        x = self.up2(x, x3)
        if self.verbose: print(f"up2   : {x.shape}")
        x = self.up3(x, x2)
        if self.verbose: print(f"up3   : {x.shape}")
        x = self.up4(x, x1)
        if self.verbose: print(f"up4   : {x.shape}")
        x = self.outc(x)
        if self.verbose: print(f"outc  : {x.shape}")
        return x

"""class AAFSTNet2Sobel(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False, sobel=True, TTB=False, norm_type='instance', rcab_on=True):
        super(AAFSTNet2Sobel, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.sobel = sobel
        self.verbose = verbose
        self.TTB = TTB
        self.srb = SemanticReasoningBlock2Sobel(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, norm_type='instance')
        if self.sobel:
            self.tex_transfar = K.filters.sobel
        else:
            self.tex_transfar = TextureTransferBlock(input_nc=input_nc)
        self.tex_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        if self.TTB:
            self.fab = FusionAttentionBlock(in_channels=128, out_channels=output_nc, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)

    def forward(self, input):
        if self.verbose: print(f'input          : {input.shape}')
        edge = K.color.rgb_to_grayscale(self.tex_transfar(input))
        texture = self.tex_conv(edge)
        
        if self.verbose: print(f'TTB            : {texture.shape}')
        srb = self.srb(input)
        if self.verbose: print(f'SRB            : {srb.shape}')
        if self.TTB:
            out = torch.cat((texture, srb), dim=1)
            if self.verbose: print(f'TTB SRB concat : {out.shape}')
        else:
            out = srb
        out = self.fab(out)
        if self.verbose: print(f'FAB            : {out.shape}')
        return out, edge
"""
class AAFSTNet2Sobel(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False, sobel=True, TTB=False, norm_type='instance', rcab_on=True, colorspace='RGB'):
        super(AAFSTNet2Sobel, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.sobel = sobel
        self.verbose = verbose
        self.TTB = TTB
        self.srb = SemanticReasoningBlock2Sobel(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, norm_type='instance')
        self.colorspace=colorspace
        if self.sobel:
            self.tex_transfar = K.filters.sobel
        else:
            self.tex_transfar = TextureTransferBlock(input_nc=input_nc)
        self.tex_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        """if self.colorspace=='Lab':
            self.fab = FusionAttentionBlock(in_channels=67, out_channels=output_nc, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)"""
        self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)

    def forward(self, input, sod_map=None):
        if self.verbose: print(f'input          : {input.shape}')
        if sod_map==None:
            edge = K.color.rgb_to_grayscale(self.tex_transfar(input))
        else:
            edge = sod_map
        #texture = self.tex_conv(edge)
        #if self.verbose: print(f'TTB            : {texture.shape}')
        srb = self.srb(input, sod_map)
        if self.verbose: print(f'SRB            : {srb.shape}')
        """if self.colorspace=='Lab':
            out = torch.cat((input, srb), dim=1)
            if self.verbose: print(f'TTB SRB concat : {out.shape}')
        else:
            out = srb"""
        out = self.fab(srb)
        if self.verbose: print(f'FAB            : {out.shape}')
        return out, edge


class SemanticReasoningBlock2Seg(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, rcab_on=True, norm_type='instance', is_input_encoder=True, is_input_decoder=False):
        super(SemanticReasoningBlock2Seg, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose
        self.rcab_on = rcab_on 
        self.is_input_decoder = is_input_decoder
        self.is_input_encoder = is_input_encoder

        if is_input_encoder:
            enc_seg_channels = 64
        else:
            enc_seg_channels = 0

        if is_input_decoder:
            dec_seg_channels = 3
        else:
            dec_seg_channels = 0
        
        self.resize = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.inc = DoubleConv2(in_channels=input_nc+enc_seg_channels, out_channels=64, norm_type=norm_type)
        self.down1 = Down2(in_channels=64+enc_seg_channels, out_channels=128, norm_type=norm_type, rcab_on=rcab_on)
        self.down2 = Down2(in_channels=128+enc_seg_channels, out_channels=256, norm_type=norm_type, rcab_on=rcab_on)
        self.down3 = Down2(in_channels=256+enc_seg_channels, out_channels=512, norm_type=norm_type, rcab_on=rcab_on)
        factor = 2 if bilinear else 1
        self.down4 = Down2(in_channels=512, out_channels=1024 // factor, norm_type=norm_type, rcab_on=rcab_on)
        self.down5 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.down6 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.up1 = Up2(in_channels=1024+dec_seg_channels, out_channels=512 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up2 = Up2(in_channels=512+dec_seg_channels, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up3 = Up2(in_channels=256+dec_seg_channels, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up4 = Up2(in_channels=128, out_channels=64, bilinear=bilinear, norm_type=norm_type)
        self.outc = OutConv(in_channels=64, out_channels=output_nc)
        self.edge_conv = DoubleConv(in_channels=3, out_channels=64, norm_type=norm_type)

    def forward(self, x, segmap):
        # downsampling
            # self.rcab_onでRCABを使用
            # self.verbose==Trueで詳細表示
        edge_0 = segmap
        edge_0 = self.edge_conv(edge_0)
        edge_1 = edge_0
        edge_2 = self.resize(edge_1)
        #print(edge_2.shape)
        edge_3 = self.resize(edge_2)
        #print(f"x={x.shape}, edge_0={edge_0.shape}")
        if self.is_input_encoder:
            x1_in = torch.cat((x,edge_0), dim=1)
        else:
            x1_in = x
        x1 = self.inc(x1_in)
        if self.is_input_encoder:
            x2_in = torch.cat((x1.clone(),edge_1), dim=1)
        else:
            x2_in = x1
        x2 = self.down1(x2_in)
        #print(x2.shape)
        if self.is_input_encoder:
            x3_in = torch.cat((x2.clone(),edge_2), dim=1)
        else:
            x3_in = x1
        x3 = self.down2(x3_in)
        if self.is_input_encoder:
            x4_in = torch.cat((x3.clone(),edge_3), dim=1)
        else:
            x4_in = x1
        x4 = self.down3(x4_in)
        x5_ = self.down4(x4.clone())
        x5_ = self.down5(x5_)
        x5 = self.down6(x5_)
        if self.verbose: print(f"inc   : {x1.shape}")
        if self.verbose: print(f"down1 : {x2.shape}")
        if self.verbose: print(f"down2 : {x3.shape}")
        if self.verbose: print(f"down3 : {x4.shape}")
        if self.verbose: print(f"down4 : {x5.shape}")

        # upsampling
        x = self.up1(x5, x4)
        if self.verbose: print(f"up1   : {x.shape}")
        x = self.up2(x, x3)
        if self.verbose: print(f"up2   : {x.shape}")
        x = self.up3(x, x2)
        if self.verbose: print(f"up3   : {x.shape}")
        x = self.up4(x, x1)
        if self.verbose: print(f"up4   : {x.shape}")
        x = self.outc(x)
        if self.verbose: print(f"outc  : {x.shape}")
        return x


class AAFSTNet2Seg(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False,  norm_type='instance', rcab_on=True, colorspace='RGB', is_input_decoder=False):
        super(AAFSTNet2Seg, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.verbose = verbose
        self.is_input_decoder = is_input_decoder
        self.srb = SemanticReasoningBlock2Seg(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, norm_type='instance', is_input_decoder=self.is_input_decoder)
        self.colorspace=colorspace
        """if self.colorspace=='Lab':
            self.fab = FusionAttentionBlock(in_channels=67, out_channels=output_nc, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)"""
        self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)

    def forward(self, input, segmap):
        if self.verbose: print(f'input          : {input.shape}')
        out_seg = segmap
        srb = self.srb(input, segmap)
        if self.verbose: print(f'SRB            : {srb.shape}')
        """if self.colorspace=='Lab':
            out = torch.cat((input, srb), dim=1)
            if self.verbose: print(f'TTB SRB concat : {out.shape}')
        else:
            out = srb"""
        out = self.fab(srb)
        if self.verbose: print(f'FAB            : {out.shape}')
        return out, out_seg

class AAFSTNet2Seg2(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False,  norm_type='instance', rcab_on=True, colorspace='RGB', is_input_decoder=False):
        super(AAFSTNet2Seg2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.verbose = verbose
        self.is_input_decoder = is_input_decoder
        self.srb = SemanticReasoningBlock2Seg(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, norm_type='instance', is_input_decoder=self.is_input_decoder)
        self.colorspace=colorspace
        """if self.colorspace=='Lab':
            self.fab = FusionAttentionBlock(in_channels=67, out_channels=output_nc, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)"""
        self.fab1 = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)
        self.fab2 = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)

    def forward(self, input, segmap):
        if self.verbose: print(f'input          : {input.shape}')
        out_seg = segmap
        srb = self.srb(input, segmap)
        if self.verbose: print(f'SRB            : {srb.shape}')
        """if self.colorspace=='Lab':
            out = torch.cat((input, srb), dim=1)
            if self.verbose: print(f'TTB SRB concat : {out.shape}')
        else:
            out = srb"""
        out1 = self.fab1(srb)
        out2 = self.fab2(srb)
        if self.verbose: print(f'FAB            : {out.shape}')
        return out1, out2, out_seg

class SemanticReasoningBlock2andSegNet(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, rcab_on=True, norm_type='instance', is_input_encoder=False, is_input_decoder=False):
        super(SemanticReasoningBlock2andSegNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose
        self.rcab_on = rcab_on 
        self.is_input_decoder = is_input_decoder
        self.is_input_encoder = is_input_encoder

        if is_input_encoder:
            enc_seg_channels = [0, 128, 256, 512]
        else:
            enc_seg_channels = [0,0,0,0]

        if is_input_decoder:
            dec_seg_channels = [0, 128, 256, 512]
        else:
            dec_seg_channels = 0

        self.resize = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.inc = DoubleConv2(in_channels=input_nc+enc_seg_channels[0], out_channels=64, norm_type=norm_type)
        self.down1 = Down2(in_channels=64+enc_seg_channels[0], out_channels=128, norm_type=norm_type, rcab_on=rcab_on)
        self.down2 = Down2(in_channels=128+enc_seg_channels[1], out_channels=256, norm_type=norm_type, rcab_on=rcab_on)
        self.down3 = Down2(in_channels=256+enc_seg_channels[2], out_channels=512, norm_type=norm_type, rcab_on=rcab_on)
        factor = 2 if bilinear else 1
        self.down4 = Down2(in_channels=512+enc_seg_channels[3], out_channels=1024 // factor, norm_type=norm_type, rcab_on=rcab_on)
        self.down5 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.down6 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.up1 = Up2(in_channels=1024+dec_seg_channels, out_channels=512 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up2 = Up2(in_channels=512+dec_seg_channels, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up3 = Up2(in_channels=256+dec_seg_channels, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up4 = Up2(in_channels=128, out_channels=output_nc, bilinear=bilinear, norm_type=norm_type)
        self.outc = OutConv(in_channels=64, out_channels=output_nc)
        self.seg_conv1 = DoubleConv(in_channels=256, out_channels=128, norm_type=norm_type)
        self.seg_conv2 = DoubleConv(in_channels=512, out_channels=256, norm_type=norm_type)
        self.seg_conv3 = DoubleConv(in_channels=1024, out_channels=512, norm_type=norm_type)


    def forward(self, x, seg_feature, input_size):
        # downconv
            # self.rcab_onでRCABを使用
            # self.verbose==Trueで詳細表示
        input_h = input_size[0]
        input_w = input_size[1]

        # seg_featureのサイズを各レイヤに合わせる
        if self.is_input_encoder or self.is_input_decoder:
            #print(f"low_level.shape={seg_feature['low_level'].shape}")
            #print(f"low_level2.shape={seg_feature['low_level2'].shape}")
            #print(f"low_level3.shape={seg_feature['low_level3'].shape}")
            seg_feature['low_level'] = F.interpolate(self.seg_conv1(seg_feature['low_level']), size=(input_h//2, input_w//2))
            seg_feature['low_level2'] = F.interpolate(self.seg_conv2(seg_feature['low_level2']), size=(input_h//4, input_w//4))
            seg_feature['low_level3'] = F.interpolate(self.seg_conv3(seg_feature['low_level3']), size=(input_h//8, input_w//8))

        x1_in = x
        x1 = self.inc(x1_in)

        x2_in = x1.clone()
        x2 = self.down1(x2_in)
        if self.is_input_encoder:
            x3_in = torch.cat((x2, seg_feature['low_level']), dim=1)
        else:
            x3_in = x2

        x3 = self.down2(x3_in)
        if self.is_input_encoder:
            x4_in = torch.cat((x3, seg_feature['low_level2']), dim=1)
        else:
            x4_in = x3

        x4 = self.down3(x4_in)
        if self.is_input_encoder:
            x5_in = torch.cat((x4, seg_feature['low_level3']), dim=1)
        else:
            x5_in = x4
        
        x5_ = self.down4(x5_in)
        x5 = self.down5(x5_)
        #x5 = self.down6(x5_)

        
        if self.verbose: print(f"inc   : {x1.shape}")
        if self.verbose: print(f"down1 : {x2.shape}")
        if self.verbose: print(f"down2 : {x3.shape}")
        if self.verbose: print(f"down3 : {x4.shape}")
        if self.verbose: print(f"down4 : {x5.shape}")

        # upconv
        out= self.up1(x5, x4)
        if self.verbose: print(f"up1   : {out.shape}")
        out= self.up2(out, x3)
        if self.verbose: print(f"up2   : {out.shape}")
        out= self.up3(out, x2)
        if self.verbose: print(f"up3   : {out.shape}")
        out= self.up4(out, x1)
        if self.verbose: print(f"up4   : {out.shape}")
        out= self.outc(out)
        if self.verbose: print(f"outc  : {out.shape}")
        return out



class AAFSTNet2andSegNet(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False,  norm_type='instance', rcab_on=True, colorspace='RGB', is_input_encoder=False, is_input_decoder=False):
        super(AAFSTNet2andSegNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.verbose = verbose
        self.srb = SemanticReasoningBlock2andSegNet(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, norm_type='instance', is_input_encoder=is_input_encoder, is_input_decoder=is_input_decoder)
        self.colorspace=colorspace
        """if self.colorspace=='Lab':
            self.fab = FusionAttentionBlock(in_channels=67, out_channels=output_nc, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)"""
        self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)

    def forward(self, input, seg_feature, input_size, seg_pred=None):
        if self.verbose: print(f'input          : {input.shape}')
        srb = self.srb(input, seg_feature, input_size)
        if self.verbose: print(f'SRB            : {srb.shape}')
        """if self.colorspace=='Lab':
            out = torch.cat((input, srb), dim=1)
            if self.verbose: print(f'TTB SRB concat : {out.shape}')
        else:
            out = srb"""
        # seg_predがある場合はseg_predをsrbと共にFABに入力
        #if seg_pred is not None:
        #    srb = torch.cat((srb, seg_pred), dim=1)

        out = self.fab(srb)
        if self.verbose: print(f'FAB            : {out.shape}')
        return out


class AAFSTNet2andSegNet2(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False,  norm_type='instance', rcab_on=True, colorspace='RGB', is_input_encoder=False, is_input_decoder=False, seg_num_classes=0, use_rgb_map=False):
        super(AAFSTNet2andSegNet2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.fab_additional_channels = seg_num_classes
        self.verbose = verbose
        self.srb = SemanticReasoningBlock2andSegNet2(input_nc=input_nc, output_nc=64, verbose=verbose, rcab_on=rcab_on, norm_type='instance', is_input_encoder=is_input_encoder, is_input_decoder=is_input_decoder, use_rgb_map=use_rgb_map)
        self.colorspace=colorspace
        """if self.colorspace=='Lab':
            self.fab = FusionAttentionBlock(in_channels=67, out_channels=output_nc, rcab_on=rcab_on)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=output_nc, rcab_on=rcab_on)"""
        self.fab = FusionAttentionBlock(in_channels=64+seg_num_classes, out_channels=output_nc, rcab_on=rcab_on)

    def forward(self, input, seg_feature, input_size, seg_pred=None):
        if self.verbose: print(f'input          : {input.shape}')
        srb = self.srb(input, seg_feature, input_size)
        if self.verbose: print(f'SRB            : {srb.shape}')
        """if self.colorspace=='Lab':
            out = torch.cat((input, srb), dim=1)
            if self.verbose: print(f'TTB SRB concat : {out.shape}')
        else:
            out = srb"""
        # seg_predがある場合はseg_predをsrbと共にFABに入力
        if self.fab_additional_channels != 0:
            srb = torch.cat((srb, seg_pred), dim=1)

        out = self.fab(srb)
        if self.verbose: print(f'FAB            : {out.shape}')
        return out

class SemanticReasoningBlock2andSegNet2(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, rcab_on=True, norm_type='instance', is_input_encoder=False, is_input_decoder=False, use_rgb_map=False):
        super(SemanticReasoningBlock2andSegNet2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose
        self.rcab_on = rcab_on
        self.is_input_decoder = is_input_decoder
        self.is_input_encoder = is_input_encoder
        self.use_rgb_map = use_rgb_map

        if is_input_encoder:
            if self.use_rgb_map:
                enc_seg_channels = [0, 64,64,64]
            else:
                enc_seg_channels = [0, 128, 256, 512]
        else:
            enc_seg_channels = [0,0,0,0]

        if is_input_decoder:
            if self.use_rgb_map:
                dec_seg_channels = [0, 64,64,64]
            else:
                dec_seg_channels = [0, 128, 256, 512]
        else:
            dec_seg_channels = 0

        self.resize = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.inc = DoubleConv2(in_channels=input_nc+enc_seg_channels[0], out_channels=64, norm_type=norm_type)
        self.down1 = Down2(in_channels=64+enc_seg_channels[0], out_channels=128, norm_type=norm_type, rcab_on=rcab_on)
        self.down2 = Down2(in_channels=128+enc_seg_channels[1], out_channels=256, norm_type=norm_type, rcab_on=rcab_on)
        self.down3 = Down2(in_channels=256+enc_seg_channels[2], out_channels=512, norm_type=norm_type, rcab_on=rcab_on)
        factor = 2 if bilinear else 1
        self.down4 = Down2(in_channels=512+enc_seg_channels[3], out_channels=1024 // factor, norm_type=norm_type, rcab_on=rcab_on)
        #self.down5 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        #self.down6 = Down2(in_channels=512, out_channels=512, norm_type=norm_type, rcab_on=False)
        self.bottleneck = DoubleConv(in_channels=512, out_channels=512, norm_type=norm_type)
        self.up1 = Up2(in_channels=1024+dec_seg_channels, out_channels=512 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up2 = Up2(in_channels=512+dec_seg_channels, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up3 = Up2(in_channels=256+dec_seg_channels, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type, rcab_on=rcab_on)
        self.up4 = Up2(in_channels=128, out_channels=64, bilinear=bilinear, norm_type=norm_type)
        #self.outc = OutConv(in_channels=64, out_channels=output_nc)
        if self.use_rgb_map:
            self.seg_conv1 = SingleConv(in_channels=3, out_channels=64, norm_type=norm_type)
        else:
            self.seg_conv1 = SingleConv(in_channels=256, out_channels=128, norm_type=norm_type)
            self.seg_conv2 = SingleConv(in_channels=512, out_channels=256, norm_type=norm_type)
            self.seg_conv3 = SingleConv(in_channels=1024, out_channels=512, norm_type=norm_type)


    def forward(self, x, seg_feature, input_size):
        # downconv
            # self.rcab_onでRCABを使用
            # self.verbose==Trueで詳細表示
        input_h = input_size[0]
        input_w = input_size[1]

        # seg_featureのサイズを各レイヤに合わせる
        if self.is_input_encoder or self.is_input_decoder:
            if self.use_rgb_map:
                seg_feature = self.seg_conv1(seg_feature)
                seg_feature_1 = F.interpolate(seg_feature, size=(input_h//2, input_w//2))
                seg_feature_2 = F.interpolate(seg_feature, size=(input_h//4, input_w//4))
                seg_feature_3 = F.interpolate(seg_feature, size=(input_h//8, input_w//8))
            else:
                #print(f"low_level.shape={seg_feature['low_level'].shape}")
                #print(f"low_level2.shape={seg_feature['low_level2'].shape}")
                #print(f"low_level3.shape={seg_feature['low_level3'].shape}")
                seg_feature_1 = F.interpolate(self.seg_conv1(seg_feature['low_level']), size=(input_h//2, input_w//2))
                seg_feature_2 = F.interpolate(self.seg_conv2(seg_feature['low_level2']), size=(input_h//4, input_w//4))
                seg_feature_3 = F.interpolate(self.seg_conv3(seg_feature['low_level3']), size=(input_h//8, input_w//8))

        x1_in = x
        x1 = self.inc(x1_in)

        x2_in = x1.clone()
        x2 = self.down1(x2_in)
        if self.is_input_encoder:
            x3_in = torch.cat((x2, seg_feature_1), dim=1)
        else:
            x3_in = x2

        x3 = self.down2(x3_in)
        if self.is_input_encoder:
            x4_in = torch.cat((x3, seg_feature_2), dim=1)
        else:
            x4_in = x3

        x4 = self.down3(x4_in)
        if self.is_input_encoder:
            x5_in = torch.cat((x4, seg_feature_3), dim=1)
        else:
            x5_in = x4
        
        x5_ = self.down4(x5_in)
        x5 = self.bottleneck(x5_)

        
        if self.verbose: print(f"inc   : {x1.shape}")
        if self.verbose: print(f"down1 : {x2.shape}")
        if self.verbose: print(f"down2 : {x3.shape}")
        if self.verbose: print(f"down3 : {x4.shape}")
        if self.verbose: print(f"down4 : {x5_.shape}")
        if self.verbose: print(f"bottleneck : {x5.shape}")

        # upconv
        out= self.up1(x5, x4)
        if self.verbose: print(f"up1   : {out.shape}")
        out= self.up2(out, x3)
        if self.verbose: print(f"up2   : {out.shape}")
        out= self.up3(out, x2)
        if self.verbose: print(f"up3   : {out.shape, x1.shape}")
        out= self.up4(out, x1)
        if self.verbose: print(f"up4   : {out.shape}")
        #out= self.outc(out)
        #if self.verbose: print(f"outc  : {out.shape}")
        return out



if __name__ == '__main__':
    torch.random.manual_seed(0)
    batch_size = 1
    input_nc = 3
    output_nc = 64
    seg_featire = {'low_level':torch.randn((batch_size, 256, 256, 256)), 'low_level2':torch.randn((batch_size, 512, 128, 128)), 'low_level3':torch.randn((batch_size, 1024, 64, 64))}
    H = 256
    W = 256
    input_size = (H, W)


    input = torch.randn((batch_size, input_nc, H, W))
    #model = UNet_Discriminator3(input_nc=input_nc, output_nc=output_nc, bilinear=True, verbose=True)
    #model = SemanticReasoningBlock2Sobel(input_nc=input_nc, output_nc=output_nc, verbose=True, norm_type='instance')
    model = SemanticReasoningBlock2andSegNet2(input_nc=input_nc, output_nc=output_nc, verbose=True, norm_type='instance', is_input_encoder=True, is_input_decoder=False)
    #model = CAB(input_nc=64, output_nc=64)
    out = model(input, seg_featire, input_size)
    print(len(out))
