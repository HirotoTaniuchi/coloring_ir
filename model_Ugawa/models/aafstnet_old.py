import torch
import torch.nn as nn
from torchvision.transforms import Grayscale
import torch.nn.functional as F
from torch.autograd import Variable
from .net_canny import Canny
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

# old


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
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4, stride=2),
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
        model += [nn.ReLU(inplace=True)]

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
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

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
        # Grayscale
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
    def __init__(self, in_channels, out_channels):
        super(FusionAttentionBlock, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)]
        model += [nn.LeakyReLU(0.2)]
        model += [RCAB(input_nc=in_channels, output_nc=in_channels)]
        #model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)]
        #model += [nn.Sigmoid()]
        model += [OutConv(in_channels=in_channels, out_channels=3)]
        self.fab = nn.Sequential(*model)
    
    def forward(self, input):
        out = self.fab(input)
        return out


######################################################################
#                            AAFSTNet2old                               #
######################################################################
class AAFSTNet2old(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=False, canny=False, TTB=True, norm_type='instance'):
        super(AAFSTNet2old, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.canny = canny
        self.verbose = verbose
        self.TTB = TTB
        self.srb = SemanticReasoningBlock2(input_nc=input_nc, output_nc=64, verbose=verbose, norm_type='instance')
        if self.canny:
            self.tex_transfar = Canny(threshold=1, use_cuda=True)
        else:
            self.tex_transfar = TextureTransferBlock(input_nc=input_nc)
        self.tex_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        if self.TTB:
            self.fab = FusionAttentionBlock(in_channels=128, out_channels=3)
        else:
            self.fab = FusionAttentionBlock(in_channels=64, out_channels=3)

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
        self.down1 = Down2(in_channels=64, out_channels=128, norm_type=norm_type)
        self.down2 = Down2(in_channels=128, out_channels=256, norm_type=norm_type)
        self.down3 = Down2(in_channels=256, out_channels=512, norm_type=norm_type)
        factor = 2 if bilinear else 1
        self.down4 = Down2(in_channels=512, out_channels=1024 // factor)
        self.up1 = Up2(in_channels=1024, out_channels=512 // factor, bilinear=bilinear)
        self.up2 = Up2(in_channels=512, out_channels=256 // factor, bilinear=bilinear, norm_type=norm_type)
        self.up3 = Up2(in_channels=256, out_channels=128 // factor, bilinear=bilinear, norm_type=norm_type)
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
        x5 = self.down4(x4.clone())
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
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down2(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        super(Down2, self).__init__()
        norm_layer = set_normalize(norm_type=norm_type)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4, stride=2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type),
            RCAB(input_nc=out_channels, output_nc=out_channels),
            norm_layer(out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up2(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type='instance'):
        super(Up2, self).__init__()
        norm_layer = set_normalize(norm_type=norm_type)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, \
                                    mid_channels=in_channels // 2, norm_type=norm_type)
            self.rcab = nn.Sequential(RCAB(input_nc=out_channels, output_nc=out_channels),
                                      norm_layer(out_channels))
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            self.rcab = nn.Sequential(RCAB(input_nc=out_channels, output_nc=out_channels),
                                      norm_layer(out_channels))
            


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
        out = self.conv(x)
        out = self.rcab(out) 
        return out


if __name__ == '__main__':
    torch.random.manual_seed(0)
    batch_size = 1
    input_nc = 3
    output_nc = 64
    H = 256
    W = 256


    input = torch.randn((batch_size, input_nc, H, W))
    #model = UNet(input_nc=input_nc, output_nc=output_nc, bilinear=True, verbose=True)
    #model = AAFSTNet2old(input_nc=input_nc, output_nc=output_nc, verbose=True, norm_type='instance')
    model = Canny(threshold=2, use_cuda=True)
    fake = model(input)
    img = to_pil_image(fake[0])
    img.save('edge.jpg')
    print(f"{fake.shape}")