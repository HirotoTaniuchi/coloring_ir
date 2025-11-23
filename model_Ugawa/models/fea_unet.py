import torch
import torch.nn as nn
import torch.nn.functional as F
from .thermal_feanet import FEAModule
from torchvision.transforms import Grayscale
import torch.nn.functional as F
from torch.autograd import Variable

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

# https://github.com/milesial/Pytorch-UNet
class FEAUNet(nn.Module):
    def __init__(self, input_nc, output_nc, bilinear=True, verbose=False, feam_on=True, norm_type='instance'):
        super(FEAUNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.verbose = verbose
        self.feam_on = feam_on

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
        if feam_on:
            self.feam = FEAModule()

    def forward(self, x):
        # downsampling
            # self.feam_onでFEAModuleを使用
            # self.verbose==Trueで詳細表示
        x1 = self.inc(x)
        if self.feam_on: 
            x1_ = self.feam(x1)
        else:
            x1_ = x1
        x2 = self.down1(x1_)
        if self.feam_on: 
            x2_ = self.feam(x2)
        else:
            x2_ = x2
        x3 = self.down2(x2_)
        if self.feam_on: 
            x3_ = self.feam(x3)
        else:
            x3_ = x3
        x4 = self.down3(x3_)
        if self.feam_on: 
            x4_ = self.feam(x4)
        else:
            x4_ = x4
        x5 = self.down4(x4_)
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
            nn.MaxPool2d(2),
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                  nn.Tanh())

    def forward(self, x):
        return self.conv(x)




if __name__ == '__main__':
    torch.random.manual_seed(0)
    batch_size = 4
    input_nc = 3
    output_nc = 3
    H = 224
    W = 224
    input = torch.randn((batch_size, input_nc, H, W))
    #input = torch.full(size=(batch_size, input_nc, H, W), fill_value=1).float()
    model = FEAUNet(input_nc=input_nc, output_nc=output_nc, verbose=True, bilinear=False, norm_type='instance')
    fake = model(input)
    print(fake)
