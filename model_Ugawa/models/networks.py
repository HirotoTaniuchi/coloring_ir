import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.models as models
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
from .aafstnet import *
from .aafstnet_old import AAFSTNet2old
from .thermal_feanet import FEANet, FEANet2, FEAModule
from .fea_unet import FEAUNet
from .BigGAN import Unet_Discriminator, Unet_Discriminator2
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], colorspace='RGB', seg_num_classes=20):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'gll':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm_layer=norm_layer)
    elif which_model_netG == 'cascaded':
        netG = cascaded(input_nc, output_nc, ngf)
    elif which_model_netG == 'fea_net':
        netG = FEANet2(input_nc, output_nc, verbose=False, resnet_type='resnet152')
    elif which_model_netG == 'AAFSTNet':
        netG = AAFSTNet(input_nc, output_nc, verbose=False, norm_type='instance')
    elif which_model_netG == 'SRB':
        netG = SemanticReasoningBlock(input_nc, output_nc, verbose=False, srb_only=True, norm_type='instance')
    elif which_model_netG == 'AAFSTNet2':
        netG = AAFSTNet2(input_nc, output_nc, verbose=False, norm_type='instance', canny=False)
    elif which_model_netG == 'AAFSTNet2noTTB':
        netG = AAFSTNet2(input_nc, output_nc, verbose=False, norm_type='instance', TTB=False)
    elif which_model_netG == 'AAFSTNet2noRCAB':
        netG = AAFSTNet2(input_nc, output_nc, verbose=False, norm_type='instance', rcab_on=False)
    elif which_model_netG == 'AAFSTNet2old':
        netG = AAFSTNet2old(input_nc, output_nc, verbose=False, norm_type='instance')
    elif which_model_netG == 'AAFSTNet2Canny':
        netG = AAFSTNet2(input_nc, output_nc, verbose=False, canny=False, norm_type='instance')
    elif which_model_netG == 'AAFSTNet2Sobel':
        netG = AAFSTNet2Sobel(input_nc, output_nc, verbose=False, sobel=True, norm_type='instance', colorspace=colorspace)
    elif which_model_netG =='siggraph':
        netG = SIGGRAPHGenerator(input_nc, output_nc, norm_layer=norm_layer, use_tanh=True, classification=False)
    elif which_model_netG =='instance':
        netG = InstanceGenerator(input_nc, output_nc, norm_layer=norm_layer, use_tanh=True, classification=False)
    elif which_model_netG == 'fusion':
        netG = FusionGenerator(input_nc, output_nc, norm_layer=norm_layer, use_tanh=True, classification=False)
    elif which_model_netG == 'FusionAttentionBlock':
        netG = FusionAttentionBlock(input_nc, output_nc, rcab_on=True)
    elif which_model_netG == 'AAFSTNet2Seg':
        netG = AAFSTNet2Seg(input_nc, output_nc, verbose=False, norm_type='instance', colorspace=colorspace)
    elif which_model_netG == 'AAFSTNet2Seg2':
        netG = AAFSTNet2Seg2(input_nc, output_nc, verbose=False, norm_type='instance', colorspace=colorspace)
    elif which_model_netG == 'AAFSTNet2andSegNet':
        netG = AAFSTNet2andSegNet(input_nc, output_nc, verbose=False, norm_type='instance', colorspace=colorspace, is_input_encoder=True, is_input_decoder=False)
    elif which_model_netG == 'AAFSTNet2andSegNet2':
        netG = AAFSTNet2andSegNet2(input_nc, output_nc, verbose=False, norm_type='instance', colorspace=colorspace, is_input_encoder=True, is_input_decoder=False, seg_num_classes=seg_num_classes)
    elif which_model_netG == 'AAFSTNet2andSegNet_NoSeg':
        netG = AAFSTNet2andSegNet(input_nc, output_nc, verbose=False, norm_type='instance', colorspace=colorspace, is_input_encoder=False, is_input_decoder=False)
    elif which_model_netG == 'AAFSTNet2andSegNet2_NoSeg':
        netG = AAFSTNet2andSegNet2(input_nc, output_nc, verbose=False, norm_type='instance', colorspace=colorspace, is_input_encoder=False, is_input_decoder=False, seg_num_classes=0)
    elif which_model_netG == 'AAFSTNet2andSegNet2_RGB':
        netG = AAFSTNet2andSegNet2(input_nc, output_nc, verbose=False, norm_type='instance', colorspace=colorspace, is_input_encoder=True, is_input_decoder=False, seg_num_classes=seg_num_classes, use_rgb_map=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
            n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'basic_DCR':
        netD = NLayerDiscriminator_DCR(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'multi':
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D=3, 
                                        getIntermFeat=False)
    elif which_model_netD == 'unet':
        netD = Unet_Discriminator(unconditional=True)
        #netD = UNet_Discriminator3(input_nc=input_nc, output_nc=1, bilinear=True, verbose=False)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                which_model_netD)
    return init_net(netD, init_type, gpu_ids)

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

# RGANの損失の計算
def relativistic_gan_loss(pred1, pred2):
    sigmoid = torch.sigmoid(pred1 - pred2)
    log = torch.log(sigmoid + 1e-8)
    mean = -torch.mean(log)
    return mean

def relativistic_gan_loss_multi(real, fake):
    loss_ = 0
    for i in range(len(real)):
        for j in range(len(real[i])):
            loss_ += -torch.mean(torch.log(torch.sigmoid(real[i][j] - fake[i][j])))
    if len(real) <= 0:
        loss = 0
    else:
        loss = loss_ / len(real)
    return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
        """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

        Arguments:
            netD (network)              -- discriminator network
            real_data (tensor array)    -- real images
            fake_data (tensor array)    -- generated images from the generator
            device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
            type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
            constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
            lambda_gp (float)           -- weight for this loss

        Returns the gradient penalty loss
        """
        if lambda_gp > 0.0:
            if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
                interpolatesv = real_data
            elif type == 'fake':
                interpolatesv = fake_data
            elif type == 'mixed':
                alpha = torch.rand(real_data.shape[0], 1, device=device)
                alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
                interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
            else:
                raise NotImplementedError('{} not implemented'.format(type))
            interpolatesv.requires_grad_(True)
            disc_interpolates = netD(interpolatesv)
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)
            gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
            gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
            return gradient_penalty, gradients
        else:
            return 0.0, None

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_type='SGAN', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if gan_type=='LSGAN':
            print("lsgan")
            self.loss = nn.MSELoss()
        elif gan_type=='SGAN':
            print("SGAN")
            self.loss = nn.BCELoss()
        elif gan_type=='RSGAN':
            print("RSGAN")
            self.loss = relativistic_gan_loss
        elif gan_type=='WGANGP':
            print("WGANGP")
            self.loss = None
        self.gan_type = gan_type

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def rsgan(self, pred_1, pred_2):
        return self.loss(pred_1, pred_2)

    def __call__(self, input, target_is_real):
        if self.gan_type in ['LSGAN', 'SGAN']:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = self.loss(input, target_tensor)
        elif self.gan_type == 'WGANGP':
            if target_is_real:
                loss = -input.mean()
            else:
                loss = input.mean()
        return loss
    
class GANLossDec(nn.Module):
    def __init__(self, gan_type='SGAN', target_real_label=1.0, target_fake_label=0.0):
        super(GANLossDec, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if gan_type=='LSGAN':
            print("lsgan")
            self.loss = nn.MSELoss(reduction='mean')
        elif gan_type=='SGAN':
            print("SGAN")
            self.loss = nn.BCELoss(reduction='mean')
        elif gan_type=='RSGAN':
            print("RSGAN")
            self.loss = relativistic_gan_loss

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def rsgan(self, pred_1, pred_2):
        return self.loss(pred_1, pred_2)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    
class GANLoss_multi(nn.Module):
    def __init__(self, gan_type='SGAN', target_real_label=1.0, target_fake_label=0.0,
                tensor=torch.cuda.FloatTensor):
        super(GANLoss_multi, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if gan_type=='LSGAN':
            print("lsgan")
            self.loss = nn.MSELoss()
        elif gan_type=='SGAN':
            print("SGAN")
            self.loss = nn.BCELoss()
        elif gan_type=='RSGAN':
            print("RSGAN")
            self.loss = relativistic_gan_loss_multi

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def rsgan(self, pred_1, pred_2):
        return self.loss(pred_1, pred_2)

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

# code borrowed from https://github.com/styler00dollar/pytorch-loss-functions
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                    stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
# gloal local generator
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock_gll(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                                norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock_gll(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             

class ResnetBlock_gll(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock_gll, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim),
                        activation]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Define a resnet block
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


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2,
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


# DCR module
class DCR(nn.Module):
    def __init__(self, input_nc, inner_nc, output_nc):
        super(DCR, self).__init__()
        conv1 = nn.Conv2d(in_channels=input_nc, out_channels=inner_nc, kernel_size=1, stride=1, padding=0)
        l_relu = nn.LeakyReLU()
        conv2 = nn.Conv2d(in_channels=inner_nc, out_channels=output_nc, kernel_size=1, stride=1, padding=0)
        self.model = nn.Sequential(*[conv1, l_relu, conv2])

    def forward(self, x):
        out = self.model(x)
        return out
        

class NLayerDiscriminator_DCR(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator_DCR, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.d_0 = nn.Sequential(*sequence)
        print(f"ndf * nf_mult={ndf * nf_mult}")
        self.dcr = DCR(input_nc=ndf * nf_mult, inner_nc=512, output_nc=ndf * nf_mult)
        
        d_1 = []
        d_1 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            d_1 += [nn.Sigmoid()]
        self.d_1 = nn.Sequential(*d_1)

    def forward(self, input):
        out = self.d_1(self.d_0(input))
        return out
    
    def dcr_loss(self, input):
        verbose = False
        N, C, H, W = input.shape
        if verbose: print(f"N,C,H,W={N},{C},{H},{W}")
        H_1_start = np.random.randint(0, H//2)
        W_1_start = np.random.randint(0, W//2)
        H_1_end = H_1_start + H//2
        W_1_end = W_1_start + W//2

        H_2_start = np.random.randint(H_1_start, H//2)
        W_2_start = np.random.randint(W_1_start, W//2)
        H_2_end = H_2_start + H//2
        W_2_end = W_2_start + W//2
        
        mask = torch.zeros_like(input=input)
        mask[..., H_2_start:H_1_end, W_2_start:W_1_end] = 1

        view_1 = input[:, :, H_1_start:H_1_end, W_1_start:W_1_end].clone()
        view_2 = input[:, :, H_2_start:H_2_end, W_2_start:W_2_end].clone()

        r1 = self.dcr(self.d_0(view_1))
        z2 = self.d_0(view_2)
        z1 = self.d_0(view_1)
        r2 = self.dcr(self.d_0(view_2))
        
        _, _, H_r1, W_r1 = r1.shape
        k_width = H_r1 // 2
        loss_1 = 0
        loss_2 = 0
        if verbose:
            print(f"r1.shape = {r1.shape}")
            print(f'H={H}, W={W}')

        """for i in range(H_2_start, H_1_end):
            for j in range(W_2_start, W_1_end):
                loss_1 += -torch.cosine_similarity(r1[:, :, i, j], z2[:, :, max(i-k_width,H_2_start):min(i+k_width,H_1_end), max(j-k_width,W_2_start):min(j+k_width, W_1_end)].detach())
                loss_2 += -torch.cosine_similarity(r2[:, :, i, j], z1[:, :, max(i-k_width,H_2_start):min(i+k_width,H_1_end), max(j-k_width,W_2_start):min(j+k_width, W_1_end)].detach())"""

        criterion = nn.CosineSimilarity()
        loss_1 = -criterion(r1, z2).mean()
        loss_2 = -criterion(r2, z1).mean()

        loss = loss_1 / 2 + loss_2 / 2
        return loss

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminator_multi(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator_multi, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator_multi(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


##cascaded network
class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-12, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

class cascaded(nn.Module):
    
    def __init__(self, input_nc, output_nc, ngf):
        super(cascaded, self).__init__()
        
        #Layer1 4*4---8*8 
        self.conv1=nn.Conv2d(input_nc, ngf*16, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay1=LayerNorm(ngf*16, eps=1e-12, affine=True)      
        self.relu1=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv11=nn.Conv2d(ngf*16, ngf*16, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay11=LayerNorm(ngf*16, eps=1e-12, affine=True)        
        self.relu11=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        #Layer2 8*8---16*16        
        self.conv2=nn.Conv2d(ngf*16+input_nc, ngf*16, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay2=LayerNorm(ngf*16, eps=1e-12, affine=True)
        self.relu2=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv22=nn.Conv2d(ngf*16, ngf*16, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay22=LayerNorm(ngf*16, eps=1e-12, affine=True)
        self.relu22=nn.LeakyReLU(negative_slope=0.2,inplace=True)        
        
        #layer3 16*16---32*32        
        self.conv3=nn.Conv2d(ngf*16+input_nc, ngf*8, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay3=LayerNorm(ngf*8, eps=1e-12, affine=True)
        self.relu3=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv33=nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay33=LayerNorm(ngf*8, eps=1e-12, affine=True)
        self.relu33=nn.LeakyReLU(negative_slope=0.2,inplace=True)

        #layer4 32*32---64*64               
        self.conv4=nn.Conv2d(ngf*8+input_nc, ngf*4, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay4=LayerNorm(ngf*4, eps=1e-12, affine=True)
        self.relu4=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv44=nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay44=LayerNorm(ngf*4, eps=1e-12, affine=True)
        self.relu44=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        #layer5 64*64---128*128 
        
        self.conv5=nn.Conv2d(ngf*4+input_nc, ngf*2, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay5=LayerNorm(ngf*2, eps=1e-12, affine=True)
        self.relu5=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv55=nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay55=LayerNorm(ngf*2, eps=1e-12, affine=True)
        self.relu55=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        #layer6 128*128---256*256       
        self.conv6=nn.Conv2d(ngf*2+input_nc, ngf, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay6=LayerNorm(ngf, eps=1e-12, affine=True)
        self.relu6=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv66=nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1,bias=True)
        self.lay66=LayerNorm(ngf, eps=1e-12, affine=True)
        self.relu66=nn.LeakyReLU(negative_slope=0.2,inplace=True)

        #Layer7 256*256
        self.conv7=nn.Conv2d(ngf+input_nc, output_nc, kernel_size=3, stride=1, padding=1,bias=True)

        #Layer_downsample
        self.downsample = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)

    def forward(self, input):
        input_128 = self.downsample(input)
        input_64 = self.downsample(input_128)
        input_32 = self.downsample(input_64)
        input_16 = self.downsample(input_32)
        input_8 = self.downsample(input_16)
        input_4 = self.downsample(input_8)

        #Layer1 4*4---8*8
        out1= self.conv1(input_4)
        L1=self.lay1(out1)
        out2= self.relu1(L1)
        
        out11= self.conv11(out2)
        L11=self.lay11(out11)
        out22= self.relu11(L11)

        m = nn.Upsample(size=(input_4.size(3)*2,input_4.size(3)*2), mode='bilinear')        

        img1 = torch.cat((m(out22), input_8),1)        

        #Layer2 8*8---16*16        
        out3= self.conv2(img1)
        L2=self.lay2(out3)
        out4= self.relu2(L2)
        
        out33= self.conv22(out4)
        L22=self.lay22(out33)
        out44= self.relu22(L22)
        
        m = nn.Upsample(size=(input_8.size(3)*2,input_8.size(3)*2), mode='bilinear')
        
        img2 = torch.cat((m(out44), input_16),1)
        
        #Layer3 16*16---32*32
        out5= self.conv3(img2)
        L3=self.lay3(out5)
        out6= self.relu3(L3)
        
        out55= self.conv33(out6)
        L33=self.lay33(out55)
        out66= self.relu33(L33)
        
        m = nn.Upsample(size=(input_16.size(3)*2,input_16.size(3)*2),mode='bilinear')
        
        img3 = torch.cat((m(out66), input_32),1)
        
        #Layer4 32*32---64*64
        out7= self.conv4(img3)
        L4=self.lay4(out7)
        out8= self.relu4(L4)
        
        out77= self.conv44(out8)
        L44=self.lay44(out77)
        out88= self.relu44(L44)

        m = nn.Upsample(size=(input_32.size(3)*2,input_32.size(3)*2),mode='bilinear')
        
        img4 = torch.cat((m(out88), input_64),1)        
        
        #Layer5 64*64---128*128
        out9= self.conv5(img4)
        L5=self.lay5(out9)
        out10= self.relu5(L5)
        
        out99= self.conv55(out10)
        L55=self.lay55(out99)
        out110= self.relu55(L55)
        
        m = nn.Upsample(size=(input_64.size(3)*2,input_64.size(3)*2),mode='bilinear')
        
        img5 = torch.cat((m(out110), input_128),1)
        
        #Layer6 128*128---256*256       
        out11= self.conv6(img5)
        L6=self.lay6(out11)
        out12= self.relu6(L6)
        
        out111= self.conv66(out12)
        L66=self.lay66(out111)
        out112= self.relu66(L66)
        
        m = nn.Upsample(size=(input_128.size(3)*2, input_128.size(3)*2), mode='bilinear')
        
        img6 = torch.cat((m(out112), input),1)       
        
        #Layer7 256*256 
        out13 = self.conv7(img6)

        return out13


from torch.autograd import Variable
from torchvision.transforms import Grayscale
import torch.nn.functional as F

class ImageSharpening(nn.Module):
    def __init__(self, input_nc=3):
        super(ImageSharpening, self).__init__()
        self.input_nc = input_nc
        self.grayscale = Grayscale(num_output_channels=1)
        # ラプラシアンフィルタ
        weight = [1., 1., 1.,
                  1., -8.,  1.,
                  1., 1., 1.]
        self.filter_weight = torch.cuda.FloatTensor(weight).view(1, 1, 3, 3).repeat(1, input_nc, 1, 1)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, input):
        # apply filter
        padded = self.pad(input)
        output = input - F.conv2d(padded, weight=Variable(self.filter_weight), stride=1)
        return output


class SIGGRAPHGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True, classification=True):
        super(SIGGRAPHGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        model1=[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model1+=[norm_layer(64),]
        model1+=[nn.ReLU(True),]
        # model1+=[nn.ReflectionPad2d(1),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
        # add a subsampling operation

        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model2+=[norm_layer(128),]
        model2+=[nn.ReLU(True),]
        # model2+=[nn.ReflectionPad2d(1),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        # Conv7
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        # model47+=[norm_layer(256),]
        model8=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model8+=[norm_layer(256),]
        model8+=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above        

        # model9=[norm_layer(128),]
        model9=[nn.ReLU(True),]
        # model9+=[nn.ReflectionPad2d(1),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above

        # model10=[norm_layer(128),]
        model10=[nn.ReLU(True),]
        # model10+=[nn.ReflectionPad2d(1),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias),]
        model10+=[nn.LeakyReLU(negative_slope=.2),]

        # classification output
        model_class=[nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]

        # regression output
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]
        if(use_tanh):
            model_out+=[nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'),])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])

    def forward(self, input_A, input_B, mask_B):
        conv1_2 = self.model1(torch.cat((input_A,input_B,mask_B),dim=1))
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        if(self.classification):
            out_class = self.model_class(conv8_3)
            conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach())
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2.detach())
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
        else:
            out_class = self.model_class(conv8_3.detach())

            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)

        return (out_class, out_reg)


class FusionGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True, classification=True):
        super(FusionGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        model1=[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model1+=[norm_layer(64),]
        model1+=[nn.ReLU(True),]
        # model1+=[nn.ReflectionPad2d(1),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
        # add a subsampling operation

        self.weight_layer = WeightGenerator(64)
        
        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model2+=[norm_layer(128),]
        model2+=[nn.ReLU(True),]
        # model2+=[nn.ReflectionPad2d(1),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        self.weight_layer2 = WeightGenerator(128)

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        self.weight_layer3 = WeightGenerator(256)

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        self.weight_layer4 = WeightGenerator(512)

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        self.weight_layer5 = WeightGenerator(512)

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        self.weight_layer6 = WeightGenerator(512)

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        self.weight_layer7 = WeightGenerator(512)

        # Conv7
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        self.weight_layer8_1 = WeightGenerator(256)

        # model47+=[norm_layer(256),]
        model8=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model8+=[norm_layer(256),]
        model8+=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        self.weight_layer8_2 = WeightGenerator(256)

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above        

        self.weight_layer9_1 = WeightGenerator(128)

        # model9=[norm_layer(128),]
        model9=[nn.ReLU(True),]
        # model9+=[nn.ReflectionPad2d(1),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        self.weight_layer9_2 = WeightGenerator(128)

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above

        self.weight_layer10_1 = WeightGenerator(128)

        # model10=[norm_layer(128),]
        model10=[nn.ReLU(True),]
        # model10+=[nn.ReflectionPad2d(1),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias),]
        model10+=[nn.LeakyReLU(negative_slope=.2),]

        self.weight_layer10_2 = WeightGenerator(128)

        # classification output
        model_class=[nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]

        # regression output
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]
        if(use_tanh):
            model_out+=[nn.Tanh()]

        self.weight_layerout = WeightGenerator(2)

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'),])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])

    def forward(self, input_A, input_B, mask_B, instance_feature, box_info_list):
        conv1_2 = self.model1(torch.cat((input_A,input_B,mask_B),dim=1))
        conv1_2 = self.weight_layer(instance_feature['conv1_2'], conv1_2, box_info_list[0])

        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv2_2 = self.weight_layer2(instance_feature['conv2_2'], conv2_2, box_info_list[1])

        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv3_3 = self.weight_layer3(instance_feature['conv3_3'], conv3_3, box_info_list[2])

        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv4_3 = self.weight_layer4(instance_feature['conv4_3'], conv4_3, box_info_list[3])

        conv5_3 = self.model5(conv4_3)
        conv5_3 = self.weight_layer5(instance_feature['conv5_3'], conv5_3, box_info_list[3])

        conv6_3 = self.model6(conv5_3)
        conv6_3 = self.weight_layer6(instance_feature['conv6_3'], conv6_3, box_info_list[3])

        conv7_3 = self.model7(conv6_3)
        conv7_3 = self.weight_layer7(instance_feature['conv7_3'], conv7_3, box_info_list[3])

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_up = self.weight_layer8_1(instance_feature['conv8_up'], conv8_up, box_info_list[2])

        conv8_3 = self.model8(conv8_up)
        conv8_3 = self.weight_layer8_2(instance_feature['conv8_3'], conv8_3, box_info_list[2])

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_up = self.weight_layer9_1(instance_feature['conv9_up'], conv9_up, box_info_list[1])

        conv9_3 = self.model9(conv9_up)
        conv9_3 = self.weight_layer9_2(instance_feature['conv9_3'], conv9_3, box_info_list[1])

        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_up = self.weight_layer10_1(instance_feature['conv10_up'], conv10_up, box_info_list[0])

        conv10_2 = self.model10(conv10_up)
        conv10_2 = self.weight_layer10_2(instance_feature['conv10_2'], conv10_2, box_info_list[0])
        
        out_reg = self.model_out(conv10_2)
        return out_reg


class WeightGenerator(nn.Module):
    def __init__(self, input_ch, inner_ch=16):
        super(WeightGenerator, self).__init__()
        self.simple_instance_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.simple_bg_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.normalize = nn.Softmax(1)
    
    def resize_and_pad(self, feauture_maps, info_array):
        feauture_maps = torch.nn.functional.interpolate(feauture_maps, size=(info_array[5], info_array[4]), mode='bilinear')
        feauture_maps = torch.nn.functional.pad(feauture_maps, (info_array[0], info_array[1], info_array[2], info_array[3]), "constant", 0)
        return feauture_maps
    
    def forward(self, instance_feature, bg_feature, box_info):
        mask_list = []
        featur_map_list = []
        mask_sum_for_pred = torch.zeros_like(bg_feature)[:1, :1]
        for i in range(instance_feature.shape[0]):
            tmp_crop = torch.unsqueeze(instance_feature[i], 0)
            conv_tmp_crop = self.simple_instance_conv(tmp_crop)
            pred_mask = self.resize_and_pad(conv_tmp_crop, box_info[i])
            
            tmp_crop = self.resize_and_pad(tmp_crop, box_info[i])

            mask = torch.zeros_like(bg_feature)[:1, :1]
            mask[0, 0, box_info[i][2]:box_info[i][2] + box_info[i][5], box_info[i][0]:box_info[i][0] + box_info[i][4]] = 1.0
            device = mask.device
            mask = mask.type(torch.FloatTensor).to(device)

            mask_sum_for_pred = torch.clamp(mask_sum_for_pred + mask, 0.0, 1.0)

            mask_list.append(pred_mask)
            featur_map_list.append(tmp_crop)

        pred_bg_mask = self.simple_bg_conv(bg_feature)
        mask_list.append(pred_bg_mask + (1 - mask_sum_for_pred) * 100000.0)
        mask_list = self.normalize(torch.cat(mask_list, 1))

        mask_list_maskout = mask_list.clone()
        
        instance_mask = torch.clamp(torch.sum(mask_list_maskout[:, :instance_feature.shape[0]], 1, keepdim=True), 0.0, 1.0)

        featur_map_list.append(bg_feature)
        featur_map_list = torch.cat(featur_map_list, 0)
        mask_list_maskout = mask_list_maskout.permute(1, 0, 2, 3).contiguous()
        out = featur_map_list * mask_list_maskout
        out = torch.sum(out, 0, keepdim=True)
        return out # , instance_mask, torch.clamp(mask_list, 0.0, 1.0)


class InstanceGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True, classification=True):
        super(InstanceGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        model1=[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model1+=[norm_layer(64),]
        model1+=[nn.ReLU(True),]
        # model1+=[nn.ReflectionPad2d(1),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
        # add a subsampling operation

        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model2+=[norm_layer(128),]
        model2+=[nn.ReLU(True),]
        # model2+=[nn.ReflectionPad2d(1),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        # Conv7
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        # model47+=[norm_layer(256),]
        model8=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model8+=[norm_layer(256),]
        model8+=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above        

        # model9=[norm_layer(128),]
        model9=[nn.ReLU(True),]
        # model9+=[nn.ReflectionPad2d(1),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above

        # model10=[norm_layer(128),]
        model10=[nn.ReLU(True),]
        # model10+=[nn.ReflectionPad2d(1),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias),]
        model10+=[nn.LeakyReLU(negative_slope=.2),]

        # classification output
        model_class=[nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]

        # regression output
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]
        if(use_tanh):
            model_out+=[nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'),])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])

    def forward(self, input_A, input_B, mask_B):
        conv1_2 = self.model1(torch.cat((input_A,input_B,mask_B),dim=1))
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        if(self.classification):
            out_class = self.model_class(conv8_3)
            conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach())
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2.detach())
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
        else:
            out_class = self.model_class(conv8_3.detach())

            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)

        feature_map = {}
        feature_map['conv1_2'] = conv1_2
        feature_map['conv2_2'] = conv2_2
        feature_map['conv3_3'] = conv3_3
        feature_map['conv4_3'] = conv4_3
        feature_map['conv5_3'] = conv5_3
        feature_map['conv6_3'] = conv6_3
        feature_map['conv7_3'] = conv7_3
        feature_map['conv8_up'] = conv8_up
        feature_map['conv8_3'] = conv8_3
        feature_map['conv9_up'] = conv9_up
        feature_map['conv9_3'] = conv9_3
        feature_map['conv10_up'] = conv10_up
        feature_map['conv10_2'] = conv10_2
        feature_map['out_reg'] = out_reg
        return (out_reg, feature_map)


if __name__ == '__main__':
    torch.manual_seed(0)
    batch_size = 4
    H = 480
    W = 640
    input_nc = 3
    output_nc = 3

    input_2 = torch.randn((batch_size, input_nc, H, W))
    print(input_2[0][0])
    model = FEANet2(input_nc=input_nc, output_nc=3, verbose=True, resnet_type='resnet152')
    model = NLayerDiscriminator(input_nc, 64, n_layers=3, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True), use_sigmoid=True)
    #model = GlobalGenerator(input_nc, output_nc)
    pred = model(input_2)
    #model_2 = FEANet2(resnet_type='resnet34')
    #pred = model_2(input_2)
    print(f"\npred_2 : {pred[0][0]}")
