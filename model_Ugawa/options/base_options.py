import argparse
from email.policy import default
import os
from util import util
import torch
import models


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', default="../datasets/kaist-cvpr15/copy/train_day/AB/", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--dataroot_A', default="../datasets/kaist-cvpr15/copy/train_day/trainA")
        parser.add_argument('--dataroot_B', default="../datasets/kaist-cvpr15/copy/train_day/trainB")
        parser.add_argument('--seg_dir', default="../datasets/FLIR_aligned_TICCGAN/JPEGImages/train/B/day_segmentation/resnet101/segmentation")
        parser.add_argument('--seg_class_type', type=str, default='cityscapes', choices=['cityscapes', 'voc'], help='type of segmentation class')
        parser.add_argument('--use_seg_model', action='store_true', help='if True, use segmentation model')
        parser.add_argument('--input_type_for_seg_model', type=str, default='real_A', choices=['real_A', 'real_B'], help='type of input for segmentation model')
        parser.add_argument('--seg_ckpt_path', default=None, help='path to segmentation model')
        parser.add_argument('--use_segmap', action='store_true', help='if True, use segmentation map')
        parser.add_argument('--use_segmap_prob', type=float, default=1, help='probability of using segmap as an input')
        parser.add_argument('--seg_mask_size', type=int, default=16, help='size of segmentation mask')
        parser.add_argument('--seg_mask_ratio', type=float, default=0, help='ratio of masked patches of segmentation mask')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize_W', type=int, default=256, help='scale images to this size')
        parser.add_argument('--loadSize_H', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize_W', type=int, default=256, help='then crop to this size')
        parser.add_argument('--fineSize_H', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--which_model_netD_inst', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--which_model_netG', type=str, default='gll', help='selects model to use for netG')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. cycle_gan, pix2pix, test, pix2pix2way')
        parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--use_GAN', default=1, help='1 is use gan')
        parser.add_argument('--transform', type=str, default='centercrop', choices=['centercrop', 'randomcrop'])
        parser.add_argument('--t_prob', type=float, default=1, help='probability of applying transforms on images')
        parser.add_argument('--l1', type=str, default='l1', choices=['l1', 'charbonnier', 'PatchNCE'], help='type of l1 loss')
        parser.add_argument('--w_l1', default=1, type=float, help='weight of the l1 loss')
        parser.add_argument('--w_gan', default=0.03, type=float, help='weight of the gan loss')
        parser.add_argument('--w_vgg', default=1, type=float, help='weight of the vgg loss')
        parser.add_argument('--w_tv', default=1, type=float, help='weight of the tv loss')
        parser.add_argument('--w_edge', default=0, type=float, help='weight of the edge loss')
        parser.add_argument('--w_reg', default=0, type=float, help='weight of the regularization loss')
        parser.add_argument('--w_gan_inst', default=0, type=float, help='weight of the gan_inst loss')
        parser.add_argument('--w_vgg_inst', default=0, type=float, help='weight of the vgg_inst loss')
        parser.add_argument('--w_l1_inst', default=0, type=float, help='weight of the l1_inst loss')
        parser.add_argument('--w_con', default=0, type=float, help='weight of the D_dec_con loss')
        parser.add_argument('--w_sal', default=0, type=float, help='weight of saliency loss')
        parser.add_argument('--w_cd', default=0, type=float, help='weight of color diversity loss')
        parser.add_argument('--w_ssim', default=0, type=float, help='weight of ssim  loss')
        parser.add_argument('--w_seg', default=0, type=float, help='weight of segmentation loss')
        parser.add_argument('--w_gan_seg', default=0, type=float, help='weight of adversarial loss for segmentation masked image')
        parser.add_argument('--start_D_dec_epoch', type=int, default=1, help='when do you start back propagating from D_dec loss')
        parser.add_argument('--no_D_enc', action='store_true', help='if True, ignore output from D_enc')
        parser.add_argument('--rotation', action='store_true', help='if True, input will be rotate')

        parser.add_argument('--gan_type', type=str, default='SGAN', choices=['SGAN', 'LSGAN', 'RSGAN', 'WGANGP'], help='type of GAN loss')
        parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update (WGANGP)')
        parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty (WGANGP)')
        parser.add_argument('--colorspace', type=str, default='RGB', choices=['RGB', 'Lab', 'YCbCr', 'HSV'], help='color space of input and groundtruth images')
        parser.add_argument('--l1_channel', type=str, default='all', choices=['all', 'luminance', 'chrominance'], help='select channel for l1 loss')
        parser.add_argument('--l1_bgOnly', action='store_true', help='if true, calculate l1 loss only for background')
        parser.add_argument('--contrast_factor', default=1, type=float, help='factor for contrast enhancement, 1 is for original images, 0 is for completely grayed images')
        parser.add_argument('--sharp_input', action='store_true', help='if true, laplacian filter will be applied to the input.')
        parser.add_argument('--use_condition', default=1, help='1 means add condition in discriminator')
        parser.add_argument('--stage', default='full', choices=['global', 'instance', 'full'], help='stage of training')
        parser.add_argument('--bbox_threshold', default=512, type=int, help='The factor which divide b-box threshold size. If you set this to 8, b-boxes which are bigger than (img_size / 8) are obtained.')
        parser.add_argument('--D_label', default='normal', choices=['noisy', 'soft', 'normal', 'noisy_soft'], help='option for D label and D_inst label')
        parser.add_argument('--D_inst_label', default='normal', choices=['noisy', 'soft', 'normal', 'noisy_soft'], help='option for D label and D_inst label')
        parser.add_argument('--mirror', action='store_true', help='if true, input image will be randomly flipped.')
        parser.add_argument('--need_bbox', action='store_true', help='if true, train network only when bbox is obtained.')
        parser.add_argument('--A_invert', action='store_true', help='if True, brightness of input images will be inverted.')
        parser.add_argument('--A_invert_prob', default=1, type=float, help='probability of A_invert')
        parser.add_argument('--out_gray', action='store_true', help='if True, outputs will be grayscale images.')
        parser.add_argument('--clahe', action='store_true', help='if True, apply clahe to input.')
        parser.add_argument('--normalize_sal', action='store_true', help='if True, normalize saliency map.')
        parser.add_argument('--sal_mask', action='store_true', help='if True, apply bbox mask to predicted saliency map.')
        parser.add_argument('--input_sod', action='store_true', help='if True, input sod map to the Generator.')
        parser.add_argument('--sal_map', type=str, default='None', choices=['saliency', 'sod', 'None'], help='if True, use saliency map to enhance color inside bboxes.')
        parser.add_argument('--sal_img_path', default='/home/usrs/ugawa/lab/work/TICCGAN/sal_backbone_def/res_imagenet.pth',type=str, help='weight path for saliency backbone.')
        parser.add_argument('--sal_pla_path', default='/home/usrs/ugawa/lab/work/TICCGAN/sal_backbone_def/res_places.pth',type=str, help='weight path for saliency backbone.')
        parser.add_argument('--sal_dec_path', default='/home/usrs/ugawa/lab/work/TICCGAN/sal_backbone_def/res_decoder.pth',type=str, help='weight path for saliency backbone.')
        parser.add_argument('--sal_num_feat', default=5,type=int, help='the number of features collected from each model')

        parser.add_argument('--seg_model_name', default='deeplabv3plus_resnet101', type=str, help='name of segmentation model')
        parser.add_argument('--seg_to_D', action='store_true', help='if True, use segmentation map as input of D.')
        parser.add_argument('--seg_to_D_inst', action='store_true', help='if True, use segmentation map as input of D.')
        parser.add_argument('--seg_to_FAB', action='store_true', help='if True, use segmentation map as input of FAB.')
        parser.add_argument('--seg_num_classes', default=20, type=int, help='the number of classes of segmentation map')
        parser.add_argument('--seg_filter', action='store_true', help='If true, apply gaussian filter to segmentation image.')
        parser.add_argument('--seg_filter_radius', default=2, type=int, help='size of gaussian filter.')


        self.initialized = True
        return parser

    def gather_options(self):

        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)            
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # POSSIBLE FEATURE:
        # modify dataset-related parser options

        self.parser = parser
        
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        if opt.isTrain:
            file_name = os.path.join(expr_dir, f'train_opt_{opt.epoch_count}.txt')
        else:
            file_name = os.path.join(expr_dir, f'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
