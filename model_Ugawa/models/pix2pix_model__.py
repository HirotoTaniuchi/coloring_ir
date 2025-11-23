import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import util.util as myutl
from PIL import Image
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import network_segmentation
from .vgg import Vgg16
from .aafstnet import TextureTransferBlock
from .net_canny import Canny
from models.nce import PatchNCELoss
import torchvision.transforms as TF
from pytorch_msssim import ssim
import random
import argparse
import kornia as K
from BASNet.model.BASNet import BASNet
import torchvision
from collections import namedtuple

############################    CX Loss implimentation   ############################
# original implimentation

# by S-aiueo
#import contextual_loss as cl 
#import contextual_loss.functional as clf
# by z-bingo
#from models.ContextualLossPyTorch.ContextualLoss import Contextual_Loss
#####################################################################################

# sodマップの計算で使用
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn   

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        #parser.set_defaults(dataset_mode='aligned')
        #parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # tensorboardなどに表示するの名前のリスト
        self.loss_names = myutl.make_loss_names_list(opt)

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if ('AAFSTNet2' in opt.which_model_netG) and (opt.phase == 'train') and not(opt.which_model_netG=='AAFSTNet2andSegNet_NoSeg'):
            self.visual_names.append("edge")

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # use_gan
        self.use_gan = opt.use_GAN

        if self.opt.which_model_netD == 'basic_DCR':
            self.loss_names.append('D_DCR')

        # instを使用するか
        self.use_inst = True

        # flip threshold
        self.D_label_flip_threshold = 0.1

        # fullの場合はmodel_names, loss_names, visual_namesにinstance関連の名前を追加する
        if self.isTrain and opt.stage == 'full':
            if self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0:
                self.visual_names.append('real_A_inst')
                self.visual_names.append('real_B_inst')
                self.visual_names.append('fake_B_inst')
                self.model_names.append('D_inst')
                if not(self.opt.gan_type=='RSGAN'):
                    self.loss_names.append('D_real_inst')
                    self.loss_names.append('D_fake_inst')
                    self.loss_names.append('D_fake_inst')
                self.loss_names.append('D_inst')    
                self.loss_names.append('G_GAN_inst')    
            if self.opt.w_l1_inst > 0:
                self.loss_names.append('l1_inst')
            if self.opt.w_vgg_inst > 0:
                self.loss_names.append('vgg_inst')
            # 使ってないので消しても良い
            if self.opt.w_cx_inst > 0:
                self.loss_names.append('cx_inst')

        # 各lossの重みをselfに格納
        self.w_l1 = opt.w_l1
        self.w_vgg = opt.w_vgg
        self.w_tv = opt.w_tv
        self.w_gan = opt.w_gan
        self.w_cx = opt.w_cx
        self.w_edge = opt.w_edge
        self.w_cd = opt.w_cd
        self.w_gan_inst = opt.w_gan_inst
        self.w_vgg_inst = opt.w_vgg_inst
        self.w_l1_inst = opt.w_l1_inst
        self.w_cx_inst = opt.w_cx_inst
        self.w_seg = opt.w_seg
        self.w_gan_seg = opt.w_gan_seg
        self.w_ssim = opt.w_ssim

        self.resize = TF.Resize((self.opt.fineSize_H, self.opt.fineSize_W), interpolation=Image.BICUBIC)
        self.avepool = torch.nn.AvgPool2d(3,2,1)

        # 条件画像を使うかどうか
        self.use_condition = opt.use_condition

        # ネットワークの定義とロード
        if opt.use_seg_model and opt.seg_to_FAB:
            fab_additional_channels = opt.seg_num_classes
        else:
            fab_additional_channels = 0
        self.contrast_factor = opt.contrast_factor
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, colorspace=self.opt.colorspace, seg_num_classes=fab_additional_channels)
        
        # use_seg_modelがTrueならセグメンテーションモデルを読み込む
        if self.opt.use_seg_model and not self.opt.which_model_netG == 'AAFSTNet2andSegNet_NoSeg':
            self.load_segmentation_model()

        # 追加の着色モデルとsaliency mapデコーダ
        if opt.sal_map == 'saliency':
            #self.netG2 = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, 'resnet_6blocks', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
            self.sal_img_model, self.sal_pla_model, self.sal_dec_model = myutl.create_sal_nets(opt)
            self.sal_img_model = self.sal_img_model.to(self.device)
            self.sal_pla_model = self.sal_pla_model.to(self.device)
            self.sal_dec_model = self.sal_dec_model.to(self.device)
            print(self.device)
        elif opt.sal_map == 'sod':
            self.sod_net = BASNet(3,1)
            self.sod_net.load_state_dict(torch.load('/home/usrs/ugawa/lab/work/TICCGAN/BASNet/saved_models/basnet_bsi/basnet.pth'))
            if torch.cuda.is_available():
                self.sod_net.eval().to(self.device).requires_grad_(False)
            print(self.device)


        if self.isTrain:
            # セグメンテーション画像をDiscriminatorに入力する場合
            if self.opt.seg_to_D:   
                seg_nc = self.opt.seg_num_classes
            else:
                seg_nc = 0
            
            if self.opt.seg_to_D_inst:
                seg_nc_inst = self.opt.seg_num_classes
            else:
                seg_nc_inst = 0
            
            # Discriminatorの選択
            if self.opt.gan_type=='SGAN':
                # SGANかRSGANならDiscriminatorの最後にsigmoidを入れる
                use_sigmoid = True 
            else:
                use_sigmoid = False

            # 条件画像を入力する場合
            if self.use_condition == 1: 
                self.netD = networks.define_D(opt.input_nc + opt.output_nc + seg_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                
                # 画像からinstanceを切り出し、そこにペナルティを課すためのDiscriminatorを用意
                if opt.stage == 'full' and (self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0):
                    self.netD_inst = networks.define_D(opt.input_nc + opt.output_nc + seg_nc_inst, opt.ndf,
                                            opt.which_model_netD_inst,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            # 条件画像を入力しない場合
            else: 
                self.netD = networks.define_D(opt.input_nc + seg_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                # 画像からinstanceを切り出し、そこにペナルティを課すためのDiscriminatorを用意
                if opt.stage == 'full' and (self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0):
                    self.netD_inst = networks.define_D(opt.input_nc + seg_nc_inst, opt.ndf,
                                            opt.which_model_netD_inst,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)


        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            if opt.stage == 'full':
                self.fake_AB_pool_inst = ImagePool(opt.pool_size)
            
            # 損失関数の定義
            # adversarial loss
            if opt.which_model_netD == 'multi':
                self.criterionGAN = networks.GANLoss_multi(gan_type=self.opt.gan_type).to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(gan_type=self.opt.gan_type).to(self.device)
                self.criterionGAN_dec = networks.GANLossDec(gan_type=self.opt.gan_type).to(self.device)
            
            # L1 loss
            if self.opt.l1 == 'l1':
                self.criterionL1 = torch.nn.L1Loss()
            elif self.opt.l1 == 'charbonnier':
                self.criterionL1 = networks.CharbonnierLoss()
            self.criterionL2 = torch.nn.MSELoss()

            # vgg loss
            self.vgg = Vgg16().type(torch.cuda.FloatTensor)

            # edge loss
            if opt.which_model_netG == 'AAFSTNet2Canny':
                self.ttb = Canny(threshold=1, use_cuda=True).type(torch.cuda.FloatTensor)
            else:
                self.ttb = K.filters.sobel

            # optimizerの初期化
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.stage == 'full' and (self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0):
                self.optimizer_D = torch.optim.Adam(list(self.netD.parameters())+list(self.netD_inst.parameters()), lr=opt.lr_const*opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_const*opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input, bboxes=0):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if self.opt.use_segmap:
            self.seg = input['seg'].to(self.device)
        # instanceをbboxの大きさに切り出し
        self.bbox_mask = myutl.create_bbox_mask_from_bbox(self.real_A[:, 0:1, ...].shape, bboxes).to(self.device)
        if not (bboxes == 0) and (self.opt.w_gan_inst > 0 or self.opt.w_l1_inst > 0):
            self.real_A_inst, self.real_A_bg = myutl.create_inst_images(self.real_A, bboxes)
            self.real_A_inst = self.real_A_inst.to(self.device)
            self.real_A_bg = self.real_A_bg.to(self.device)
            self.real_B_inst, self.real_B_bg = myutl.create_inst_images(self.real_B, bboxes)
            self.real_B_inst = self.real_B_inst.to(self.device)
            self.real_B_bg = self.real_B_bg.to(self.device)
            self.use_inst = True

            self.nonzero_inst = 1    # instロスの損失はinst画像のピクセル数に依存しない
            #self.nonzero_inst = torch.count_nonzero(self.real_A_inst).item() / (self.opt.loadSize_W * self.opt.loadSize_H)
        # use_seg_modelがTrueかつ、w_gan_segが0より大きいとき
        if self.opt.use_seg_model and self.opt.w_gan_seg > 0:
            self.nonzero_inst = 1
            self.use_inst = True
        else:
            self.use_inst = False
            self.nonzero_inst = 0
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def load_segmentation_model(self):
        # num_classes = self.opt.seg_num_classes
        # output_stride = self.opt.seg_output_stride
        output_stride=16
        ckpt_path = self.opt.seg_ckpt_path
        
        # モデルの読み込み
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_name = self.opt.seg_model_name
        model = network_segmentation.modeling.__dict__[model_name](num_classes=self.opt.seg_num_classes, output_stride=output_stride)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        
        self.seg_model = model
    
    def get_segmentation_feature(self):
        # セグメンテーションモデルのバックボーンから特徴量を取得
        # optに応じてセグメンテーションモデルへの入力を変更
        if self.opt.input_type_for_seg_model == 'real_B':
            seg_input = self.real_B.clone().detach()
        elif self.opt.input_type_for_seg_model == 'real_A':
            seg_input = self.real_A.clone().detach()
        else:
            raise NotImplementedError('optionのinput_type_for_seg_modelが不正です。')
        scale_factor = 2
        seg_input = F.interpolate(seg_input, size=(scale_factor*self.opt.fineSize_H, scale_factor*self.opt.fineSize_W), mode='bilinear', align_corners=True)
        pred_, feature = self.seg_model(seg_input)
        # 勾配を計算しないようにする
        for key in feature.keys():
            feature[key] = feature[key].detach()
        self.seg_feature = feature
        pred_ = F.max_pool2d(pred_, kernel_size=2, stride=2) # (1, 19, 256, 256)
        pred = pred_.detach().argmax(1)
        # one-hotに変換
        self.seg_pred = torch.zeros_like(pred_).scatter_(1, pred.unsqueeze(1), 1.).detach()
        #print(self.seg_pred)

        # RGBラベルに変換
        decoded_pred = self.decode_segmentation_map(pred[0], self.opt.seg_class_type)
        self.edge = decoded_pred

        """seg_input = F.interpolate(seg_input, size=(self.opt.fineSize_H, self.opt.fineSize_W), mode='bilinear', align_corners=True)
                pred_, feature = self.seg_model(seg_input)
                # 勾配を計算しないようにする
                for key in feature.keys():
                    feature[key] = F.interpolate(feature[key].detach(), scale_factor=0.5, mode='bilinear', align_corners=True)
                self.seg_feature = feature
                pred = pred_.detach().max(1)[1][0]
                decoded_pred = self.decode_segmentation_map(pred, self.opt.seg_class_type)
                self.seg_pred = pred
                self.edge = decoded_pred"""

    def create_random_class_mask(self, segmentation_image):
        # Get unique class labels from the segmentation image
        unique_classes = torch.unique(segmentation_image)

        # Remove background class (usually labeled as 0)
        unique_classes = unique_classes[unique_classes != 0]

        # Randomly select a class from the unique classes
        selected_class = random.choice(unique_classes)

        # Create a binary mask where selected class pixels are 1 and others are 0
        mask = (segmentation_image == selected_class).type(torch.uint8)
        return mask, selected_class

    def create_random_class_mask2(self, label_map:torch.Tensor):
        """
        Args:
            label_map (torch.Tensor): 形状 (N, C, H, W) のラベルマップ
        Returns:
            class_mask (torch.Tensor): ランダムに選択された空でないクラスに対応するマスク (N, 1, H, W)
            random_class (int): ランダムに選択されたクラス
        """
        N, C, H, W = label_map.shape

        # 空でないクラスのリストを作成
        non_empty_classes = [c for c in range(C) if label_map[:, c, :, :].sum() > 0]
        #print(f"non_empty_classes: {non_empty_classes}")
        if not non_empty_classes:
            # 全てのクラスが空の場合は何も返さない
            return None

        # ランダムにクラスを選択
        random_class = random.choice(non_empty_classes)
        #print(f"random_class: {random_class}")

        # ラベルマップから選択されたクラスの部分を抽出
        class_mask = label_map[:, random_class, :, :]
        #print(class_mask)

        return class_mask, random_class


    def create_seg_inst_images(self, image, mask):
        # image: (N, C, H, W)
        # mask: (N, H, W)
        # imageにmaskをかける
        inst_image = image * mask
        # maskの反転
        inv_mask = (mask == 0).type(torch.uint8)
        # maskの反転にimageをかける
        bg_image = image * inv_mask
        return inst_image, bg_image

    def decode_segmentation_map(self, pred:torch.Tensor, seg_class_type:str):
        if seg_class_type.lower() == 'cityscapes':
            CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])
            classes = [
                CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
                CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
                CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
                CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
                CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
                CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
                CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
                CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
                CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
                CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
                CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
                CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
                CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
                CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
                CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
                CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
                CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
                CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
                CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
                CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
                CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
                CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
                CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
                CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
                CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
                CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
                CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
                CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
                CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
                CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
                CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
                CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
                CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
                CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
                CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
            ]

            train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
            train_id_to_color.append([0, 0, 0])
            train_id_to_color = torch.tensor(train_id_to_color,dtype=float).to(self.device)
            id_to_train_id = torch.tensor([c.train_id for c in classes])
            pred[pred == 255] = 19
            # 軸を入れ替える
            out = train_id_to_color[pred].permute(2,0,1).unsqueeze(0)
            out = (out / 255) * 2 - 1
            #tmp = out[:, 0, :, :].clone()
            #out[:, 0, :, :] = out[:, 2, :, :]
            #out[:, 2, :, :] = tmp
            return out

    def forward(self, bboxes=0):
        self.seg_pred = None
        if self.opt.use_seg_model and not self.opt.which_model_netG == 'AAFSTNet2andSegNet_NoSeg':
            self.get_segmentation_feature()
            seg_mask, seg_classes = self.create_random_class_mask2(self.seg_pred)
            

        # sodマップの出力
        if self.opt.sal_map == 'sod':
            # sod予測ネットが256*256出力なのでリサイズ
            if self.real_A.shape!=(..., 256,256):
                self.real_A_ = torchvision.transforms.functional.resize(self.real_A, size=(256, 256))
            d1,d2,d3,d4,d5,d6,d7,d8 = self.sod_net(self.real_A_)    # 使うのはd1だけ
            self.sal_ir = d1[:,0,:,:]    # ir画像のsodマップ
            self.sal_ir = self.sal_ir[None, ...]
            if self.opt.normalize_sal:
                self.sal_ir = normPRED(self.sal_ir)
            if self.real_A.shape!=(..., self.opt.fineSize_H, self.opt.fineSize_W):
                self.sal_ir = torchvision.transforms.functional.resize(self.sal_ir, size=(self.opt.fineSize_H, self.opt.fineSize_W))

        ##### AAFSTNet2Sobelはsodマップを受け取る場合は補助入力としてsodマップを用いる          ######
        ##### sodマップを受け取らない場合はsobelエッジを補助入力として用いる                    ######
        ##### netGから出力されるedgeはvisualizeのために使うためのものであり、損失計算に用いない ######
        if 'AAFSTNet2Sobel' == self.opt.which_model_netG:
            if self.opt.input_sod:
                # print(self.real_A.shape, self.sal_ir.shape)
                self.fake_B, self.edge = self.netG(self.real_A, self.sal_ir)
            else:
                self.fake_B, self.edge = self.netG(self.real_A)
            self.edge = torch.cat((self.edge,)*3, dim=1)

        elif 'AAFSTNet2Seg' == self.opt.which_model_netG:
            self.fake_B, self.edge = self.netG(self.real_A, self.seg)

        elif 'AAFSTNet2Seg2' == self.opt.which_model_netG:
            self.fake_B, self.pred_seg, self.edge = self.netG(self.real_A, self.seg)
        
        elif 'AAFSTNet2andSegNet' == self.opt.which_model_netG:
            self.fake_B = self.netG(self.real_A, self.seg_feature, input_size=(self.opt.fineSize_H, self.opt.fineSize_W))

        elif 'AAFSTNet2andSegNet2' == self.opt.which_model_netG:
            if not self.opt.seg_to_FAB:
                self.fake_B = self.netG(self.real_A, self.seg_feature, input_size=(self.opt.fineSize_H, self.opt.fineSize_W), seg_pred=None)
            else:
                self.fake_B = self.netG(self.real_A, self.seg_feature, input_size=(self.opt.fineSize_H, self.opt.fineSize_W), seg_pred=self.seg_pred)

        elif 'AAFSTNet2andSegNet_NoSeg' == self.opt.which_model_netG:
            self.fake_B = self.netG(self.real_A, None, input_size=(self.opt.fineSize_H, self.opt.fineSize_W))

        elif 'AAFSTNet2' in self.opt.which_model_netG:
            self.fake_B, self.edge = self.netG(self.real_A)
            self.edge = torch.cat((self.edge,)*3, dim=1)
        else:
            self.fake_B = self.netG(self.real_A)

        # Nanを置き換える
        if torch.isnan(self.fake_B).any():
            self.fake_B[torch.isnan(self.fake_B.clone())] = 0
        assert not torch.isnan(self.fake_B).any(), 'self.fake_Bにnanが含まれています。'

        # bboxがあるとき(self.opt.stage == 'full'のとき)はinstance切り出しをする
        if not (bboxes == 0) and (self.opt.w_gan_inst > 0 or self.opt.w_l1_inst > 0):
            self.fake_B_inst, self.fake_B_bg = myutl.create_inst_images(self.fake_B, bboxes)
            self.fake_B_inst = self.fake_B_inst.to(self.device)
            self.fake_B_bg = self.fake_B_bg.to(self.device)
        
        # use_seg_modelがTrueかつ、w_gan_segが0より大きいときはinst画像としてseg_maskでmaskされた画像を作成
        if self.opt.use_seg_model and self.opt.w_gan_seg > 0:
            self.fake_B_inst, self.fake_B_bg = self.create_seg_inst_images(self.fake_B, seg_mask)
            self.fake_B_inst = self.fake_B_inst.to(self.device)
            self.fake_B_bg = self.fake_B_bg.to(self.device)
            self.real_A_inst, self.real_A_bg = self.create_seg_inst_images(self.real_A, seg_mask)
            self.real_A_inst = self.real_A_inst.to(self.device)
            self.real_A_bg = self.real_A_bg.to(self.device)
            self.real_B_inst, self.real_B_bg = self.create_seg_inst_images(self.real_B, seg_mask)
            self.real_B_inst = self.real_B_inst.to(self.device)
            self.real_B_bg = self.real_B_bg.to(self.device)

        ##### fakeとrealの両方からsaliency mapを生成する
        ##### self.opt.sal_mapが'saliency'なら視線ベースのsaliency mapを生成する
        ##### 'sod'ならsod予測ネットを用いてsaliency mapを生成する
        if self.opt.sal_map == 'saliency':
            sal_img_feat = self.sal_img_model(self.fake_B, decode=True)
            sal_pla_feat = self.sal_pla_model(self.fake_B, decode=True)
            self.sal_pred = self.sal_dec_model([sal_img_feat, sal_pla_feat])
            if self.opt.normalize_sal:
                self.sal_pred = self.sal_pred / max(torch.max(self.sal_pred).item(), 1e-10)

        elif self.opt.sal_map == 'sod' and self.isTrain:
            # fake_B
            if self.fake_B.shape!=(..., 256, 256) :
                self.fake_B = torchvision.transforms.functional.resize(self.fake_B, size=(256, 256))
            d1,d2,d3,d4,d5,d6,d7,d8 = self.sod_net(self.fake_B)
            self.sal_pred = d1[:,0,:,:]
            self.sal_pred = self.sal_pred[None, ...]
            if self.opt.normalize_sal:
                self.sal_pred = normPRED(self.sal_pred)
            # real_B
            if self.real_B.shape!=(..., 256, 256):
                self.real_B = torchvision.transforms.functional.resize(self.real_B, size=(256, 256))
            d1_r,d2,d3,d4,d5,d6,d7,d8 = self.sod_net(self.real_B)
            self.sal_real = d1_r[:,0,:,:]
            self.sal_real = self.sal_real[None, ...]
            if self.opt.normalize_sal:
                self.sal_real = normPRED(self.sal_real)
    


    def backward_D(self, use_inst=False):
        # self.D_labelが'noisy''noisy_soft'ならランダムにDiscriminatorのラベルを交換
        random1 = np.random.rand()
        if (self.opt.D_label == 'noisy' or self.opt.D_label == 'noisy_soft') and random1 < self.D_label_flip_threshold:
            D_label_flip = True
        else:
            D_label_flip = False
        
        if self.opt.D_label == 'soft' or self.opt.D_label == 'noisy_soft':
            deviation = 0.6 * np.random.rand() - 0.3
            true_label = min(1, 1 + deviation)
            false_label = max(0, deviation)
        else:
            true_label = 1
            false_label = 0

        # Fakeの処理
        if self.use_condition == 1:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        else:
            fake_AB = self.fake_B.clone()
        if self.opt.seg_to_D:
            fake_AB = torch.cat((fake_AB, self.seg_pred), 1)
        if self.opt.which_model_netD == 'unet':
            pred_fake_map, pred_fake_ = self.netD(fake_AB.detach())
            pred_fake = pred_fake_[0][0]
            self.pred_fake_map = torch.cat((pred_fake_map, pred_fake_map, pred_fake_map), dim=1)
        else:
            pred_fake = self.netD(fake_AB.detach())

        # Realの処理
        if self.use_condition == 1:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            real_AB = self.real_B
        if self.opt.seg_to_D:  
            real_AB = torch.cat((real_AB, self.seg_pred), 1)
        if self.opt.which_model_netD == 'unet':
            pred_real_map, pred_real_ = self.netD(real_AB)
            pred_real = pred_real_[0][0]
            self.pred_real_map = torch.cat((pred_real_map, pred_real_map, pred_real_map), dim=1)
            mix_b_x = np.random.randint(0, self.opt.fineSize_W//2)
            mix_b_y = np.random.randint(0, self.opt.fineSize_H//2)
            mixed_img = fake_AB.detach().clone()
            mixed_img_ = self.fake_B.clone()
            mixed_img[..., mix_b_y:mix_b_y+(self.opt.fineSize_H // 2), mix_b_x:mix_b_x+(self.opt.fineSize_W // 2)] = real_AB[..., mix_b_y:mix_b_y+(self.opt.fineSize_H // 2), mix_b_x:mix_b_x+(self.opt.fineSize_W // 2)].clone()
            mixed_img_[..., mix_b_y:mix_b_y+(self.opt.fineSize_H // 2), mix_b_x:mix_b_x+(self.opt.fineSize_W // 2)] = self.real_B[..., mix_b_y:mix_b_y+(self.opt.fineSize_H // 2), mix_b_x:mix_b_x+(self.opt.fineSize_W // 2)].clone()
            self.mixed_img = mixed_img
            self.mixed_img_ = mixed_img_
            pred_mix_map, pred_mix = self.netD(mixed_img)
            self.pred_mix_map = torch.cat((pred_mix_map, pred_mix_map, pred_mix_map), dim=1)
            mixed_pred_map = pred_fake_map.clone()
            mixed_pred_map[..., mix_b_y:mix_b_y+(self.opt.fineSize_H // 2), mix_b_x:mix_b_x+(self.opt.fineSize_W // 2)] = pred_real_map[..., mix_b_y:mix_b_y+(self.opt.fineSize_H // 2), mix_b_x:mix_b_x+(self.opt.fineSize_W // 2)].clone()
            # visual_namesにセグメンテーションマップ群を追加
            self.visual_names.append("mixed_img_")
            self.visual_names.append("pred_fake_map")
            self.visual_names.append("pred_real_map")
            self.visual_names.append("pred_mix_map")
        else:
            pred_real = self.netD(real_AB)

        # loss_Dの計算
        if self.opt.gan_type=='RSGAN':
            # RSGANの場合
            # flip = TrueならTrueとFalseのラベルを交換
            if D_label_flip:
                self.loss_D = self.criterionGAN.rsgan(pred_fake, pred_real)
            else:
                self.loss_D = self.criterionGAN.rsgan(pred_real, pred_fake)
        else:
            # SGANかLSGANの場合
            if D_label_flip:
                self.loss_D_fake = self.criterionGAN(pred_fake, true_label)
                self.loss_D_real = self.criterionGAN(pred_real, false_label)
            else:
                self.loss_D_fake = self.criterionGAN(pred_fake, false_label)
                self.loss_D_real = self.criterionGAN(pred_real, true_label)

            # unet discriminatorの場合はloss_Dにセグメンテーションマップに関するロスを追加
            if self.opt.which_model_netD == 'unet':
                if self.opt.no_D_enc == True:
                    self.loss_D = 0
                # セグメンテーションマップから計算するロス
                self.loss_D_dec_fake = self.criterionGAN_dec(pred_fake_map, false_label)
                self.loss_D_dec_real = self.criterionGAN_dec(pred_real_map, true_label)
                self.loss_D_dec_con = self.opt.w_con * self.criterionL2(pred_mix_map, mixed_pred_map)
                # セグメンテーションマップからのロスをloss_Dに追加
                self.loss_D += self.loss_D_dec_fake
                self.loss_D += self.loss_D_dec_real
                self.loss_D += self.loss_D_dec_con
            self.loss_D = (self.loss_D_fake + self.loss_D_real)

        # GPがある場合はloss_DにGradient Penaltyを追加
        if 'GP' in self.opt.gan_type:
            self.loss_D += networks.cal_gradient_penalty(self.netD, real_AB, fake_AB, self.device, type='mixed', constant=1.0, lambda_gp=self.opt.lambda_gp)[0]
        
        #INST
        if use_inst:
            # fake_instの処理
            if self.use_condition == 1:
                fake_AB_inst = self.fake_AB_pool_inst.query(torch.cat((self.real_A_inst, self.fake_B_inst), 1))
            else: 
                fake_AB_inst = self.fake_B_inst
            if self.opt.seg_to_D_inst: 
                fake_AB_inst = torch.cat((fake_AB_inst, self.seg_pred), 1)
            pred_fake_inst = self.netD_inst(fake_AB_inst.detach())

            # Real_instの処理
            if self.use_condition == 1:
                real_AB_inst = torch.cat((self.real_A_inst, self.real_B_inst), 1)
            else:
                real_AB_inst = self.real_B_inst
            if self.opt.seg_to_D:
                real_AB_inst = torch.cat((real_AB_inst, self.seg_pred), 1)
            pred_real_inst = self.netD_inst(real_AB_inst)

            if self.opt.gan_type=='RSGAN':
                if D_label_flip:
                    self.loss_D_inst = self.nonzero_inst * self.criterionGAN.rsgan(pred_fake_inst, pred_real_inst)
                else:
                    self.loss_D_inst = self.nonzero_inst * self.criterionGAN.rsgan(pred_real_inst, pred_fake_inst)
            else:
                if D_label_flip:
                    self.loss_D_fake_inst = self.nonzero_inst * self.criterionGAN(pred_fake_inst, true_label)
                    self.loss_D_real_inst = self.nonzero_inst * self.criterionGAN(pred_real_inst, false_label)
                else:
                    self.loss_D_fake_inst = self.nonzero_inst * self.criterionGAN(pred_fake_inst, false_label)
                    self.loss_D_real_inst = self.nonzero_inst * self.criterionGAN(pred_real_inst, true_label)
                self.loss_D += (self.loss_D_fake_inst + self.loss_D_real_inst)

            # GPがある場合はloss_DにGradient Penaltyを追加
            if 'GP' in self.opt.gan_type:
                self.loss_D += networks.cal_gradient_penalty(self.netD_inst, real_AB_inst, fake_AB_inst, self.device, type='mixed', constant=1.0, lambda_gp=self.opt.lambda_gp)[0]
                retain_graph = True
            else:
                retain_graph = False
        if self.opt.w_gan <= 0:
            self.opt.loss_D = 0
        self.loss_D.backward(retain_graph=retain_graph)

    def backward_G(self):
        self.loss_G = 0
        ##### GAN lossの計算
        if self.opt.w_gan > 0:
            if self.use_gan == 1:
                # condition画像を使う場合
                if self.use_condition == 1:
                    # 全体のadversarial loss
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    real_AB = torch.cat((self.real_A, self.real_B), 1)
                    # segをDに入力する場合
                    if self.opt.D_input == 'seg':
                        fake_AB = torch.cat((fake_AB, self.seg), 1)
                        real_AB = torch.cat((real_AB, self.seg), 1)
                    # instance部分についてのadversarial lossの計算
                    if self.opt.stage == 'full' and (self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0):
                        # prediction
                        fake_AB_inst = torch.cat((self.real_A_inst, self.fake_B_inst), 1)
                        real_AB_inst = torch.cat((self.real_A_inst, self.real_B_inst), 1)
                        pred_fake_inst = self.netD_inst(fake_AB_inst)
                        pred_real_inst = self.netD_inst(real_AB_inst.detach())
                        # lossの計算
                        if self.opt.gan_type=='RSGAN':
                            self.loss_G_GAN_inst = self.w_gan_seg * self.nonzero_inst * self.criterionGAN.rsgan(pred_fake_inst, pred_real_inst)
                        else:
                            self.loss_G_GAN_inst = self.w_gan_seg * self.nonzero_inst * self.criterionGAN(pred_fake_inst, True)

                else: # condition画像を使わない場合
                    fake_AB = self.fake_B
                    real_AB = self.real_B
                    # segをDに入力する場合
                    if self.opt.seg_to_D:
                        fake_AB = torch.cat((fake_AB, self.seg_pred), 1)
                        real_AB = torch.cat((real_AB, self.seg_pred), 1)
                    # instance部分についてのadversarial lossの計算
                    if self.opt.stage == 'full' and (self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0):
                        fake_AB_inst = self.fake_B_inst
                        real_AB_inst = self.real_B_inst
                        # segをD_instに入力する場合
                        if self.opt.seg_to_D_inst:
                            fake_AB_inst = torch.cat((fake_AB_inst, self.seg_pred), 1)
                            real_AB_inst = torch.cat((real_AB_inst, self.seg_pred), 1)
                        pred_fake_inst = self.netD_inst(fake_AB_inst)
                        pred_real_inst = self.netD_inst(real_AB_inst.detach())
                        # loss_G_GAN_instの計算
                        if self.opt.gan_type=='RSGAN':
                            self.loss_G_GAN_inst = self.w_gan_seg * self.nonzero_inst * self.criterionGAN.rsgan(pred_fake_inst, pred_real_inst)
                        else:
                            self.loss_G_GAN_inst = self.w_gan_seg * self.nonzero_inst * self.criterionGAN(pred_fake_inst, True)

                # unet discriminatorを使う場合はセグメンテーションマップを出力
                if self.opt.which_model_netD == 'unet':
                    pred_fake_map, pred_fake = self.netD(fake_AB)
                    pred_real_map, pred_real = self.netD(real_AB.detach())
                else:
                    pred_fake = self.netD(fake_AB)
                    pred_real = self.netD(real_AB.detach())

                # RSGANの場合のlossの計算
                if self.opt.gan_type=='RSGAN':
                    self.loss_G_GAN = self.w_gan * self.criterionGAN.rsgan(pred_fake, pred_real) 
                # SGANの場合のlossの計算
                else:
                    # unet discriminatorを使う場合はdecoder分のロスを計算
                    if self.opt.which_model_netD == 'unet':
                        pred_fake = pred_fake[0][0]
                        self.loss_G_GAN_dec = self.w_gan * self.criterionGAN_dec(pred_fake_map, True)
                        self.loss_G += self.loss_G_GAN_dec
                    # loss_G_GANの計算
                    # no_D_encの場合はencoder分のロスを0にする
                    if self.opt.which_model_netD == 'unet' and self.opt.no_D_enc == True:
                        self.loss_G_GAN = 0
                    else:
                        self.loss_G_GAN = self.w_gan * self.criterionGAN(pred_fake, True)
            else:
                self.loss_G_GAN = 0
            assert not torch.isnan(self.loss_G_GAN).any(), 'loss_G_GANにnanが含まれています。'
            self.loss_G += self.loss_G_GAN

        ##### L1 lossの計算
        # stageがfullの場合は背景領域と物体領域を分けてL1の計算をする
        if self.opt.w_l1 > 0:
            if self.opt.stage == 'full':
                if self.opt.l1_bgOnly == True:
                    # 背景領域のL1 loss
                    # train_channelの設定によって輝度のみ、色度のみ、全てのチャンネルを使うかを決める
                    if self.opt.colorspace == 'Lab' or self.opt.colorspace == 'YCbCr':
                        if self.opt.l1_channel == 'all':
                            self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B_bg, self.real_B_bg)
                        elif self.opt.l1_channel == 'luminance' :
                            self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B_bg[:, :1, :, :], self.real_B_bg[:, :1, :, :])
                        elif self.opt.l1_channel == 'chrominance':
                            self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B_bg[:, 1:, :, :], self.real_B_bg[:, 1:, :, :])
                    elif self.opt.colorspace == 'HSV':
                        if self.opt.l1_channel == 'all':
                            self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B_bg, self.real_B_bg)
                        elif self.opt.l1_channel == 'luminance' :
                            self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B_bg[:, 2:, :, :], self.real_B_bg[:, 2:, :, :])
                        elif self.opt.l1_channel == 'chrominance':
                            self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B_bg[:, :2, :, :], self.real_B_bg[:, :2, :, :])
                    else:
                        self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B_bg, self.real_B_bg)

                    # 物体領域のL1 loss
                    # 各色空間の輝度チャネルのみを使用する
                    # 色空間がRGBの場合は全チャネルを使用する
                    if self.opt.colorspace == 'Lab' or self.opt.colorspace == 'YCbCr':
                        self.loss_G_L1 += self.w_l1 * self.criterionL1(self.fake_B_inst[:, :1, :, :], self.real_B_inst[:, :1, :, :])
                    elif self.opt.colorspace == 'HSV':
                        self.loss_G_L1 += self.w_l1 * self.criterionL1(self.fake_B_inst[:, 2:, :, :], self.real_B_inst[:, 2:, :, :])
                    else:
                        self.loss_G_L1 += self.w_l1 * self.criterionL1(self.fake_B_inst, self.real_B_inst)
                else:
                    self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B, self.real_B)
            else:
                self.loss_G_L1 = self.w_l1 * self.criterionL1(self.fake_B, self.real_B)

            assert not torch.isnan(self.loss_G_L1).any(), 'loss_G_L1にnanが含まれています。'
            self.loss_G += self.loss_G_L1

        ##### vgg lossの計算
        if self.opt.w_vgg > 0:
            self.real_B_features = self.vgg(self.real_B)
            self.fake_B_features = self.vgg(self.fake_B)
            self.loss_vgg = self.w_vgg * (self.criterionL1(self.fake_B_features[1], self.real_B_features[1])*1 \
                                        + self.criterionL1(self.fake_B_features[2], self.real_B_features[2])*1 \
                                        + self.criterionL1(self.fake_B_features[3], self.real_B_features[3])*1 \
                                        + self.criterionL1(self.fake_B_features[0], self.real_B_features[0])*1)
            assert not torch.isnan(self.loss_vgg).any(), 'loss_vggにnanが含まれています。'
            self.loss_G += self.loss_vgg

        ##### TV lossの計算
        if self.opt.w_tv > 0:
            diff_i = torch.sum(torch.abs(self.fake_B[:, :, :, 1:] - self.fake_B[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(self.fake_B[:, :, 1:, :] - self.fake_B[:, :, :-1, :]))
            self.loss_tv = ((diff_i + diff_j) / (256 * 256)) * self.w_tv   
            assert not torch.isnan(self.loss_tv).any(), 'loss_tvにnanが含まれています。' 
            self.loss_G += self.loss_tv   

        ##### ssim lossの計算
        if self.opt.w_ssim > 0:
            # 画像のレンジを0~1に変換
            X = (self.real_B + 1) / 2
            Y = (self.fake_B + 1) / 2
            self.loss_ssim = self.w_ssim * (1 - ssim(X, Y, data_range=1, size_average=True))
            assert not torch.isnan(self.loss_ssim).any(), 'loss_ssimにnanが含まれています。' 
            self.loss_G += self.loss_ssim

        ##### stageをinstanceにして実験するな #####
        # stageがinstanceの場合はGAN lossのみ使用
        if self.opt.stage == 'instance':
            self.loss_G = self.loss_G_GAN  

        ##### edge lossがある場合は加算
        if self.opt.w_edge > 0.00001:
            self.edge_real = self.ttb(self.real_A)
            self.edge_fake = self.ttb(self.fake_B)
            self.loss_edge = self.w_edge * self.criterionL1(self.edge_fake, self.edge_real)
            self.loss_G += self.loss_edge
            assert not torch.isnan(self.loss_G).any(), 'loss_edgeにnanが含まれています。'    
            #self.edge_real_vis = torch.cat((self.edge_real, self.edge_real, self.edge_real), dim=1)
            #self.edge_fake_vis = torch.cat((self.edge_fake, self.edge_fake, self.edge_fake), dim=1)
            self.visual_names.append("edge_real")
            self.visual_names.append("edge_fake")

        ##### Color Diversity lossの計算
        # HSVのSとVについてペナルティを与える
        # Hの分散を小さくすることで色相の一貫性を保つ
        # Sの分散を小さくすることで彩度の一貫性を保つ
        if self.opt.w_cd > 0 and self.opt.stage == 'full':
            # fake_Bの物体領域を二値抽出
            sal_real_binaries = torch.where(self.sal_real > 0.3, input=torch.ones_like(self.sal_real),
                                             other=torch.zeros_like(self.sal_real))   # (N, 1, H, W)
            self.fake_B_masked = sal_real_binaries * self.fake_B.clone()

            # fakeの各チャネルの平均値を計算し、平均値マップmean_map_maskedを作成
            mean_map_masked = torch.zeros_like(self.fake_B)
            for i in range(3):
                fake_B_mean = torch.sum(self.fake_B_masked[:, i:i+1, :, :]) / (torch.sum(sal_real_binaries) + 1e-7)
                fake_B_mean_map = torch.full_like(self.fake_B[:, i:i+1, :, :], fill_value=fake_B_mean.item())
                mean_map_masked[:, i:i+1, :, :] += sal_real_binaries * fake_B_mean_map
                
            # マスクしたfake_Bと平均値マップのL1 lossを計算
            if self.opt.colorspace == 'HSV':
                # HとSのみ計算
                self.loss_cd = self.criterionL1(mean_map_masked[:, :2, :, :], self.fake_B_masked[:, :2, :, :]) * self.w_cd
            elif self.opt.colorspace == 'Lab' or self.opt.colorspace == 'YCbCr':
                # aとbのみ計算
                self.loss_cd = self.criterionL1(mean_map_masked[:, 1:, :, :], self.fake_B_masked[:, 1:, :, :]) * self.w_cd
            else:
                # RGBの場合は全チャネル計算
                self.loss_cd = self.criterionL1(mean_map_masked, self.fake_B_masked) * self.w_cd

            self.loss_G += self.loss_cd

        ##### seg lossがある場合はloss_Gに加算
        if self.opt.w_seg > 0:
            self.loss_seg = self.w_seg * self.criterionL1(self.pred_seg, self.seg)
            self.loss_G += self.loss_seg
            assert not torch.isnan(self.loss_seg).any(), 'loss_segにnanが含まれています。'

        ##### reg_lossがある場合はloss_Gに加算
        if self.opt.w_reg > 0:
            self.loss_G += self.loss_reg
            assert not torch.isnan(self.loss_G).any(), 'loss_regにnanが含まれています。'  

        ##### ssim lossがある場合はloss_Gに加算
        if self.opt.w_ssim > 0:
            self.loss_G += self.loss_ssim
            assert not torch.isnan(self.loss_G).any(), 'loss_ssimにnanが含まれています。'

        ##### cx lossがある場合はloss_Gに加算
        #if self.opt.w_cx > 0:
            #self.loss_cx = self.criterion_cx(self.fake_B, self.real_B)
            #self.loss_cx = self.w_cx * CX_loss(self.real_B, self.fake_B)
            #self.loss_cx = clf.contextual_loss(self.real_B, self.fake_B, band_width=0.1, loss_type='cosine')
            #self.loss_G += self.loss_cx 

        ##### sal lossがある場合はloss_Gに加算
        if self.opt.w_sal > 0 and not(self.opt.sal_map=='None'):
            if self.opt.sal_map=='saliency':
                self.masked_sal_pred = self.bbox_mask * self.sal_pred
                self.loss_sal = self.opt.w_sal * self.criterionL2(self.bbox_mask, self.masked_sal_pred)
                self.visual_names.append("sal_pred")
                self.visual_names.append("masked_sal_pred")
                self.visual_names.append("bbox_mask")
                self.masked_sal_pred = torch.cat((self.masked_sal_pred,self.masked_sal_pred,self.masked_sal_pred), dim=1)
            elif self.opt.sal_map=='sod':
                self.loss_sal = self.opt.w_sal * self.criterionL2(self.sal_real, self.sal_pred)
                self.visual_names.append("sal_pred")
                self.visual_names.append("sal_real")
                self.sal_real = torch.cat((self.sal_real,self.sal_real,self.sal_real), dim=1)
            self.loss_G += self.loss_sal

            self.sal_pred = torch.cat((self.sal_pred,self.sal_pred,self.sal_pred), dim=1)
            self.bbox_mask = torch.cat((self.bbox_mask, self.bbox_mask, self.bbox_mask), dim=1)
            assert not torch.isnan(self.loss_sal).any(), 'loss_salにnanが含まれています。'
            assert not torch.isnan(self.loss_G).any(), 'loss_salにnanが含まれています。'

        ##### instance部分のadversarial lossをloss_Gに加算
        if self.opt.stage == 'full':
            # bboxがあればinst損失を加算するw > 0の損失を加算する
            if (not self.skip) and self.use_inst:
                # vgg_inst
                if self.opt.w_vgg_inst > 0:
                    self.fake_B_inst_features = self.vgg(self.fake_B_inst)
                    self.real_B_inst_features = self.vgg(self.real_B_inst)
                    self.loss_vgg_inst = self.w_vgg_inst * self.criterionL1(self.fake_B_inst_features[1], self.real_B_inst_features[1])*1 \
                                                + self.criterionL1(self.fake_B_inst_features[2], self.real_B_inst_features[2])*1 \
                                                + self.criterionL1(self.fake_B_inst_features[3], self.real_B_inst_features[3])*1 \
                                                + self.criterionL1(self.fake_B_inst_features[0], self.real_B_inst_features[0])*1
                    self.loss_G += self.loss_vgg_inst
                    assert not torch.isnan(self.loss_G).any(), 'loss_vgg_instにnanが含まれています。'
                # l1_inst
                if self.opt.w_l1_inst > 0:
                    self.loss_l1_inst = self.w_l1_inst * self.criterionL1(self.fake_B_inst, self.real_B_inst)
                    self.loss_G += self.loss_l1_inst
                    assert not torch.isnan(self.loss_G).any(), 'loss_l1_instにnanが含まれています。'
                # cx_inst
                #if self.opt.w_cx_inst > 0:
                    #self.loss_cx_inst = self.criterion_cx_inst(self.fake_B_inst, self.real_B_inst)
                    #self.loss_cx_inst = self.w_cx_inst * CX_loss(self.fake_B_inst_features[0], self.real_B_inst_features[0])
                    #self.loss_cx_inst = clf.contextual_loss(self.real_B_inst, self.fake_B_inst, band_width=0.1, loss_type='cosine')
                    #self.loss_G += self.loss_cx_inst[0]
                # gan_inst
                if self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0:
                    self.loss_G += self.loss_G_GAN_inst
                    assert not torch.isnan(self.loss_G).any(), 'loss_G_GAN_instにnanが含まれています。'

                
        assert not torch.isnan(self.loss_G).any(), 'loss_Gにnanが含まれています。'
        self.loss_G.backward()

    def optimize_parameters(self, bbox=None, skip=False):
        #with torch.autograd.detect_anomaly():
            self.skip = skip
            # fullの場合はbboxを入力する
            self.forward(bboxes=bbox)

            # Dの更新
            # WGAN-GPの場合はDを複数回更新
            if self.opt.gan_type == 'WGANGP':
                n_critic = self.opt.n_critic
            else:
                n_critic = 1
            if self.opt.stage == 'full' and (self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0):
                use_inst = True
            for i in range(n_critic):
                self.set_requires_grad(self.netD, True)
                self.optimizer_D.zero_grad()
                self.backward_D(use_inst=use_inst)
                self.optimizer_D.step()

            # Gを更新
            self.set_requires_grad(self.netD, False)
            if self.opt.stage == 'full' and (self.opt.w_gan_inst > 0 or self.opt.w_gan_seg > 0):
                self.set_requires_grad(self.netD_inst, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
