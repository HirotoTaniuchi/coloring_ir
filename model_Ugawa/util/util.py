from __future__ import print_function
from curses import keyname
import torch
import numpy as np
from PIL import Image
import os
import models.resnet_for_sal
import models.decoder
import cv2
import kornia as K 
import random

def split_image_to_patches(image, patch_size):
    # 画像のサイズを取得
    width, height = image.size

    # パッチの数を計算
    num_patches = (width // patch_size) * (height // patch_size)

    # パッチのリストを作成
    patches = []

    # 画像をパッチに分割
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches

def mask_random_patches(image, patch_size, mask_ratio):
    # 画像をパッチに分割
    patches = split_image_to_patches(image, patch_size)

    # マスクするパッチの数を計算
    num_masked_patches = int(len(patches) * mask_ratio)

    # ランダムにパッチをマスク
    masked_patches = []
    for patch in patches:
        if num_masked_patches > 0 and random.random() < mask_ratio:
            # パッチをマスク
            masked_patch = Image.new("RGB", (patch_size, patch_size), color=(255, 255, 255))
            masked_patches.append(masked_patch)
            num_masked_patches -= 1
        else:
            masked_patches.append(patch)

    # マスクされたパッチを結合して画像を作成
    masked_image = Image.new("RGB", image.size)
    width, height = image.size
    x, y = 0, 0
    for patch in masked_patches:
        masked_image.paste(patch, (x, y))
        x += patch_size
        if x >= width:
            x = 0
            y += patch_size

    return masked_image

def make_loss_names_list(opt):
    loss_names = ['G']
    if opt.w_l1 > 0:
        loss_names.append('G_L1')
    
    if opt.w_gan > 0:
        loss_names.append('G_GAN')
        loss_names.append('D')
        if opt.gan_type!='RSGAN':
            loss_names.append('D_real')
            loss_names.append('D_fake')        
            # unet gan
            if opt.which_model_netD == 'unet':
                loss_names.append('D_dec_fake')
                loss_names.append('D_dec_real')
                loss_names.append('D_dec_con')
                loss_names.append('G_GAN_dec')

    if opt.w_vgg > 0:
        loss_names.append('vgg')

    if opt.w_tv > 0:
        loss_names.append('tv')
    
    if opt.w_edge > 0: 
        loss_names.append('edge')

    # saliencyマップに関する損失
    # l1で計算
    if opt.w_sal > 0:
        loss_names.append('sal')

    if opt.w_cd > 0:
        loss_names.append('cd')
    
    if opt.w_seg > 0:
        loss_names.append('seg')
    
    if opt.w_ssim > 0:
        loss_names.append('ssim')
        
    return loss_names

def make_transform_list(opt):
    """
    opt.transformに応じて、transformsのシーケンスを作成する。
    """
    if opt.transform == 'centercrop':
        transforms = K.augmentation.AugmentationSequential(
            K.augmentation.Resize(size=(opt.loadSize_H, opt.loadSize_W)),
            K.augmentation.CenterCrop(size=(opt.fineSize_H, opt.fineSize_W)),
            K.enhance.Normalize(mean=(0.5,)*opt.input_nc, std=(0.5,)*opt.input_nc),
            same_on_batch=False)
    elif opt.transform == 'randomcrop':
        transforms = K.augmentation.AugmentationSequential(
            K.augmentation.Resize(size=(opt.loadSize_H, opt.loadSize_W)),
            K.enhance.Normalize(mean=(0.5,)*opt.input_nc, std=(0.5,)*opt.input_nc),
            same_on_batch=False)
    print(opt.transform)
    return transforms

def set_random_resize_param(min_size_w, min_size_h, max_size_w, max_size_h):
    if (min_size_w == min_size_h) and (max_size_w == max_size_h):
        loadSize_H = np.random.randint(low=min_size_h, high=max_size_h)
        loadSize_W = loadSize_H
    else:
        loadSize_H = np.random.randint(low=min_size_h, high=max_size_h)
        loadSize_W = np.random.randint(low=min_size_w, high=max_size_w)
    return loadSize_H, loadSize_W

def change_colorspace(rgb, colorspace, normalize=True):
    """
    rgbのtensorをcolorspaceで指定された色空間に変換する。
    rgb: torch.tensor, shape=(3, H, W) range=[0,1]
    colorspace: str, ['RGB', 'Lab', 'YCbCr', 'HSV']
    normalize: bool, Trueの場合、値を[0,1]に正規化する。
    """
    if colorspace == 'RGB':
        return rgb
    elif colorspace == 'Lab':
        lab = K.color.rgb_to_lab(rgb)
        if normalize:
            lab[0] = lab[0]/100    # L channel rangeを[0,100]から[0,1]へ変換
            lab[1] = (lab[1] + 128) / 255    # a channel rangeを[-128, 127]から[0,1]へ変換
            lab[2] = (lab[2] + 128) / 255    # b channel rangeを[-128, 127]から[0,1]へ変換
        return lab
    elif colorspace == 'YCbCr':
        # YCbCrはnomralize不要
        ycbcr = K.color.rgb_to_ycbcr(rgb)
        return ycbcr
    elif colorspace == 'HSV':
        hsv = K.color.rgb_to_hsv(rgb)
        if normalize:
            hsv[0] = hsv[0] / (2 * torch.pi)
            hsv[1] = hsv[1] 
            hsv[2] = hsv[2] 
        return hsv
    else:
        raise Exception('opt.colorspaceが不正です。')
    
def change_colorspace_to_rgb(imgs, colorspace, normalize=True):
    imgs = imgs.detach()
    if len(imgs.shape) == 4:    # (N, C, H, W)のとき
        out_imgs = torch.zeros_like(imgs)
        for i, img in enumerate(imgs):
            if colorspace == 'RGB':
                out_imgs[i] = img
            elif colorspace == 'Lab':
                if normalize:
                    img[0] = img[0].clone() * 100    # L channel rangeを[0,100]から[0,1]へ変換
                    img[1] = img[1].clone() * 255 - 128    # a channel rangeを[-128, 127]から[0,1]へ変換
                    img[2] = img[2].clone() * 255 - 128    # b channel rangeを[-128, 127]から[0,1]へ変換
                out_imgs[i] = K.color.lab_to_rgb(img)
            elif colorspace == 'YCbCr':
                out_imgs[i] = K.color.ycbcr_to_rgb(img)
            elif colorspace == 'HSV':
                if normalize:
                    img[0] = img[0].clone() * 2 * torch.pi
                    img[1] = img[1].clone() 
                    img[2] = img[2].clone()
                out_imgs[i] = K.color.hsv_to_rgb(img)
            else:
                raise Exception('opt.colorspaceが不正です。')
        return out_imgs
    
    elif len(imgs.shape)==3:    # (C, H, W)のとき
        img = imgs    
        if colorspace == 'RGB':
            return img
        elif colorspace == 'Lab':
            if normalize:
                img[0] = img[0] * 100    # L channel rangeを[0,100]から[0,1]へ変換
                img[1] = img[1] * 255 - 128    # a channel rangeを[-128, 127]から[0,1]へ変換
                img[2] = img[2] * 255 - 128    # b channel rangeを[-128, 127]から[0,1]へ変換
            return K.color.lab_to_rgb(img)
        elif colorspace == 'YCbCr':
            return K.color.ycbcr_to_rgb(img)
        elif colorspace == 'HSV':
            if normalize:
                img[0] = img[0].clone() * 2 * torch.pi
                img[1] = img[1].clone() * 2 * torch.pi
                img[2] = img[2].clone()
            return K.color.hsv_to_rgb(img)
        else:
            raise Exception('opt.colorspaceが不正です。')
        
    else:
        raise Exception('imgsのshapeが不正です。(N, C, H, W)か(C. H, W)の画像を入力してください。')


def pil_to_cv2BGR(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    else:
        raise Exception("pil2cvに与えられたチャネル数が適切ではありません。")
    return new_image

def cv2BGR_to_pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def apply_clahe_to_cv2BGR(img,cl,gsize):
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=gsize)
    bgr = img
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def create_sal_nets(opt):
    print(type(opt))
    img_model = models.resnet_for_sal.resnet50(opt.sal_img_path).eval().requires_grad_(False)
    pla_model = models.resnet_for_sal.resnet50(opt.sal_pla_path).eval().requires_grad_(False)
    decoder_model = models.decoder.build_decoder(opt.sal_dec_path, (opt.fineSize_W, opt.fineSize_H), opt.sal_num_feat, opt.sal_num_feat).eval().requires_grad_(False)
    return img_model, pla_model, decoder_model

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) * 255), 0, 255)
    return image_numpy.astype(imtype)


def create_inst_images(image, bboxes):
    """
    image: (..., H, W)
    bboxes: (N, 4)
    imageからbboxesの領域を除いた画像を返す
    """
    img_inst = torch.zeros_like(image)
    img_background = image.clone()
    if bboxes != 0:
        for bbox in bboxes:
            img_inst[..., bbox[1]:bbox[3], bbox[0]:bbox[2]] = image[..., bbox[1]:bbox[3], bbox[0]:bbox[2]].clone()
            img_background[..., bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
    return img_inst, img_background

def create_bbox_mask_from_bbox(img_shape, bboxes):
    sal_mask = torch.zeros(img_shape)
    # print("bboxes.shape",bboxes.shape)
    # print("bboxes",bboxes)
    if not torch.all(bboxes == 0):
        for bbox in bboxes:
            sal_mask[..., bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            #sal_mask[..., 0:1, 0:1] = 1
    return sal_mask

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        print(path)
        os.makedirs(path)
