import numpy as np
from data.image_folder import make_dataset
from PIL import Image, ImageFilter
import torch.utils.data as Data
import torchvision.transforms.functional as F
from os import listdir
from os.path import isfile, join
from random import sample
from .image_util import *
import torchvision.transforms as transforms
import cv2
import kornia as K
from pathlib import Path
from util.util import change_colorspace, make_transform_list, set_random_resize_param, split_image_to_patches, mask_random_patches


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

# test時に使ったりする
class Global_Dataset_TIR(Data.Dataset):
    def initialize(self, opt):
            self.opt = opt
            if self.opt.transform == 'randomcrop':
                if opt.fineSize_W < opt.loadSize_W:
                    self.opt.loadSize_W = np.random.randint(opt.fineSize_W, opt.loadSize_W)
                else:
                    self.opt.loadSize_W = opt.loadSize_W
                self.opt.loadSize_H = self.opt.loadSize_W
            self.IMAGE_DIR_A = opt.dataroot_A
            self.IMAGE_DIR_B = opt.dataroot_B
            self.A_paths = sorted(make_dataset(self.IMAGE_DIR_A))
            self.B_paths = sorted(make_dataset(self.IMAGE_DIR_B))
            self.A_size = len(self.A_paths)
            self.B_size = len(self.B_paths)
            self.IMAGE_ID_LIST = sorted([f for f in listdir(self.IMAGE_DIR_A) if isfile(join(self.IMAGE_DIR_A, f))])


            # transform
            self.transform_list = make_transform_list(self.opt)

    def __getitem__(self, index):
        skip = False
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        t = transforms.ToTensor()
        
        if Path(A_path).name != Path(B_path).name and self.opt.phase == 'train':
            print(f'A : {Path(A_path).name}')
            print(f'B : {Path(B_path).name}')
            raise Exception('AとBに一致対応していません。')

        if self.opt.out_gray:
            A_img = t(Image.open(A_path).convert('RGB'))
            B_img = t(Image.open(B_path).convert('RGB'))
        else:
            A_img = t(Image.open(A_path).convert('RGB'))
            B_img = t(Image.open(B_path).convert('RGB'))

        # CLAHEの適用
        if self.opt.clahe:
            A_img = K.enhance.equalize_clahe(input=A_img, clip_limit=2.0, grid_size=(8,8))

        # 色空間の変換
        A_img = change_colorspace(A_img ,self.opt.colorspace)
        B_img = change_colorspace(B_img ,self.opt.colorspace)

        # mirror
        num_random_mirror = np.random.rand()
        if self.opt.mirror and num_random_mirror < 0.5: self.flip = True
        else: self.flip = False
        if self.flip:
            A_img = K.geometry.transform.hflip(A_img)
            B_img = K.geometry.transform.hflip(B_img)

        # 輝度の反転
        num_random_inv = np.random.rand()
        if num_random_inv <= self.opt.A_invert_prob: input_invert = True
        else: input_invert = False
        if self.opt.A_invert and input_invert:
            A_img = K.enhance.invert(A_img)

        # transformの適用
        A = self.transform_list(A_img)[0]
        B = self.transform_list(B_img)[0]

        # randomcrop処理
        num_random_crop = np.random.rand()
        if self.opt.transform == 'randomcrop' and (num_random_crop <= self.opt.t_prob):
            # crop座標の取得とcrop
            crop_i, crop_j, crop_h, crop_w = transforms.RandomCrop.get_params(A, output_size=(self.opt.fineSize_H, self.opt.fineSize_W))
            A = F.crop(A, crop_i, crop_j, crop_h, crop_w)
            B = F.crop(B, crop_i, crop_j, crop_h, crop_w)
            
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path, 'skip': skip}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'Global_Dataset_TIR'
    

class Full_Dataset_TIR(Data.Dataset):
    def initialize(self, opt):
        self.opt = opt        
        # パスの読み込み
        self.IMAGE_DIR_A = opt.dataroot_A   # ir画像
        self.IMAGE_DIR_B = opt.dataroot_B   # rgb画像
        self.A_paths = sorted(make_dataset(self.IMAGE_DIR_A))   # ir画像のパスのリスト
        self.B_paths = sorted(make_dataset(self.IMAGE_DIR_B))   # rgb画像のパスのリスト
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.IMAGE_ID_LIST = sorted([f for f in listdir(self.IMAGE_DIR_A) if isfile(join(self.IMAGE_DIR_A, f))])
        if opt.use_segmap:
            self.seg_dir = opt.seg_dir          # セグメンテーション画像
            self.seg_paths = sorted(make_dataset(self.seg_dir))     # セグメンテーション画像のパスのリスト
            self.seg_size = len(self.seg_paths)

        # bboxの抽出
        self.PRED_BBOX_DIR = '{0}_bbox'.format(opt.dataroot_A)

        # transformの設定
        self.transforms = make_transform_list(self.opt)


    def __getitem__(self, index):
        # 画像パスの取得
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        if (Path(A_path).name != Path(B_path).name) and self.opt.isTrain:
            print(f'A : {Path(A_path).name}')
            print(f'B : {Path(B_path).name}')
            raise Exception('画像A,Bのいずれかが一致していません')

        # 画像の読み込み
        to_tensor = transforms.ToTensor()
        A_img = to_tensor(Image.open(A_path).convert('RGB'))
                
        if self.opt.l1_channel == 'luminance':
            B_img = to_tensor(Image.open(B_path).convert('L').convert('RGB'))
        else:
            B_img = to_tensor(Image.open(B_path).convert('RGB'))
        
        # 事前に用意されたセグメンテーション画像を用いる場合
        if self.opt.use_segmap:
            num_random_seg = np.random.rand()
            if num_random_seg > self.opt.use_segmap_prob:
                # num_random_segがuse_segmap_probより大きい場合は、セグメンテーション画像を使用しない
                seg_img = torch.ones_like(B_img)
            else:
                seg_path = self.seg_paths[index % self.seg_size]
                seg_img = Image.open(seg_path).convert('RGB')
                if self.opt.seg_filter:    # セグメンテーション画像にガウシアンフィルタをかける
                    filter = ImageFilter.GaussianBlur(radius=self.opt.seg_filter_radius)
                    seg_img = seg_img.filter(filter)
                seg_img = mask_random_patches(seg_img, self.opt.seg_mask_size, self.opt.seg_mask_ratio)
                seg_img = to_tensor(seg_img)

        # CLAHEの適用
        if self.opt.clahe:
            A_img = K.enhance.equalize_clahe(input=A_img, clip_limit=2.0, grid_size=(8,8))
            
        # 色空間の変換
        A_img = change_colorspace(A_img ,self.opt.colorspace)
        B_img = change_colorspace(B_img ,self.opt.colorspace)
        if self.opt.use_segmap:
            seg_img = change_colorspace(seg_img ,self.opt.colorspace)

        # mirror
        # 50%の確率で左右反転
        num_random_mirror = np.random.rand()
        if self.opt.mirror and num_random_mirror < 0.5:self.flip = True
        else: self.flip = False
        if self.flip:
            A_img = K.geometry.transform.hflip(A_img)
            B_img = K.geometry.transform.hflip(B_img)
            if self.opt.use_segmap:
                seg_img = K.geometry.transform.hflip(seg_img)

        # 閾値を超えるサイズのbboxだけを残すようにフィルタリング&クロップサイズに合わせてリサイズ
        skip = False   # 変数初期化
        if self.opt.isTrain and self.opt.sal_mask:
            pred_info_path = join(self.PRED_BBOX_DIR, self.IMAGE_ID_LIST[index].split('.')[0] + '.npz')
            pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path)
            thresholds = [A_img.shape[1] // self.opt.bbox_threshold, A_img.shape[2] // self.opt.bbox_threshold]
            bboxes_tensor, skip = filter_and_resize_bboxes(pred_bbox, thresholds=thresholds, loadsize_h=self.opt.loadSize_H, loadsize_w=self.opt.loadSize_W, height=A_img.shape[1], width=A_img.shape[2])
        else:
            bboxes_tensor = 0

        # 輝度の反転
        # A_invert_probの確率でA画像の輝度を反転
        num_random_inv = np.random.rand()
        if self.opt.A_invert and (num_random_inv <= self.opt.A_invert_prob):
            A_img = K.enhance.invert(A_img)

        # transformsの適用
        A = self.transforms(A_img)[0]
        B = self.transforms(B_img)[0]
        if self.opt.use_segmap:
            seg = self.transforms(seg_img)[0]
        
        if (self.opt.fineSize_W < self.opt.loadSize_W) or (self.opt.fineSize_H < self.opt.loadSize_H):
            load_H, load_W = set_random_resize_param(min_size_w=self.opt.fineSize_W, min_size_h=self.opt.fineSize_H, max_size_w=self.opt.loadSize_W, max_size_h=self.opt.loadSize_H)
        else:
            load_H, load_W = self.opt.loadSize_H, self.opt.loadSize_W
        A = F.resize(A, (load_H, load_W))
        B = F.resize(B, (load_H, load_W))
        if self.opt.use_segmap:
            seg = F.resize(seg, (load_H, load_W))
        
        # randomcrop処理
        num_random_crop = np.random.rand()
        if self.opt.transform == 'randomcrop' and num_random_crop <= self.opt.t_prob:
            # crop座標の取得とcrop
            crop_i, crop_j, crop_h, crop_w = transforms.RandomCrop.get_params(A, output_size=(self.opt.fineSize_H, self.opt.fineSize_W))
            A = F.crop(A, crop_i, crop_j, crop_h, crop_w)
            B = F.crop(B, crop_i, crop_j, crop_h, crop_w)
            if self.opt.use_segmap:
                seg = F.crop(seg, crop_i, crop_j, crop_h, crop_w)
            if self.opt.isTrain:
                # bboxをcrop座標に合わせてトリミング
                if bboxes_tensor != 0:
                    if self.flip:
                        bboxes_tensor = flip_bboxes(bboxes_tensor, self.opt.loadSize_W)
                    bboxes_tensor = crop_bboxes(bboxes_tensor, [crop_i, crop_j, crop_h, crop_w])

        if self.opt.use_segmap:
            return {'A': A, 'B': B, 'seg':seg, 'A_paths': A_path, 'B_paths': B_path, 'skip': skip, 'bbox': bboxes_tensor}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'skip': skip, 'bbox': bboxes_tensor}

    def __len__(self):
        return len(self.IMAGE_ID_LIST)

    def name(self):
        return 'Full_Dataset_TIR'