import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image, ImageCms, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor, adjust_contrast
from models.networks import ImageSharpening
from util.util import apply_clahe_to_cv2BGR, cv2BGR_to_pil, pil_to_cv2BGR
from .lab2rgb import rgb2lab
import kornia as K
from util.util import change_colorspace, change_colorspace_to_rgb

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.sharp_filter = ImageSharpening()
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)

        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize_W, self.opt.loadSize_H), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize_W, self.opt.loadSize_H), Image.BICUBIC)

        if self.opt.A_invert:
            A = ImageOps.invert(A)

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize_W - self.opt.fineSize_W - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize_H - self.opt.fineSize_H - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize_H, w_offset:w_offset + self.opt.fineSize_W]
        B = B[:, h_offset:h_offset + self.opt.fineSize_H, w_offset:w_offset + self.opt.fineSize_W]
        if int(self.opt.contrast_factor) != 1:
            A = adjust_contrast(A, contrast_factor=self.opt.contrast_factor)
        if self.opt.sharp_input:
            A = self.sharp_filter(A)

        A = change_colorspace(A, self.opt.colorspace)
        B = change_colorspace(B, self.opt.colorspace)

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(torch.clip(A, 0, 1))
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path, 'skip': False}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

