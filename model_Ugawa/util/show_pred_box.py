from pathlib import Path
import numpy as np
from data.image_util import *
import os
import glob
from PIL import Image, ImageDraw
import torch, torchvision

def show_bbox(img, bboxes, thresholds):
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        startx, starty, endx, endy = bbox
        len_x = abs(endx - startx)
        len_y = abs(endy - starty)
        if (len_x > thresholds[0]) and (len_y > thresholds[1]): 
            draw.rectangle(((startx, starty), (endx, endy)), outline='red')
        else:
            draw.rectangle(((startx, starty), (endx, endy)), outline='blue')
    return img

def show_only_bboxed_pixel(img, bboxes):
    img_np = np.array(img)
    out_np = np.zeros_like(img_np)
    for bbox in bboxes:
        startx, starty, endx, endy = bbox
        out_np[starty:endy, startx:endx] = img_np[starty:endy, startx:endx]
    out = Image.fromarray(out_np)
    return out

def show_only_bboxed_pixel2(img, bboxes):
    img_t = torchvision.transforms.ToTensor()(img)
    out_t = torch.zeros_like(img_t)
    for bbox in bboxes:
        startx, starty, endx, endy = bbox
        out_t[..., starty:endy, startx:endx] = img_t[..., starty:endy, startx:endx]
    out = torchvision.transforms.ToPILImage()(out_t)
    return out


def main():
    # 画像のパスを取得
    dataroot_A_path = Path('/home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train/A/day')
    dataroot_B_path = Path('/home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN/JPEGImages/train/B/day')
    bbox_dir_path = Path(f'{dataroot_A_path.as_posix()}_bbox')
    img_name_list = [Path(f).name for f in glob.glob((dataroot_A_path / '*').as_posix()) if os.path.isfile(f)]
    img_name_list.sort()
    target_dir_path = Path('/home/usrs/ugawa/lab/work/TICCGAN/show_bbox')

    # bboxの取得と表示
    i = 0   
    bbox_threshold = 16
    for img_name in img_name_list:
        if i == 20:
            break
        i += 1

        img_path = dataroot_B_path / img_name
        img = Image.open(img_path).convert('RGB')
        height = img.size[0]
        width = img.size[1]
        # bbox取得
        # 画像1枚につき複数のbboxがあるんで注意
        bbox_path = (bbox_dir_path / img_name).as_posix().replace('.jpg', '.npz')
        bboxes = gen_maskrcnn_bbox_fromPred(bbox_path)

        thresholds = [height // bbox_threshold, width // bbox_threshold]
        img_with_bbox = show_bbox(img, bboxes, thresholds=thresholds)
        #img_with_bbox = show_only_bboxed_pixel2(img, bboxes)
        img_with_bbox.save(target_dir_path / img_name)
        print(f"name: {img_name}, n_bboxes: {len(bboxes)}")
        
if __name__=='__main__':
    main()


