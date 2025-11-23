import xml.etree.ElementTree as ET
from os.path import join, isfile
import os
import numpy as np
import argparse


# FLIR_aligned用のセッティングがしてあります
# bboxをnpz形式で保存するものです
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='/home/usrs/ugawa/lab/work/datasets/FLIR_aligned_TICCGAN')
parser.add_argument("--stage", type=str, choices=["train_day", "train_night", "val_day", "val_night"])
args = parser.parse_args()
dir_path = args.dataset_dir # Annotationの入ってるディレクトリ
anno_path = join(dir_path, 'Annotations')
stage = args.stage

# 照合先ファイルリスト取得
if stage == "train_day":
    check_path = join(dir_path, "JPEGImages/train/A/day") # ファイル名を照合する対象のディレクトリ
    check_list = [f for f in os.listdir(check_path) if (isfile(join(check_path, f)) and '.jpg' in f)]
    savedir_path = check_path.replace("day", "day_bbox")
    os.makedirs(savedir_path, exist_ok=True)
elif stage == "train_night":
    check_path = join(dir_path, "JPEGImages/train/A/night")
    check_list = [f for f in os.listdir(check_path) if (isfile(join(check_path, f)) and '.jpg' in f)]
    savedir_path = check_path.replace("night", "night_bbox")
    os.makedirs(savedir_path, exist_ok=True)
elif stage == "val_day":
    check_path = join(dir_path, "JPEGImages/val/A/day")
    check_list = [f for f in os.listdir(check_path) if (isfile(join(check_path, f)) and '.jpg' in f)]
    savedir_path = check_path.replace("day", "day_bbox")
    os.makedirs(savedir_path, exist_ok=True)
elif stage == "val_night":
    check_path = join(dir_path, "JPEGImages/val/A/night")
    check_list = [f for f in os.listdir(check_path) if (isfile(join(check_path, f)) and '.jpg' in f)]
    savedir_path = check_path.replace("night", "night_bbox")
    os.makedirs(savedir_path, exist_ok=True)
else:
    raise NotImplementedError(f"{stage} is not implemented.")

xml_list = [f for f in os.listdir(anno_path) if (isfile(join(anno_path, f)) and '.xml' in f)]
print(check_list)
for filename in xml_list:

    if filename.replace("_PreviewData.xml", ".jpg") in check_list:
        print(f"{filename}")                
        file_path = join(anno_path, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        bbox_list = []
        for child in root:
            if child.tag == "object":
                for g_child in child:
                    if g_child.tag == "bndbox":
                        xmin = g_child.find('xmin').text
                        ymin = g_child.find('ymin').text
                        xmax = g_child.find('xmax').text
                        ymax = g_child.find('ymax').text
                        bbox = list(map(int, [xmin, ymin, xmax, ymax]))
                        bbox_list.append(bbox)
                        #print(f"        {bbox}")
        save_path = os.path.join(savedir_path, filename.replace('_PreviewData.xml', ''))
        np.savez(save_path, bbox = bbox_list)