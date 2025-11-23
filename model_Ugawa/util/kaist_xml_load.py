import xml.etree.ElementTree as ET
from os.path import join, isfile, isdir
import os
import numpy as np
import argparse

# FLIR_aligned用のセッティングがしてあります

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15/annotations-xml-new-sanitized')
parser.add_argument("--stage", type=str, choices=["train_day", "train_night", "test_day", "test_night"])
args = parser.parse_args()

stage = args.stage # ファイルを整理するか
dir_path = args.dataset_dir # rootフォルダ
dir_path2 = "/home/usrs/ugawa/lab/work/datasets/kaist-cvpr15/copy"

if stage == 'train_day':
    set_list = ["set00", "set01", "set02"]
    savedir_path = join(dir_path2, 'train_day/trainA_bbox')
    os.makedirs(savedir_path, exist_ok=True)
elif stage == 'train_night':
    set_list = ["set03", "set04", "set05"]
    savedir_path = join(dir_path2, 'train_night/trainA_bbox')
    os.makedirs(savedir_path, exist_ok=True)
elif stage == 'test_day':
    set_list = ["set06", "set07", "set08"]
    savedir_path = join(dir_path2, 'test_day/testA_bbox')
    os.makedirs(savedir_path, exist_ok=True)
elif stage == 'test_night':
    set_list = ["set09", "set10", "set11"]
    savedir_path = join(dir_path2, 'test_night/testA_bbox')
    os.makedirs(savedir_path, exist_ok=True)
else:
    raise NotImplementedError(f"{stage} is not implemented.")


for set_name in set_list:
    set_path = join(dir_path, set_name)
    V_list = [f for f in os.listdir(set_path) if isdir(join(set_path, f))]
    for V in V_list:
        V_path = join(set_path, V)
        xml_list = [f for f in os.listdir(V_path) if (isfile(join(V_path, f)) and '.xml' in f)]
        for file_name in xml_list:
            print(f"{set_name}_{V}_{file_name}")
            file_path = join(V_path, file_name)
            tree = ET.parse(file_path)
            root = tree.getroot()
            bbox_list = []
            for child in root:
                if child.tag == 'object':
                    for g_child in child:
                        if g_child.tag == "bndbox":
                            x = int(g_child.find('x').text)
                            y = int(g_child.find('y').text)
                            w = int(g_child.find('w').text)
                            h = int(g_child.find('h').text)
                            bbox = [x, y, x+w, y+h]
                            bbox_list.append(bbox)
                            print(f"        {bbox}")
            savefile_path = join(savedir_path, f"{set_name}_{V}_{file_name.replace('.xml', '')}")
            np.savez(savefile_path, bbox = bbox_list)



