# 谷内により作成
# とりあえず、カラー写真、白黒写真をそれぞれセグメンテーションしてみたい
# https://github.com/VainF/DeepLabV3Plus-Pytorch
# /home/usrs/taniuchi/sotsuken/TICCGAN/scripts/exp1/test_FLIR_day.sh

import torch

import network_segmentation as network


MODEL_NAME = "deeplabv3plus_resnet101 "
NUM_CLASSES = 19
OUTPUT_STRIDE = 16


model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_STRIDE)
model.load_state_dict( torch.load( PATH_TO_PTH )['model_state']  )
