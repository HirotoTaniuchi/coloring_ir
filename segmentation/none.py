import random
import math
import time
import tqdm
import datetime
import os
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from now import now1, now2

from seg_dataloader import make_datapath_list, DataTransform, MFNetDataset
from seg_loss import SegLoss
# from plus_train import train_model
from DeepLabV3Plus_Pytorch import network

# from plus_dataloader import make_datapath_list, DataTransform, MFNetDataset
# from seg_loss import SegLoss
# from plus_train import train_model
# from DeepLabV3Plus_Pytorch import network


# MODEL_NAME = "deeplabv3plus_resnet101"
# NUM_CLASSES = 19 #まずは19クラスで指定する必要あり
# OUTPUT_SRTIDE = 16
# PATH_TO_PTH = "/home/usrs/taniuchi/workspace/projects/coloring_ir/DeepLabV3Plus_Pytorch/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
# torch.serialization.safe_globals([np.core.multiarray.scalar])
# model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_SRTIDE)
# model.load_state_dict( torch.load( PATH_TO_PTH , weights_only = False)['model_state']  )
# # model.classifier.classifier = nn.Sequential(
# #     nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
# #     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
# #     nn.ReLU(inplace=True),
# #     nn.Conv2d(256, 9, kernel_size=(1, 1), stride=(1, 1))
# #     )

# モデルの設定
n_layers = 100 ###
MODEL_NAME = "deeplabv3plus_mobilenet"
# MODEL_NAME = "deeplabv3plus_mobilenet"
NUM_CLASSES = 19 #まずは19クラスで指定する必要あり ###
OUTPUT_SRTIDE = 16 ###
PATH_TO_PTH = "/home/usrs/taniuchi/workspace/projects/coloring_ir/DeepLabV3Plus_Pytorch/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
# PATH_TO_PTH = "/home/usrs/taniuchi/workspace/projects/coloring_ir/DeepLabV3Plus_Pytorch/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
torch.serialization.safe_globals([np.core.multiarray.scalar])
model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_SRTIDE)
model.load_state_dict( torch.load( PATH_TO_PTH , weights_only = False)['model_state']  )
for param in model.parameters(): param.requires_grad = True
# print(model)

def count_parameters(model):
    return sum(param.numel() for param in model.parameters())

# 表示
print(f"Total trainable parameters: {count_parameters(model)}")

# model.classifier.classifier = nn.Sequential(
#     nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#     nn.ReLU(inplace=True),
#     nn.Conv2d(256, 9, kernel_size=(1, 1), stride=(1, 1))
#     )

# for param in model.parameters():
#     param.requires_grad = False
# for param in model.classifier.parameters():
#     param.requires_grad = True