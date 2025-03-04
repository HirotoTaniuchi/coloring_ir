import random
import math
import time
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from seg4to1_dataloader import make_datapath_list, DataTransform, MFNetDataset
from seg4to1_loss import SegLoss
from seg4to1_train import train_model



# model.classifier[4] = nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
# 分類用の畳み込み層を、出力数21のものにつけかえる
# n_classes = 21
# net.decode_feature.classification = nn.Conv2d(
#     in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
# net.aux.classification = nn.Conv2d(
#     in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)



# 付け替えた畳み込み層を初期化する。活性化関数がシグモイド関数なので(?)Xavierを使用する。
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)

# スケジューラーの設定
def lambda_epoch(epoch):
    max_epoch = 100
    return math.pow((1-epoch/max_epoch), 0.9)



if __name__ == '__main__':
    # ファイルパスリスト作成
    rootpath = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset"
    train_img_list, train_anno_list, val_img_list, val_anno_list= make_datapath_list(
        rootpath=rootpath)

    # Dataset作成
    # (RGB)の色の平均値と標準偏差
    color_mean = (0.232, 0.267, 0.233)
    color_std = (0.173, 0.173, 0.172)

    # データセット作成
    train_dataset = MFNetDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size=700, color_mean=color_mean, color_std=color_std))
    val_dataset = MFNetDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size=700, color_mean=color_mean, color_std=color_std))

    # データローダーの作成
    batch_size = 4
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


    # 初期設定 # Setup seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # モデルの設定
    n_layers = 101
    model_name = f'deeplabv3_resnet{n_layers}'
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True) # deeplabv3のロード
    # モデルに内蔵されているモジュールを見やすく全列挙
    for name, module in model.named_modules():
        print(f'{name}: {module}')

    # 4チャネルのバッチを3チャネルに変換するモジュールを定義
    class FourToThreeChannel(nn.Module):
        def __init__(self):
            super(FourToThreeChannel, self).__init__()
            self.conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)
        
        def forward(self, x):
            return self.conv(x)
    
    # 変換モジュールをモデルの前に接続
    model.backbone0 = FourToThreeChannel()
    model.backbone0.apply(weights_init)


    # # 変換モジュールをモデルの前に接続
    # model = nn.Sequential(
    #     FourToThreeChannel(),
    #     model
    # )
    # model[0].apply(weights_init)


    
    criterion = SegLoss(aux_weight=0.4)  # 損失関数の設定
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
        {'params': model.aux_classifier.parameters(), 'lr': 1e-2}
    ], momentum=0.9, weight_decay=0.0001) # ファインチューニングなので、学習率は小さく
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
    num_epochs = 100 # 1にするとバグるかも


    train_model(model, dataloaders_dict, criterion, scheduler, optimizer, num_epochs=num_epochs, n_layers=n_layers)




    # # 3チャネルから4チャネルに変更
    # model.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.backbone.conv1.apply(weights_init)