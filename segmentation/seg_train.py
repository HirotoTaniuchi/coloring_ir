import argparse
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import random, math, time, tqdm, datetime, os
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import yaml

from now import now1, now2
from seg_dataloader import make_datapath_list, DataTransform, MFNetDataset
from seg_loss import SegLoss
from DeepLabV3Plus_Pytorch import network

def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, model_name, num_epochs, n_layers=50, gpu_id=0, path_cpt=None, path_logs=None):
    """
    current directoryにweightsフォルダがあることを前提としている
    """

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    net.to(device)
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    iteration = 1
    logs = []

    # checkpoint/logsディレクトリのパスを引数から受け取る
    if path_cpt is None:
        str_now = now1()
        path_cpt = f'checkpoints/{model_name}_{str_now}_'
    if not os.path.exists(path_cpt):
        os.makedirs(path_cpt, exist_ok=True)

    if path_logs is None:
        str_now = now1()
        path_logs = f'logs/{model_name}_{str_now}'
    if not os.path.exists(path_logs):
        os.makedirs(path_logs, exist_ok=True)

    batch_multiplier = 3

    str_now = now1()
    writer = SummaryWriter(log_dir=path_logs)
    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            count = 0
            for imges, anno_class_imges in dataloaders_dict[phase]:
                imges = imges.to(device)
                anno_class_imges = anno_class_imges.to(device)

                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_out = net(imges)
                    loss = criterion(outputs_out, anno_class_imges.long()) / batch_multiplier

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        count -= 1

                        if (iteration % 10 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('|| iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item()/batch_size*batch_multiplier, duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

            if phase == 'train':
                net.train()
                scheduler.step()
                optimizer.zero_grad()
                print('（train）')
            else:
                if((epoch+1) % 5 == 0):
                    net.eval()
                    print('-------------')
                    print('（val）')
                else:
                    continue

        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f} ||{}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs, now1()))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss / num_train_imgs, 'val_loss': epoch_val_loss/num_val_imgs}
        logs.append(log_epoch)
        writer.add_scalar('train_loss', logs[epoch]['train_loss'], epoch+1)
        writer.add_scalar('val_loss', logs[epoch]['val_loss'], epoch+1)

        df = pd.DataFrame(logs)
        df.to_csv(f"{path_logs}/csvloss_{n_layers}_{str_now}.csv")
        df.to_csv(f"{path_cpt}/csvloss_{n_layers}_{str_now}.csv")

        if((epoch+1) % 10 == 0) or (epoch+1 ==1):
            torch.save(net.state_dict(), f"{path_cpt}/seg_{n_layers}_{str(epoch+1)}_{str_now}.pth")

    torch.save(net.state_dict(), f"{path_cpt}/seg_{n_layers}_{str(epoch+1)}_{str_now}.pth")
    writer.close()


# 付け替えた畳み込み層を初期化する。活性化関数がシグモイド関数なので(?)Xavierを使用する。
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)

# スケジューラーの設定
def lambda_epoch(epoch):
    max_epoch = 1000
    return math.pow((1-epoch/max_epoch), 0.9)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rootpath', type=str, required=True)
    p.add_argument('--domain', type=str, choices=['ir', 'rgb', 'rgb_ir', 'ir_3ch'], required=True)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--color_mean', type=str, default='0.232,0.267,0.233')
    p.add_argument('--color_std', type=str, default='0.173,0.173,0.172')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=100)
    p.add_argument('--model_name', type=str, default='deeplabv3plus_resnet101')
    p.add_argument('--num_classes', type=int, default=19)
    p.add_argument('--output_stride', type=int, default=16)
    p.add_argument('--pretrained_path', type=str, required=True,
                   help='公式学習済み or 事前学習済モデルへのパス (.pth)')
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--path_cpt', type=str, default=None)
    p.add_argument('--path_logs', type=str, default=None)
    p.add_argument('--input_size', type=int, default=600)
    p.add_argument('--aux_weight', type=float, default=0.4)
    p.add_argument('--save_yaml', action='store_true')
    return p.parse_args()

def _str_to_tuple_f(s):
    return tuple(float(x) for x in s.split(','))

if __name__ == '__main__':
    args = parse_args()
    str_now = now1()

    MODEL_NAME = args.model_name
    DOMAIN = args.domain

    path_cpt = args.path_cpt or f'checkpoints/{DOMAIN}_{MODEL_NAME}_{str_now}'
    path_logs = args.path_logs or f'logs/{DOMAIN}_{MODEL_NAME}_{str_now}'
    os.makedirs(path_cpt, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)

    color_mean = _str_to_tuple_f(args.color_mean)
    color_std = _str_to_tuple_f(args.color_std)

    # データパス
    rootpath = args.rootpath
    imagepath = ("images_" + DOMAIN)  # 従来ロジック維持
    print("imagepath:", imagepath)
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath, image_path=imagepath)

    train_dataset = MFNetDataset(train_img_list, train_anno_list, phase="train",
                                 transform=DataTransform(input_size=args.input_size,
                                                         color_mean=color_mean, color_std=color_std))
    val_dataset = MFNetDataset(val_img_list, val_anno_list, phase="val",
                               transform=DataTransform(input_size=args.input_size,
                                                       color_mean=color_mean, color_std=color_std))
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    torch.manual_seed(1234); np.random.seed(1234); random.seed(1234)

    torch.serialization.safe_globals([np.core.multiarray.scalar])
    model = network.modeling.__dict__[MODEL_NAME](num_classes=args.num_classes, output_stride=args.output_stride)
    # 公式学習済み形式か自前かの差異に合わせ try
    state = torch.load(args.pretrained_path, weights_only=False)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)

    # 最初の conv 差し替え
    if DOMAIN == "ir":
        model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.backbone.conv1.apply(weights_init)
    elif DOMAIN == "rgb_ir":
        model.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.backbone.conv1.apply(weights_init)

    # クラス数変更が必要ならここで (19→9 等) ：引数 num_classes に従う
    if args.num_classes != 19:
        model.classifier.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, args.num_classes, kernel_size=1, stride=1)
        )

    for param in model.parameters():
        param.requires_grad = True

    criterion = SegLoss(aux_weight=args.aux_weight)
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ], momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    if args.save_yaml:
        config = {
            "rootpath": rootpath,
            "DOMAIN": DOMAIN,
            "gpu_id": args.gpu_id,
            "color_mean": color_mean,
            "color_std": color_std,
            "batch_size": args.batch_size,
            "n_layers": args.n_layers,
            "MODEL_NAME": MODEL_NAME,
            "NUM_CLASSES": args.num_classes,
            "OUTPUT_STRIDE": args.output_stride,
            "PATH_TO_PTH": args.pretrained_path,
            "num_epochs": args.num_epochs,
            "optimizer": {
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay
            },
            "input_size": args.input_size,
            "aux_weight": args.aux_weight,
            "path_cpt": path_cpt,
            "path_logs": path_logs
        }
        with open(os.path.join(path_cpt, "config.yaml"), "w") as f:
            yaml.dump(config, f, allow_unicode=True)
        with open(os.path.join(path_logs, "config.yaml"), "w") as f:
            yaml.dump(config, f, allow_unicode=True)

    train_model(
        model, dataloaders_dict, criterion, scheduler, optimizer,
        model_name=MODEL_NAME, num_epochs=args.num_epochs,
        n_layers=args.n_layers, gpu_id=args.gpu_id,
        path_cpt=path_cpt, path_logs=path_logs
    )
