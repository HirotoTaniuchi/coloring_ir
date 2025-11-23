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



if __name__ == '__main__':
    # チェックポイントとログのパスを先に宣言
    str_now = now1()
    MODEL_NAME = "deeplabv3plus_resnet101"
    DOMAIN = "ir" # 学習元データのドメイン（"ir" or "rgb" or "ir_3ch"）
    path_cpt = f'checkpoints/{DOMAIN}_{MODEL_NAME}_{str_now}' # 
    path_logs = f'logs/{DOMAIN}_{MODEL_NAME}_{str_now}'
    os.makedirs(path_cpt, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)

    # 設定をまとめる（パスも含める）
    config = {
        "rootpath": "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset",
        "DOMAIN": DOMAIN,
        "gpu_id": 0,
        "color_mean": (0.232, 0.267, 0.233),
        "color_std": (0.173, 0.173, 0.172),
        "batch_size": 4,
        "n_layers": 100,
        "MODEL_NAME": MODEL_NAME,
        "NUM_CLASSES": 19,
        "OUTPUT_SRTIDE": 16,
        "PATH_TO_PTH": "/home/usrs/taniuchi/workspace/projects/coloring_ir/DeepLabV3Plus_Pytorch/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth",
        "num_epochs": 100,
        "optimizer": {
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 0.0001
        },
        "path_cpt": path_cpt,
        "path_logs": path_logs
    }

    # YAMLとして保存
    import yaml
    with open(os.path.join(path_cpt, "config.yaml"), "w") as f:
        yaml.dump(config, f, allow_unicode=True)
    with open(os.path.join(path_logs, "config.yaml"), "w") as f:
        yaml.dump(config, f, allow_unicode=True)


    rootpath, imagepath = config["rootpath"], ("images_" + config["DOMAIN"])  ### 変更点
    print("imagepath:", imagepath)
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath, image_path=imagepath) ### 変更点
    gpu_id = config["gpu_id"]

    color_mean = config["color_mean"]
    color_std = config["color_std"]
    train_dataset = MFNetDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size=600, color_mean=color_mean, color_std=color_std))
    val_dataset = MFNetDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size=600, color_mean=color_mean, color_std=color_std))

    batch_size = config["batch_size"]
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    n_layers = config["n_layers"]
    NUM_CLASSES = config["NUM_CLASSES"]
    OUTPUT_SRTIDE = config["OUTPUT_SRTIDE"]
    PATH_TO_PTH = config["PATH_TO_PTH"]
    torch.serialization.safe_globals([np.core.multiarray.scalar])
    model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_SRTIDE)
    model.load_state_dict(torch.load(PATH_TO_PTH, weights_only=False)['model_state'])
    print("gpu_ids:", opt.gpu_ids, "device:", model.device)
    

    # print(model)
    if DOMAIN == "ir": # IR画像が1chなので、最初の畳み込み層を変更する
        model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.backbone.conv1.apply(weights_init)
    elif DOMAIN == "rgb_ir": # RGB+IR画像が4chなので、最初の畳み込み層を変更する
        model.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.backbone.conv1.apply(weights_init)
    model.classifier.classifier = nn.Sequential(
        nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 9, kernel_size=(1, 1), stride=(1, 1))
    ) ## NFNetデータセット用に19クラス→9クラスに変更


    for param in model.parameters(): param.requires_grad = True 
    # for param in model.backbone.layer3.parameters(): param.requires_grad = True ## もし容量不足で全層学習できない場合に使う

    criterion = SegLoss(aux_weight=0.4)
    num_epochs = config["num_epochs"]
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': config["optimizer"]["lr"]},
        {'params': model.classifier.parameters(), 'lr': config["optimizer"]["lr"]}
    ], momentum=config["optimizer"]["momentum"], weight_decay=config["optimizer"]["weight_decay"])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    train_model(
        model, dataloaders_dict, criterion, scheduler, optimizer,
        model_name=MODEL_NAME, num_epochs=num_epochs, n_layers=n_layers, gpu_id=gpu_id,
        path_cpt=path_cpt, path_logs=path_logs
    )
