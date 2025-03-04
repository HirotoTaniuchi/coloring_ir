# モデルを学習させる関数
import torch
import time
import pandas as pd
from torch import nn, optim
import tqdm
import datetime
import os


from tensorboardX import SummaryWriter
from now import now1, now2


def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs, n_layers=50):
    """
    current directoryにweightsフォルダがあることを前提としている
    """

    # print("net", net)

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # checkpoint用のディレクトリを作成
    str_now = now1()
    path_cpt = f'checkpoints/seg_{n_layers}_{str_now}'
    if not os.path.exists(path_cpt):
        os.makedirs(path_cpt, exist_ok=True)

    # multiple minibatch
    batch_multiplier = 3

    # epochのループ
    str_now = now1()
    writer = SummaryWriter(log_dir=f"./logs/tbloss_seg_{n_layers}_{str_now}")
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                scheduler.step()  # 最適化schedulerの更新
                optimizer.zero_grad()
                print('（train）')

            else:
                if((epoch+1) % 5 == 0):
                    net.eval()   # モデルを検証モードに
                    print('-------------')
                    print('（val）')
                else:
                    # 検証は5回に1回だけ行う
                    continue

            # データローダーからminibatchずつ取り出すループ
            count = 0  # multiple minibatch
            for imges, anno_class_imges in dataloaders_dict[phase]:
                # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
                # issue #186より不要なのでコメントアウト
                # if imges.size()[0] == 1:
                #     continue

                # print("imges.shape", imges.shape)
                # print("anno_class_imges.shape", anno_class_imges.shape)
                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)
                anno_class_imges = anno_class_imges.to(device)

                
                # multiple minibatchでのパラメータの更新
                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_out = net(imges)['out']
                    outputs_aux = net(imges)['aux']
                    loss = criterion(outputs_aux, anno_class_imges.long()) / batch_multiplier

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        optimizer.zero_grad()  # 勾配の初期化 #仮に0304:1408
                        loss.backward()  # 勾配の計算
                        count -= 1  # multiple minibatch

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item()/batch_size*batch_multiplier, duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f} ||{}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs, now1()))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss /
                     num_train_imgs, 'val_loss': epoch_val_loss/num_val_imgs}
        
        logs.append(log_epoch)
        writer.add_scalar('train_loss', logs[epoch]['train_loss'], epoch+1)
        writer.add_scalar('val_loss', logs[epoch]['val_loss'], epoch+1)

        df = pd.DataFrame(logs)
        df.to_csv(f"logs/csvloss_{n_layers}_{str_now}.csv")
        df.to_csv(path_cpt + f"/csvloss_{n_layers}_{str_now}.csv")

        if((epoch+1) % 10 == 0) or (epoch+1 ==1):
            # ネットワークを保存する
            torch.save(net.state_dict(), path_cpt + f'/seg_{n_layers}_{str(epoch+1)}_{str_now}.pth')


    # 最後のネットワークを保存する
    torch.save(net.state_dict(), path_cpt + f'/seg_{n_layers}_{str(epoch+1)}_{str_now}.pth')
    
    writer.close()
