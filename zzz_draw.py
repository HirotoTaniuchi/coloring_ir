
# Datasetから画像を取り出し、描画する

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.utils.data as data
import torch

from seg_dataloader import make_datapath_list, DataTransform, MFNetDataset


if __name__ == "__main__":
    # 動作確認 ファイルパスのリストを取得
    rootpath = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset"

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
        rootpath=rootpath)


    # (RGB)の色の平均値と標準偏差
    color_mean = (0.232, 0.267, 0.233)
    color_std = (0.173, 0.173, 0.172)

    # データセット作成
    train_dataset = MFNetDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size=500, color_mean=color_mean, color_std=color_std))

    val_dataset = MFNetDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size=500, color_mean=color_mean, color_std=color_std))

    # データの取り出し例
    # print("データの取り出し例")
    # print(val_dataset)
    # print(val_dataset.__getitem__(0))
    # print(val_dataset.__getitem__(0)[0])
    # print(val_dataset.__getitem__(0)[0].shape)



    # データローダーの作成
    batch_size = 8
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 動作の確認
    batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
    imges, anno_class_imges = next(batch_iterator)  # 1番目の要素を取り出す
    # print(imges.size())  # torch.Size([8, 3, 500, 500])
    # print(anno_class_imges.size())  # torch.Size([8, 3, 500, 500])
    










    # ## 訓練画像の描画
    # 実行するたびに変わります
    # 画像データの読み込み
    index = 0
    imges, anno_class_imges = train_dataset.__getitem__(index)

    # # 画像の表示
    # img_val = imges
    # img_val = img_val.numpy().transpose((1, 2, 0))
    # plt.imshow(img_val)
    # plt.show()

    # アノテーション画像の表示
    anno_file_path = train_anno_list[0]
    anno_class_img = Image.open(anno_file_path)   # [高さ][幅][色RGB]
    p_palette = anno_class_img.getpalette()

    anno_class_img_val = anno_class_imges.numpy()
    print("anno_class_img_val.shape", anno_class_img_val.shape)
    print("torch.squeeze(np.uint8(anno_class_img_val)).shape", np.squeeze(np.uint8(anno_class_img_val)).shape)
    anno_class_img_val = Image.fromarray(np.squeeze(np.uint8(anno_class_img_val)), mode="P")
    anno_class_img_val.putpalette(p_palette)
    plt.imshow(anno_class_img_val)
    plt.savefig("kokokokokoko.png")


    # # ## 検証画像の描画

    # # 画像データの読み込み
    # index = 0
    # imges, anno_class_imges = val_dataset.__getitem__(index)

    # # 画像の表示
    # img_val = imges
    # img_val = img_val.numpy().transpose((1, 2, 0))
    # plt.imshow(img_val)
    # plt.show()

    # # アノテーション画像の表示
    # anno_file_path = train_anno_list[0]
    # anno_class_img = Image.open(anno_file_path)   # [高さ][幅][色RGB]
    # p_palette = anno_class_img.getpalette()

    # anno_class_img_val = anno_class_imges.numpy()
    # anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
    # anno_class_img_val.putpalette(p_palette)
    # plt.imshow(anno_class_img_val)
    # plt.show()


