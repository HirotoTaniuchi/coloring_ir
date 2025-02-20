from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch

from dataloader import make_datapath_list, DataTransform, MFNetDataset
from loss import ICRLoss
from train import train_model

def predict_savefig(model, dataset, img_index):
    """
    """

    # 1. 元画像の表示
    image_file_path = val_img_list[img_index]
    img_original = Image.open(image_file_path)   # [高さ][幅][色RGB]
    img_width, img_height = img_original.size
    plt.imshow(img_original)
    plt.savefig(f"original_{img_index}.png")

    # 2. 正解アノテーション画像の表示
    anno_file_path = val_anno_list[img_index]
    anno_class_img = Image.open(anno_file_path)   # [高さ][幅][色RGB]
    p_palette = anno_class_img.getpalette()
    plt.imshow(anno_class_img)
    plt.savefig(f"target_{img_index}.png")

    # 3. PSPNetで推論する
    model.eval()
    img, anno_class_img = val_dataset.__getitem__(img_index)
    x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])
    outputs = model(x)
    y = outputs[0]  # AuxLoss側は無視

    # 4. PSPNetの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
    y = y[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
    y = np.argmax(y, axis=0)
    anno_class_img = Image.fromarray(np.uint8(y), mode="P")
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette)
    plt.imshow(anno_class_img)
    plt.savefig(f"predict_{img_index}.png")

    # 5. 画像を透過させて重ねる
    trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
    anno_class_img = anno_class_img.convert('RGBA')  # カラーパレット形式をRGBAに変換

    for x in range(img_width):
        for y in range(img_height):
            # 推論結果画像のピクセルデータを取得
            pixel = anno_class_img.getpixel((x, y))
            r, g, b, a = pixel

            # (0, 0, 0)の背景ならそのままにして透過させる
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                continue
            else:
                # それ以外の色は用意した画像にピクセルを書き込む
                trans_img.putpixel((x, y), (r, g, b, 200))
                # 200は透過度の大きさを指定している

    result = Image.alpha_composite(img_original.convert('RGBA'), trans_img)
    plt.imshow(result)
    plt.savefig(f"trans_{img_index}.png")





if __name__ == '__main__':  
    # ファイルパスリスト作成
    rootpath = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset"
    train_img_list, train_anno_list, val_img_list, val_anno_list= make_datapath_list(
        rootpath=rootpath)

    # (RGB)の色の平均値と標準偏差
    color_mean = (0.232, 0.267, 0.233)
    color_std = (0.173, 0.173, 0.172)


    # データセット作成
    val_dataset = MFNetDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size=500, color_mean=color_mean, color_std=color_std))
    
    # 訓練後のモデルをロード
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
    # state_dict = torch.load("./weights/pspnet50_30.pth", map_location={'cuda:0': 'cpu'}) ## ここを変更
    model.load_state_dict(state_dict)
    print('ネットワーク設定完了：学習済みの重みをロードしました')


    predict_savefig(model, val_dataset, img_index=4)







    