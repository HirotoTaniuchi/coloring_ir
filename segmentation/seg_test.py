# "/home/usrs/taniuchi/workspace/projects/coloring_ir"下で実行
import torch
import torch.nn as nn

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, tqdm, numpy, collections, sys
from tensorboardX import SummaryWriter

from model_DeepLabV3Plus_Pytorch import network
from seg_dataloader import make_testdatapath_list, DataTransform, MFNetDataset
from now import now1
from eval_seg import eval_seg
import argparse




def colorize_image(input_path, output_path):
    # Define the color map for values 0 to 8
    color_map = {
        0: (0, 0, 0),       # Unlabelled?
        1: (57, 5, 126),    # Car?
        2: (64, 64, 16),    # Person?
        3: (67, 126, 186),  # Bike?
        4: (32, 9, 181),    # Curve?
        5: (128, 127, 39),  # Car Stop?
        6: (63, 64, 124),   # Guardrail
        7: (183, 132, 128), # Color cone?
        8: (177, 75, 30)    # Bump?
    }

    # Load the input image
    img = Image.open(input_path)  # 修正: fopen -> open
    img = np.array(img)

    # Create an empty RGB image
    colorized_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Apply the color map
    for value, color in color_map.items():
        colorized_img[img == value] = color

    # Save the colorized image
    colorized_img = Image.fromarray(colorized_img)
    colorized_img.save(output_path)
    # print("saved!")




def predict_savefig1(model, dataset, img_index, savedir= None, img_list= None, anno_list= None, savedir2=None):
    """
    """

    p_palette = [0, 0, 0,
                57, 5, 126,
                64, 64, 16,
                67, 126, 186,
                32, 9, 181,
                128, 127, 39,
                63, 64, 124,
                183, 132, 128,
                177, 75, 30]


    # 1. 元画像の表示
    image_file_path = img_list[img_index]
    img_original = Image.open(image_file_path)   # [高さ][幅][色RGB]
    img_width, img_height = img_original.size
    original_path = savedir + f"/original/{os.path.splitext(os.path.basename(image_file_path))[0]}.png"
    img_original.save(original_path)
    if savedir2 and savedir2 != savedir:
        img_original.save(savedir2 + f"/original/{os.path.splitext(os.path.basename(image_file_path))[0]}.png")

    # 2. 正解アノテーション画像の表示
    anno_file_path = anno_list[img_index]
    out_putpath = savedir + f"/target/{os.path.splitext(os.path.basename(image_file_path))[0]}.png"
    colorize_image(anno_file_path, out_putpath)
    if savedir2 and savedir2 != savedir:
        out_putpath2 = savedir2 + f"/target/{os.path.splitext(os.path.basename(image_file_path))[0]}.png"
        colorize_image(anno_file_path, out_putpath2)

    # 3. モデル出力(0〜8クラス)
    model.eval()
    img, anno_class_img = dataset.__getitem__(img_index)
    x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 475, 475])
    outputs = model(x)
    y = outputs.detach()  # AuxLoss側は無視
    y = y[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
    y = np.argmax(y, axis=0)
    y = y.astype(np.uint8)  # 0〜8クラスに変換
    # print(img_index, "y.shape", y.shape, collections.Counter(y.flatten()))
    anno_class_img_0to8 = Image.fromarray(np.uint8(y)) #mode =Pがいらなかった！
    anno_class_img_0to8 = anno_class_img_0to8.resize((img_width, img_height), Image.NEAREST)
    pred_idx_path = savedir + f"/predict/{os.path.splitext(os.path.basename(image_file_path))[0]}.png"
    anno_class_img_0to8.save(pred_idx_path)
    if savedir2 and savedir2 != savedir:
        anno_class_img_0to8.save(savedir2 + f"/predict/{os.path.splitext(os.path.basename(image_file_path))[0]}.png")

    # 4. PSPNetの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
    outputs = model(x)
    yrgb = outputs  # AuxLoss側は無視
    yrgb = yrgb[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
    yrgb = np.argmax(yrgb, axis=0)
    anno_class_img = Image.fromarray(np.uint8(yrgb), mode="P") # *30はいらない!デバッグ2025/05/10
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette) 
    pred_rgb_path = savedir + f"/predictrgb/{os.path.splitext(os.path.basename(image_file_path))[0]}.png"
    anno_class_img.save(pred_rgb_path)
    if savedir2 and savedir2 != savedir:
        anno_class_img.save(savedir2 + f"/predictrgb/{os.path.splitext(os.path.basename(image_file_path))[0]}.png")

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
    trans_path = savedir + f"/trans/{os.path.splitext(os.path.basename(image_file_path))[0]}.png"
    result.save(trans_path)
    if savedir2 and savedir2 != savedir:
        result.save(savedir2 + f"/trans/{os.path.splitext(os.path.basename(image_file_path))[0]}.png")

    return img_original, anno_class_img, result





def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--imagepath', type=str, required=True)
    p.add_argument('--model_name', type=str, default='deeplabv3plus_resnet101')
    p.add_argument('--num_classes', type=int, default=19)
    p.add_argument('--output_stride', type=int, default=16)
    p.add_argument('--domain', type=str, choices=['ir', 'rgb', 'rgb_ir', 'ir_3ch'], default='rgb')
    p.add_argument('--pth_path', type=str, required=True)
    p.add_argument('--input_size', type=int, default=500)
    p.add_argument('--color_mean', type=str, default='0.232,0.267,0.233')
    p.add_argument('--color_std', type=str, default='0.173,0.173,0.172')
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--save_dir_2', type=str, default=None)  # 追加: セカンダリ保存先
    p.add_argument('--file_list', type=str, default=None, help='評価用 list ファイル')
    p.add_argument('--target_dir', type=str, default=None, help='GT ラベルディレクトリ')
    p.add_argument('--no_eval', action='store_true')
    return p.parse_args()

def _str_to_tuple_f(s):
    return tuple(float(x) for x in s.split(','))





if __name__ == '__main__':
    args = parse_args()
    # ファイルパスリスト作成
    imagepath = args.imagepath
    (test_img_list, test_anno_list)= make_testdatapath_list(imagepath=imagepath)

    color_mean = _str_to_tuple_f(args.color_mean)
    color_std = _str_to_tuple_f(args.color_std)

    INPUT_SIZE = args.input_size
    val_dataset = MFNetDataset(test_img_list, test_anno_list, phase="val",
                               transform=DataTransform(input_size=INPUT_SIZE,
                                                       color_mean=color_mean, color_std=color_std))

    MODEL_NAME = args.model_name
    NUM_CLASSES = args.num_classes
    OUTPUT_SRTIDE = args.output_stride
    DOMAIN = args.domain
    PATH_TO_PTH = args.pth_path

    model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_SRTIDE)
    if DOMAIN == "ir":
        model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif DOMAIN == "rgb_ir":
        model.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if NUM_CLASSES != 19:
        model.classifier.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, NUM_CLASSES, kernel_size=1, stride=1)
        )
    state = torch.load(PATH_TO_PTH, weights_only=False)
    if isinstance(state, dict) and 'model_state' in state and any(k.startswith('backbone') for k in state['model_state'].keys()):
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print('ネットワーク設定完了：学習済みの重みをロードしました')

    # 保存先の作成
    save_dir = args.save_dir or f"./output_seg/{MODEL_NAME}_{now1()}"
    os.makedirs(save_dir, exist_ok=True)
    for d in ["original", "predict", "predictrgb", "target", "trans"]:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)

    # 追加の保存先(save_dir_2)が指定された場合は同様に作成
    save_dir_2 = args.save_dir_2
    if save_dir_2 and save_dir_2 != save_dir:
        os.makedirs(save_dir_2, exist_ok=True)
        for d in ["original", "predict", "predictrgb", "target", "trans"]:
            os.makedirs(os.path.join(save_dir_2, d), exist_ok=True)

    # readme.md を両方に作成
    readme_body = (
        "# Segmentation Test 設定情報\n"
        f"- imagepath: {imagepath}\n"
        f"- MODEL_NAME: {MODEL_NAME}\n"
        f"- NUM_CLASSES: {NUM_CLASSES}\n"
        f"- OUTPUT_STRIDE: {OUTPUT_SRTIDE}\n"
        f"- PATH_TO_PTH: {PATH_TO_PTH}\n"
    )
    readme_path = os.path.join(save_dir, "readme.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(readme_body)
    if save_dir_2 and save_dir_2 != save_dir:
        readme_path2 = os.path.join(save_dir_2, "readme.md")
        if not os.path.exists(readme_path2):
            with open(readme_path2, "w") as f:
                f.write(readme_body)

    for i in tqdm.tqdm(range(len(val_dataset))):
        predict_savefig1(model, val_dataset, img_index=i,
                         savedir=save_dir, img_list=test_img_list, anno_list=test_anno_list, savedir2=save_dir_2)
    print("Segmentation Test 完了！")

    # 評価はメイン保存先のみで実施
    if (not args.no_eval) and args.file_list and args.target_dir:
        eval_seg(pred_dir=save_dir + "/predict",
                 target_dir=args.target_dir,
                 file_list=args.file_list)
        # save_dir_2 が存在し、かつ save_dir と異なる場合のみ評価
        if save_dir_2 and (save_dir_2 != save_dir):
            eval_seg(pred_dir=save_dir_2 + "/predict",
                     target_dir=args.target_dir,
                     file_list=args.file_list)
    else:
        print("評価をスキップしました (--no_eval か file_list/target_dir 未指定)")





