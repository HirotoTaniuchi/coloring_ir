# "/home/usrs/taniuchi/workspace/projects/coloring_ir"下で実行
import torch
import torch.nn as nn

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, tqdm, numpy, collections, sys
from tensorboardX import SummaryWriter

from DeepLabV3Plus_Pytorch import network
from seg_dataloader import make_testdatapath_list, DataTransform, MFNetDataset
from now import now1




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
    img = Image.open(input_path)
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




def predict_savefig1(model, dataset, img_index, savedir= None, img_list= None, anno_list= None):
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
    img_original.save(savedir + f"/original/{os.path.splitext(os.path.basename(image_file_path))[0]}.png")


    # 2. 正解アノテーション画像の表示
    anno_file_path = anno_list[img_index]
    out_putpath = savedir + f"/target/{os.path.splitext(os.path.basename(image_file_path))[0]}.png"
    colorize_image(anno_file_path, out_putpath)



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
    anno_class_img_0to8.save(savedir + f"/predict/{os.path.splitext(os.path.basename(image_file_path))[0]}.png")


    # 4. PSPNetの出力から最大クラスを求め、カラーパレット形式にし、画像サイズを元に戻す
    outputs = model(x)
    yrgb = outputs  # AuxLoss側は無視
    yrgb = yrgb[0].detach().numpy()  # y：torch.Size([1, 21, 475, 475])
    yrgb = np.argmax(yrgb, axis=0)
    anno_class_img = Image.fromarray(np.uint8(yrgb), mode="P") # *30はいらない!デバッグ2025/05/10
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette) 
    anno_class_img.save(savedir + f"/predictrgb/{os.path.splitext(os.path.basename(image_file_path))[0]}.png")


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
    result.save(savedir + f"/trans/{os.path.splitext(os.path.basename(image_file_path))[0]}.png")




    return img_original, anno_class_img, result





if __name__ == '__main__':  
    # ファイルパスリスト作成
    imagepath = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/images_rgb"
    (test_img_list, test_anno_list)= make_testdatapath_list(imagepath=imagepath)

    # (RGB)の色の平均値と標準偏差
    color_mean = (0.232, 0.267, 0.233)
    color_std = (0.173, 0.173, 0.172)

    # データセット作成
    val_dataset = MFNetDataset(test_img_list, test_anno_list, phase="val", transform=DataTransform(
        input_size=500, color_mean=color_mean, color_std=color_std))
    
    # 訓練後のモデルをロード 
    MODEL_NAME = "deeplabv3plus_mobilenet"
    NUM_CLASSES = 19
    OUTPUT_SRTIDE = 16
    PATH_TO_PTH = "/home/usrs/taniuchi/workspace/projects/coloring_ir/checkpoints/deeplabv3plus_mobilenet_202505151025/seg_100_100_202505151025.pth"
    model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_SRTIDE)
    model.classifier.classifier = nn.Sequential(
        nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 9, kernel_size=(1, 1), stride=(1, 1))
        )
    model.load_state_dict( torch.load( PATH_TO_PTH , weights_only = False)  ) # 公式訓練済を使う時のみ['model_state']
    print('ネットワーク設定完了：学習済みの重みをロードしました')



    # ディレクトリ作成
    save_dir = f"./output_seg/{MODEL_NAME}_test_{now1()}"
    if not os.path.exists(save_dir):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(save_dir)
        os.makedirs(save_dir + "/original")
        os.makedirs(save_dir + "/predict")
        os.makedirs(save_dir + "/predictrgb")
        os.makedirs(save_dir + "/target")
        os.makedirs(save_dir + "/trans")

    # readme.mdがない場合は作成し、設定情報を書き込む
    readme_path = os.path.join(save_dir, "readme.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("# Segmentation Test 設定情報\n")
            f.write(f"- imagepath: {imagepath}\n")
            f.write(f"- MODEL_NAME: {MODEL_NAME}\n")
            f.write(f"- NUM_CLASSES: {NUM_CLASSES}\n")
            f.write(f"- OUTPUT_STRIDE: {OUTPUT_SRTIDE}\n")
            f.write(f"- PATH_TO_PTH: {PATH_TO_PTH}\n")

    # テストの実行
    for i in tqdm.tqdm(range(len(val_dataset))):
        org, anno, result = predict_savefig1(model, val_dataset, img_index=i, savedir = save_dir, img_list=test_img_list, anno_list=test_anno_list)


    # # FIDの計算
    # inception_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling='avg')

    # def compute_embeddings(dataloader, count):
    #     image_embeddings = []
    #     for _ in tqdm(range(count)):
    #         images = next(iter(dataloader))
    #         embeddings = inception_model.predict(images)
    #         image_embeddings.extend(embeddings)
    #     return np.array(image_embeddings)
    
    # count = math.ceil(10000/BATCH_SIZE)
    # # compute embeddings for real images
    # real_image_embeddings = compute_embeddings(trainloader, count)
    # # compute embeddings for generated images
    # generated_image_embeddings = compute_embeddings(genloader, count)
    # real_image_embeddings.shape, generated_image_embeddings.shape


    # tensorboardへの出力
    # writer = SummaryWriter(log_dir=f"./logs/seg_test_{now1()}")
    # writer.add_figure('img_original', plt.imshow(np.asarray(org)))
    # writer.add_figure('anno_class_img', plt.imshow(np.asarray(anno)))
    # writer.add_figure('result', plt.imshow(np.asarray(result)))
    # writer.close()





    