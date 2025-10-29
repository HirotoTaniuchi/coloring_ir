import os
from PIL import Image

# フォルダパスを指定
folder_A = "/home/usrs/taniuchi/workspace/projects/coloring_ir/CycleGAN-and-pix2pix/datasets/mfnet/images_rgb"
folder_B = "/home/usrs/taniuchi/workspace/projects/coloring_ir/CycleGAN-and-pix2pix/datasets/mfnet/images_ir_3ch"
folder_C = "/home/usrs/taniuchi/workspace/projects/coloring_ir/CycleGAN-and-pix2pix/datasets/mfnet/images_combined"

os.makedirs(folder_C, exist_ok=True)

for filename in os.listdir(folder_A):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        path_A = os.path.join(folder_A, filename)
        path_B = os.path.join(folder_B, filename)
        path_C = os.path.join(folder_C, filename)

        if not os.path.exists(path_B):
            print(f"{path_B} が見つかりません。スキップします。")
            continue

        img_A = Image.open(path_A)
        img_B = Image.open(path_B)

        # サイズを揃える（必要に応じて）
        if img_A.size != img_B.size:
            img_B = img_B.resize(img_A.size)

        # 横に連結
        new_img = Image.new('RGB', (img_A.width + img_B.width, img_A.height))
        new_img.paste(img_A, (0, 0))
        new_img.paste(img_B, (img_A.width, 0))

        new_img.save(path_C)
        print(f"{path_C} を保存しました。")