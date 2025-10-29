import os
import shutil

file_A = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/train_day.txt"   # 各行に画像名（拡張子付き）が書かれたテキストファイル
folder_B = "/home/usrs/taniuchi/workspace/projects/coloring_ir/CycleGAN-and-pix2pix/datasets/mfnet/images_combined"   # 画像が存在するか調べるフォルダ
folder_D = "/home/usrs/taniuchi/workspace/projects/coloring_ir/CycleGAN-and-pix2pix/datasets/mfnet/train"   # 画像をコピーする先のフォルダ

os.makedirs(folder_D, exist_ok=True)

with open(file_A, "r") as f:
    for line in f:
        basename = line.strip()
        filename = basename + ".png"
        src_path = os.path.join(folder_B, filename)
        dst_path = os.path.join(folder_D, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"{filename} を {folder_D} にコピーしました。")
        else:
            print(f"{filename} は {folder_B} に存在しません。")