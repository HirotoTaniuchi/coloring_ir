from PIL import Image
import numpy as np

img_path = "/home/usrs/taniuchi/workspace/projects/coloring_ir/CycleGAN-and-pix2pix/datasets/facades/test/1.jpg"  # 画像ファイルのパス
img = Image.open(img_path)
img_np = np.array(img)

print("データ型:", img_np.dtype)
print("サイズ:", img_np.shape)