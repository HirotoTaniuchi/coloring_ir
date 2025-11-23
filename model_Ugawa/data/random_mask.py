import random
import numpy as np
from PIL import Image, ImageFilter

def mask_image(image, mask_ratio):
    # 画像のサイズを取得
    width, height = image.size

    # マスクするピクセル数を計算
    num_pixels = width * height
    num_masked_pixels = int(num_pixels * mask_ratio)

    # 画像をNumPy配列に変換
    np_image = np.array(image)

    # マスクするピクセルをランダムに選択
    indices = random.sample(range(num_pixels), num_masked_pixels)

    # 選択されたピクセルをマスク
    np_image_flat = np_image.reshape(-1, 3)
    np_image_flat[indices] = 0

    # マスクされた画像を作成
    masked_image = Image.fromarray(np_image)

    return masked_image

def split_image_to_patches(image, patch_size):
    # 画像のサイズを取得
    width, height = image.size

    # パッチの数を計算
    num_patches = (width // patch_size) * (height // patch_size)

    # パッチのリストを作成
    patches = []

    # 画像をパッチに分割
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches

def mask_random_patches(image, patch_size, mask_ratio):
    # 画像をパッチに分割
    patches = split_image_to_patches(image, patch_size)

    # マスクするパッチの数を計算
    num_masked_patches = int(len(patches) * mask_ratio)

    # ランダムにパッチをマスク
    masked_patches = []
    for patch in patches:
        if num_masked_patches > 0 and random.random() < mask_ratio:
            # パッチをマスク
            masked_patch = Image.new("RGB", (patch_size, patch_size), color=(255, 255, 255))
            masked_patches.append(masked_patch)
            num_masked_patches -= 1
        else:
            masked_patches.append(patch)

    # マスクされたパッチを結合して画像を作成
    masked_image = Image.new("RGB", image.size)
    width, height = image.size
    x, y = 0, 0
    for patch in masked_patches:
        masked_image.paste(patch, (x, y))
        x += patch_size
        if x >= width:
            x = 0
            y += patch_size

    return masked_image

# テスト用の画像パス
image_path = "FLIR_09019_fake_B.png"

# 画像を読み込む
image = Image.open(image_path)
filter = ImageFilter.GaussianBlur(5)
#image = image.filter(filter)

# マスクの設定
patch_size = 8
mask_ratio = 0.3

# マスク


# マスクして保存
patch_masked_image = mask_random_patches(image, patch_size, mask_ratio)
patch_masked_image = patch_masked_image.filter(filter)
patch_masked_image.save('patch_masked_image.png')
masked_image = mask_image(image, mask_ratio)
masked_image.save('masked_image.png')
