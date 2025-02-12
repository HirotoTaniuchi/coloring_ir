import cv2
import numpy as np

def check_image_channels(image_path):
    # 画像を読み込む
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 画像が4チャネルかどうか確認
    if image is None:
        print("画像が読み込めませんでした。")
        return False
    if image.shape[2] != 4:
        print("画像は4チャネルではありません。")
        return False
    
    # 前の3チャネルがRGB画像であることを確認
    rgb_image = image[:, :, :3]
    if not np.all((rgb_image >= 0) & (rgb_image <= 255)):
        print("前の3チャネルがRGB画像ではありません。")
        return False
    
    # 最後の1チャネルがその他のチャネルであることを確認
    other_channel = image[:, :, 3]
    if not np.all((other_channel >= 0) & (other_channel <= 255)):
        print("最後の1チャネルがその他のチャネルではありません。")
        return False
    
    # 各チャネルの画素値の平均値、最大値、最小値を計算して出力
    for i, channel_name in enumerate(['R', 'G', 'B', 'Other']):
        channel = image[:, :, i]
        mean_val = np.mean(channel)
        max_val = np.max(channel)
        min_val = np.min(channel)
        print(f"{channel_name}チャネル - 平均値: {mean_val}, 最大値: {max_val}, 最小値: {min_val}")
    
    print("画像は前の3チャネルがRGB画像で、最後の1チャネルがその他のチャネルです。")
    return True

# テスト用の画像パス
image_path = '/home/usrs/taniuchi/workspace/zatta/20241219_中間発表用_MFNet/00292D.png'
check_image_channels(image_path)