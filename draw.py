
# Datasetから画像を取り出し、描画する

import matplotlib.pyplot as plt


# ## 訓練画像の描画
# 実行するたびに変わります
# 画像データの読み込み
index = 0
imges, anno_class_imges = train_dataset.__getitem__(index)

# 画像の表示
img_val = imges
img_val = img_val.numpy().transpose((1, 2, 0))
plt.imshow(img_val)
plt.show()

# # アノテーション画像の表示
# anno_file_path = train_anno_list[0]
# anno_class_img = Image.open(anno_file_path)   # [高さ][幅][色RGB]
# p_palette = anno_class_img.getpalette()

# anno_class_img_val = anno_class_imges.numpy()
# anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
# anno_class_img_val.putpalette(p_palette)
# plt.imshow(anno_class_img_val)
# plt.show()


# ## 検証画像の描画

# 画像データの読み込み
index = 0
imges, anno_class_imges = val_dataset.__getitem__(index)

# 画像の表示
img_val = imges
img_val = img_val.numpy().transpose((1, 2, 0))
plt.imshow(img_val)
plt.show()

# # アノテーション画像の表示
# anno_file_path = train_anno_list[0]
# anno_class_img = Image.open(anno_file_path)   # [高さ][幅][色RGB]
# p_palette = anno_class_img.getpalette()

# anno_class_img_val = anno_class_imges.numpy()
# anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
# anno_class_img_val.putpalette(p_palette)
# plt.imshow(anno_class_img_val)
# plt.show()


