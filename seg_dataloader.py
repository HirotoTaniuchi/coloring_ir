# パッケージのimport
import os.path as osp
from PIL import Image
import torch.utils as utils
import torch.utils.data as data
import torch, torchvision
import numpy as np
import pdb


# # ファイルパスリストを作成する
def make_datapath_list(rootpath):
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath,'images_rgb', '%s.png')
    annopath_template = osp.join(rootpath, 'labels', '%s.png')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath, 'train_day.txt')
    print("train_id_names", train_id_names)
    val_id_names = osp.join(rootpath, 'val.txt')
    print("val_id_names", val_id_names)

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


def make_testdatapath_list(imagepath):

    # # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    # if rootpath == "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset":
    #     imgpath_template = osp.join(rootpath, 'images_rgb', '%s.png')
    # elif osp.isdir (osp.join(rootpath, 'fake_B')):
    #     imgpath_template = osp.join(rootpath, 'fake_B', '%s.png') # ugawatestの場合
    # else:
    #     print("Error: rootpath is not correct", osp.join(rootpath, 'fake_B'))
    #     return None
    imgpath_template = osp.join(imagepath, '%s.png')
    annopath_template = osp.join("/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset", 'labels', '%s.png')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    test_id_names = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/test_day.txt"

    # テストデータの画像ファイルとアノテーションファイルへのパスリストを作成
    test_img_list = list()
    test_anno_list = list()

    for line in open(test_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        test_img_list.append(img_path)
        test_anno_list.append(anno_path)
    
    return test_img_list, test_anno_list


if __name__ == "__main__":
    # 動作確認 ファイルパスのリストを取得
    imagepath = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/images_rgb"

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
        imagepath=imagepath)


# # Datasetの作成
# データ処理のクラスとデータオーギュメンテーションのクラスをimportする
from torchvision.transforms import ToTensor, Compose, RandomRotation, Resize, Normalize
from torchvision.transforms.functional import to_tensor


class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std):
        # print("color_mean", color_mean)
        # print("color_std", color_std)
        self.data_transform = {
            'train': Compose([
                # ToTensor(),  # テンソルに変換 ############ToTensorを使わない方針！つまり0-1正規化と、チャネル順入れ替えをしない方針
                # Scale(scale=[0.5, 1.5]), 
                # RandomRotation(degrees=(-10, 10)),
                # RandomMirror(),
                Resize(input_size), 
                # Normalize(color_mean, color_std)  # 色情報の標準化とテンソル化 
            ]),
            'val': Compose([
                # ToTensor(),  # テンソルに変換 ############ToTensorを使わない方針！つまり0-1正規化と、チャネル順入れ替えをしない方針
                Resize(input_size), 
                # Normalize(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)


class MFNetDataset(data.Dataset):
    """
    MFNetのDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み, ToTensorを使わずにテンソルに変換
        image_file_path = self.img_list[index]
        img_np = np.array(Image.open(image_file_path))   # [色RGB][高さ][幅]

        img = to_tensor(Image.open(image_file_path))   # [高さ][幅][色RGB]


        # 2. アノテーション画像読み込み, ToTensorを使わずにテンソルに変換
        anno_file_path = self.anno_list[index]
        anno_class_img_np = np.array(Image.open(anno_file_path))   # [高さ][幅]
        anno_class_img = torch.from_numpy(anno_class_img_np)   # [高さ][幅]
        anno_class_img = torch.unsqueeze(anno_class_img,0)


        # 3. 前処理を実施
        img = self.transform(self.phase, img)
        anno_class_img = self.transform(self.phase, anno_class_img)
        # print("in dataset:torch.sum(anno_class_img_transformed)", torch.sum(anno_class_img))

        return img, anno_class_img


if __name__ == "__main__":
    # (RGB)の色の平均値と標準偏差
    color_mean = (0.232, 0.267, 0.233)
    color_std = (0.173, 0.173, 0.172)

    # データセット作成
    train_dataset = MFNetDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size=500, color_mean=color_mean, color_std=color_std))

    val_dataset = MFNetDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size=500, color_mean=color_mean, color_std=color_std))

    # データローダーの作成
    batch_size = 2 # 20250430に変更
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 動作の確認
    batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
    imges, anno_class_imges = next(batch_iterator)  # 1番目の要素を取り出す