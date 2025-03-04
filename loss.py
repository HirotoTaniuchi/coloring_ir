import torch
import torch.nn as nn
import torch.nn.functional as F

# def calculate_psnr(original, reconstructed):
    # """
    # https://qiita.com/megane-9mm/items/bb3b2896c450a4abb3cd#psnr%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
    # """
#     original = original.to(device)
#     reconstructed = reconstructed.to(device)
#     mse = F.mse_loss(original, reconstructed)  # 平均二乗誤差の計算
#     mse = mse.to(device)
#     psnr = 10 * torch.log10(1 / mse)  # PSNRの計算
#     return psnr

class ICRLoss(nn.Module):
    """ICRNetの損失関数クラス"""

    def __init__(self, aux_weight=0.4):
        super(ICRLoss, self).__init__()
        self.aux_weight = aux_weight # aux_lossの重み

    def forward(self, outputs, targets):
        """
        損失関数の計算。
        入力時deeplabv3の出力(OrderedDict型)をTensorに変換しておく必要がある↓

        --All pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (N, 3, H, W),
        where N is the number of images, H and W are expected to be at least 224 pixels.
        The images have to be loaded in to a range of [0, 1] and then normalized using
        mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        The model returns an OrderedDict with two Tensors that are of the same height and width
        as the input Tensor, but with 21 classes. output['out'] contains the semantic masks,
        and output['aux'] contains the auxiliary loss values per-pixel.
        In inference mode, output['aux'] is not useful. So, output['out'] is of shape (N, 21, H, W).--

        Parameters
        ----------
        outputs : torch.Size([N, 2, H, W])
            ネットワークからの出力

        targets : torch.Size([N, H, W])
            正解のアノテーション情報

        Returns
        -------
        loss : torch.tensor
            損失の値
        """




        # crossentropylossの引数に合わせるために、targetをsqueezeする
        targets = torch.squeeze(targets)
        # print("outputs.shape=", outputs.shape, "ouputs.dtype=", outputs.dtype)
        # print("targets.shape=", targets.shape, "targets.dtype=", targets.dtype)
        # loss = nn.CrossEntropyLoss(outputs, targets, ignore_index=255)
        # これはだめ!! # https://stackoverflow.com/questions/52946920/bool-value-of-tensor-with-more-than-one-value-is-ambiguous-in-pytorch
        loss = nn.CrossEntropyLoss()
        loss_aux = nn.CrossEntropyLoss()
        loss = loss(outputs, targets)
        loss_aux = loss_aux(outputs, targets)

        return loss + self.aux_weight * loss_aux