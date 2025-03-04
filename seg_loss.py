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

class SegLoss(nn.Module):
    """DeeplabV3の損失関数クラス"""

    def __init__(self, aux_weight=0.4, gamma=2.0, alpha=0.25):
        super(SegLoss, self).__init__()
        self.aux_weight = aux_weight  # aux_lossの重み
        self.gamma = gamma  # Focal Lossのgamma値
        self.alpha = alpha  # Focal Lossのalpha値

    def forward(self, outputs, targets):
        """
        損失関数の計算。
        入力時deeplabv3の出力(OrderedDict型)をTensorに変換しておく必要がある↓

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

        c = 2.0
        d = 2.0
        f = 2.0

        # crossentropylossの引数に合わせるために、targetをsqueezeする
        targets = torch.squeeze(targets)
        
        # Mask to ignore class 0
        mask = targets != 0
        
        # Focal Loss
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        ce_loss = ce_loss[mask]
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        # Dice Loss
        smooth = 1e-5
        outputs = torch.argmax(outputs, dim=1).float()  # Convert outputs to class indices and then to float
        targets = targets.float()  # Convert targets to float
        iflat = outputs.contiguous().view(-1)
        tflat = targets.contiguous().view(-1)
        mask_flat = mask.contiguous().view(-1)
        intersection = (iflat * tflat * mask_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / ((iflat * mask_flat).sum() + (tflat * mask_flat).sum() + smooth))

        # Class-balanced Loss
        beta = 0.999
        class_counts = torch.bincount(targets[mask].view(-1).long(), minlength=outputs.size(1))
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * outputs.size(1)
        cb_loss = F.cross_entropy(outputs, targets.squeeze(1), weight=weights, reduction='none')
        cb_loss = cb_loss.mean()

        return d * dice_loss + f * focal_loss + c * cb_loss
    

if __name__ == '__main__':
    # モデルの定義
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()  # 評価モードに設定

    # モデルの出力
    with torch.no_grad():
        outputs = torch.randn(4, 21, 500, 666)

    # 正解データ
    targets = torch.randint(0, 21, (4, 1, 500, 666), dtype=torch.long)

    # 損失関数の計算
    criterion = SegLoss()
    loss = criterion(outputs, targets)
    print(loss)
    print(loss.item())
