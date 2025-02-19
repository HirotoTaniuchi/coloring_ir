import torch.nn as nn
import torch.nn.functional as F

class ICRLoss(nn.Module):
    """ICRNetの損失関数クラス"""

    def __init__(self, aux_weight=0.4):
        super(ICRLoss, self).__init__()
        self.aux_weight = aux_weight # aux_lossの重み

    def forward(self, outputs, targets):
        """
        損失関数の計算。

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

        print("lets go")
        print("outputs", outputs)
        # print("outputs.shape", outputs.shape)
        print("targets.shape", targets.shape)
        # print("outputs.type", outputs.type)
        print("targets.type", targets.type)

        loss = F.cross_entropy(outputs, targets, ignore_index=255)
        loss_aux = F.cross_entropy(outputs, targets, ignore_index=255)

        return loss + self.aux_weight * loss_aux