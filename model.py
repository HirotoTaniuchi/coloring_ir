# 全然合ってる自信がないので、とりあえずモデルの定義だけ

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_ICRNet(n_resnet_layers=50, n_classes=21):
    ls_resnet_layes = [50, 101]

    if n_resnet_layers not in ls_resnet_layes:
        raise ValueError('n_resnet_layers must be 50 or 101')
    else:
        if n_resnet_layers == 50:
            model = ICRNet50()
        else:
            model = ICRNet101()
    
    return model


# original_50 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
class ICRNet50(torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')):
    def __init__(self):
        super().__init__()
        self.name = 'ICRNet50'

    def name(self):
        return self.name
    

# original_101 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
class ICRNet101(torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101')):
    def __init__(self):
        super().__init__()
        self.name = 'ICRNet101'

    def name(self):
        return self.name




if __name__ == '__main__':
    model = init_ICRNet(n_resnet_layers=50, n_classes=21)
    print(model)
    print("konnnichiwa")
    print(model.name)