import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation.deeplabv3 import DeepLabV3

def init_ICRNet(n_resnet_layers=50, n_classes=21):
    ls_resnet_layes = [50, 101]

    if n_resnet_layers not in ls_resnet_layes:
        raise ValueError('n_resnet_layers must be 50 or 101')
    else:
        if n_resnet_layers == 50:
            model = ICRNet('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
        else:
            model = ICRNet('pytorch/vision:v0.10.0', 'deeplabv3_resnet101')
    
    return model


class ICRNet(DeepLabV3):
    def __init__(self, backbone, classifier):
        super().__init__(backbone, classifier)
        self.name = 'ICRNet50/101'
    
    def name(self):
        return self.name


if __name__ == '__main__':
    model = init_ICRNet(n_resnet_layers=50, n_classes=21)
    print(model)
    print("konnnichiwa")
    print(model.name)









# # original_50 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# class ICRNet50(torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')):
#     def __init__(self):
#         super().__init__()
    
#     def __class__(self):
#         __name__ = 'ICRNet50'

#     def name(self):
#         return self.name

# # original_101 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# class ICRNet101(torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101')):
    # def __init__(self):
    #     super().__init__()
    #     self.name = 'ICRNet101'

    # def name(self):
    #     return self.name





import torch
from PIL import Image
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

if __name__ == '__main__':
    for x in dir(model):
        print( x, ':', type(eval("model."+x)))

    print(type(model))
    # help(model)

    # ダミーデータの作成と計算
    batch_size = 2
    dummy_img = torch.rand(batch_size, 3, 475, 475)
    outputs = model(dummy_img)
    # print(outputs)



if __name__ == '__main__':
    # Download an example image from the pytorch website
    import urllib
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.232, 0.267, 0.233], std=[0.173, 0.173, 0.172]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # # create a color pallette, selecting a color for each class
    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    # r.putpalette(colors)

    # import matplotlib.pyplot as plt
    # plt.savefig("20250214_deeplab1.png")
    # plt.show()


if __name__ == '__main__':
    # Load the saved image and display its pixel values
    loaded_image = Image.open("20250214_deeplab1.png")
    loaded_image = loaded_image.convert("RGB")
    pixel_values = list(loaded_image.getdata())
    # print(pixel_values)