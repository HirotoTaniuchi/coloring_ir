import torch
import torch.nn as nn
import torchvision.models as models

class FEANet2(nn.Module):
    def __init__(self, input_nc, output_nc, verbose=True, resnet_type='resnet50'):
        super(FEANet2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.verbose = verbose
        if resnet_type == 'resnet18':
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif resnet_type == 'resnet34':
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif resnet_type == 'resnet50':
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif resnet_type == 'resnet101':
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif resnet_type == 'resnet152':
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048
        else:
            raise NotImplementedError(f'model {resnet_type} is not implemented')

        # FEAModule
        self.feam = FEAModule()

        ########  Thermal ENCODER  ########
        self.is_4ch = False
        if self.is_4ch:
            self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
            self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        else:
            self.encoder_thermal_conv1 = resnet_raw_model1.conv1
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4
        
        ######## DECODER ########
        self.decoder = FEADecoder(input_nc=self.inplanes, output_nc=self.output_nc)

    def forward(self, input):
        # inputはRGB-Tの4chで[r, g, b, t]
        self.is_4ch = False
        if input.shape[1] == 4:
            rgb = input[:, :3, ...]
            thermal = input[:, 3:, ...]  # 3:と書かないと次元が減る
        elif input.shape[1] == 3:
            thermal = input
            self.is_4ch = False
        else:
            raise NotImplementedError(f"The number of channels {input.shape[1]} is invalid. ")
        # self.verbose = Trueで詳細表示
        if self.verbose: print("----------FETNet----------")
        if self.verbose: print(f"thermal_input : {thermal.shape}")
        if self.verbose and self.is_4ch: print(f"rgb_input     : {rgb.shape}")

        # Encoder
        # ----------L0----------
        thermal = self.encoder_thermal_conv1(thermal)
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu(thermal)
        thermal = self.feam(thermal)
        if self.verbose: print(f"thermal_L0 out: {thermal.shape}")
        
        if self.is_4ch:
            rgb = self.encoder_rgb_conv1(rgb)
            rgb = self.encoder_rgb_bn1(rgb)
            rgb = self.encoder_rgb_relu(rgb)
            rgb = self.feam(rgb)
            if self.verbose: print(f"rgb_L0 out    : {rgb.shape}")
            rgb += thermal

        # ----------L1----------
        thermal = self.encoder_thermal_maxpool(thermal)
        thermal = self.encoder_thermal_layer1(thermal)
        thermal = self.feam(thermal)
        if self.verbose: print(f"thermal_L1 out: {thermal.shape}")

        if self.is_4ch:
            rgb = self.encoder_rgb_maxpool(rgb)
            rgb = self.encoder_rgb_layer1(rgb)
            rgb = self.feam(rgb)
            if self.verbose: print(f"rgb_L1 out    : {rgb.shape}")
            rgb += thermal

        # ----------L2----------
        thermal = self.encoder_thermal_layer2(thermal)
        thermal = self.feam(thermal)
        if self.verbose: print(f"thermal_L2 out: {thermal.shape}")

        if self.is_4ch:
            rgb = self.encoder_rgb_layer2(rgb)
            rgb = self.feam(rgb)
            if self.verbose: print(f"rgb_L2 out    : {rgb.shape}")
            rgb += thermal

        # ----------L3----------
        thermal = self.encoder_thermal_layer3(thermal)
        thermal = self.feam(thermal)
        if self.verbose: print(f"thermal_L3 out: {thermal.shape}")

        if self.is_4ch:
            rgb = self.encoder_rgb_layer3(rgb)
            rgb = self.feam(rgb)
            if self.verbose: print(f"rgb_L3 out    : {rgb.shape}")
            rgb += thermal

        # ----------L4----------
        thermal = self.encoder_thermal_layer4(thermal)
        thermal = self.feam(thermal)
        if self.verbose: print(f"thermal_L4 out: {thermal.shape}")
        if self.is_4ch:
            rgb = self.encoder_rgb_layer4(rgb)
            rgb = self.feam(rgb)
            if self.verbose: print(f"rgb_L4 out    : {rgb.shape}")
            rgb += thermal

        # ----------Decoder----------
        if self.is_4ch:
            rgb = self.decoder(rgb)
            return rgb
        else:
            thermal = self.decoder(thermal)
            return thermal

class FEANet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, verbose=False, resnet_type='resnet34'):
        super(FEANet,self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        layer_0 = []
        layer_1 = []
        layer_2 = []
        layer_3 = []
        layer_4 = []
        layers = [layer_0, layer_1, layer_2, layer_3, layer_4]
        self.verbose = verbose
        fea_on = False

        if resnet_type == 'resnet34':
            # n_block = [(# of residual blocks in conv2), ... , (... conv5)]
            # conv2のmaxpoolingは除く
            n_blocks = [3, 4, 6, 3]
        else:
            raise NotImplementedError(f"{resnet_type} is not implemented")

        # ----------L0----------
        padding_type = 'zero'
        pad = 0
        norm = 'batch'
        # normalize
        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError(f'normalization {norm} is not implemented')
        # padding
        if padding_type == 'reflect':
            layer_0 += [nn.ReflectionPad2d(3)]
        elif padding_type == 'zero':
            pad = 3
        else:
            raise NotImplementedError(f'padding {padding_type} is not implemented')
        layer_0 += [nn.Conv2d(in_channels=self.input_nc, out_channels=self.ngf, kernel_size=7,\
                 stride=2, padding=pad),
                 norm_layer(ngf),
                 nn.LeakyReLU()]
        if fea_on: layer_0 += [FEAModule()]
        self.layer_0 = nn.Sequential(*layer_0)

        # ----------L1----------
        layer_1 += [ResnetLayer(input_nc=self.ngf, output_nc=self.ngf, n_blocks=n_blocks[0]+1, is_conv2=True),
                  FEAModule()]
        self.layer_1 = nn.Sequential(*layer_1)

        # ----------L2からL4----------
        for i in range(3):
            mult = 2 ** i
            layers[i+2] += [ResnetLayer(input_nc=self.ngf * mult, output_nc=self.ngf * mult * 2, \
                            n_blocks=n_blocks[i+1], is_conv2=False)]
            if fea_on: layers[i+2] += [FEAModule()]
        self.layer_2 = nn.Sequential(*layer_2)
        self.layer_3 = nn.Sequential(*layer_3)
        self.layer_4 = nn.Sequential(*layer_4)
        
        # ----------Decoder----------
        self.decoder = FEADecoder(input_nc=self.ngf * mult * 2, output_nc=self.output_nc)
    
    def forward(self, input):
        out_t = []
        if self.input_nc == 3:
            # thermalの処理
            if self.verbose: print("Thermal")
            out_0 = self.layer_0(input)
            if self.verbose: print(f"layer_0: {out_0.shape}")
            out_1 = self.layer_1(out_0)
            if self.verbose: print(f"layer_1: {out_1.shape}")
            out_2 = self.layer_2(out_1)
            if self.verbose: print(f"layer_2: {out_2.shape}")
            out_3 = self.layer_3(out_2)
            if self.verbose: print(f"layer_3: {out_3.shape}")
            out_4 = self.layer_4(out_3)
            if self.verbose: print(f"layer_4: {out_4.shape}")
            # theamalの出力は各レイヤーの出力のlistを返す
            #out_t.append(out_0)
            #out_t.append(out_1)
            #out_t.append(out_2)
            #out_t.append(out_3)
            #out_t.append(out_4)
            return self.decoder(out_4)
        #elif self.input_nc == 1 and type(input_t) == list :
            # RGBの処理
            #print("\nRGB")
            #out_rgb = input_t[0] + self.layer_0(input)
            #print(f"layer_0: {out_rgb.shape}")
            #out_rgb = input_t[1] + self.layer_1(out_rgb)
            #print(f"layer_1: {out_rgb.shape}")
            #out_rgb = input_t[2] + self.layer_2(out_rgb)
            #print(f"layer_2: {out_rgb.shape}")
            #out_rgb = input_t[3] + self.layer_3(out_rgb)
            #print(f"layer_3: {out_rgb.shape}")
            #out_rgb = input_t[4] + self.layer_4(out_rgb)
            #print(f"layer_4: {out_rgb.shape}")
            # Decoder
            #out = self.decoder(out_rgb)
            #return out
        else:
            raise NotImplementedError(f"The number of channels is invalid")

class FEAModule(nn.Module):
    def __init__(self):
        super(FEAModule, self).__init__()
    
    def forward(self, input):
        # channel attention
        channel_att = nn.AdaptiveMaxPool2d(output_size=(1,1))
        out =  torch.mul(input, channel_att(input))

        # spatial attention
        spatial_att = torch.max(input=out, dim=1)[0]
        spatial_att = spatial_att[:, None, ...]
        out = torch.mul(out, spatial_att)        
        return out

class ResnetLayer(nn.Module):
    def __init__(self, input_nc, output_nc, n_blocks, is_conv2=False):
        super(ResnetLayer, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_blocks = n_blocks
        self.is_conv2 = is_conv2
        model = []
        first_block = True
        # conv2なら最初にmaxpooling
        if is_conv2:
            model += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            first_block = False

        # first_blockならBasicBlockはstride=2
        if first_block:
            model += [BasicBlock(input_nc=input_nc, output_nc=output_nc, first_block=first_block)]
            first_block = False

        # 2番目以降のconv_block
        for i in range(n_blocks - 1):
            model += [BasicBlock(input_nc=output_nc, output_nc=output_nc, first_block=first_block)]
            
        # modelをSequentialへ
        self.model = nn.Sequential(*model)

    def forward(self, input):
        out = self.model(input)
        return out

class BasicBlock(nn.Module):
    def __init__(self, input_nc, output_nc, first_block=False):
        super(BasicBlock, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        conv_block = []
        
        # padding
        padding_type = 'reflect'
        pad = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'zero':
            pad = 1
        else:
            raise NotImplementedError(f'padding {padding_type} is not implemented')

        # normalize
        norm = 'batch'
        norm_layer = set_normalize(norm)
        """if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError(f'normalization {norm} is not implemented')"""
        
        # conv layerの最初のレイヤーならstride=2
        if first_block:
            conv_block += [nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=3, stride=2, padding=pad)]
        else:
            conv_block += [nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=3, stride=1, padding=pad)]
        
        conv_block += [norm_layer(num_features=output_nc),
                       nn.LeakyReLU(),
                       nn.Conv2d(in_channels=output_nc, out_channels=output_nc, kernel_size=3, stride=1, padding=pad),
                       norm_layer(num_features=output_nc)]
        
        # conv_blockをSequentialへ
        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        out = self.conv_block(input)
        # shortcut connection
        if self.input_nc == self.output_nc:
            out = input + out
        out = self.relu(out)
        return out

class FEADecoder(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(FEADecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        # ----------D0~D4----------
        model = []
        # Transposed blockを5層重ねる
        for i in range(5):
            output_nc = int(input_nc / 2)
            model += [FEATranspose(input_nc=input_nc, output_nc=output_nc)]
            input_nc = int(input_nc / 2)
        #model += [FEATranspose(input_nc=input_nc, output_nc=self.output_nc)]
        model += [nn.ConvTranspose2d(in_channels=input_nc, out_channels=self.output_nc, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        out = self.model(input)
        return out

class FEATranspose(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(FEATranspose, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        # ----------Transposed Block A----------
        self.A = BasicBlock(input_nc=input_nc, output_nc=input_nc, first_block=False)

        # ----------Transposed Block B----------
        # main block
        block_B = []
        padding_type = 'zero'
        pad = 0
        if padding_type == 'reflect':
            block_B += [nn.ReflectionPad2d(1)]
        elif padding_type == 'zero':
            pad = 1
        else:
            raise NotImplementedError(f"padding {padding_type} is not implemented")
        norm_type = 'batch'
        norm_layer = set_normalize(norm_type=norm_type)
        block_B += [nn.Conv2d(in_channels=input_nc, out_channels=self.output_nc, kernel_size=3, stride=1, padding=pad),
                    norm_layer(self.output_nc),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(in_channels=self.output_nc, out_channels=self.output_nc, kernel_size=2, stride=2, padding=0),
                    norm_layer(self.output_nc)]
        self.B = nn.Sequential(*block_B)
        # skip connection
        self.skip_B = nn.Sequential(nn.ConvTranspose2d(in_channels=input_nc, out_channels=self.output_nc, kernel_size=2, stride=2, padding=0),
                                    norm_layer(self.output_nc))
        # last ReLU
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        output = self.B(input) + self.skip_B(input)
        output = self.relu(output)
        return output

# normalizeを選択する
def set_normalize(norm_type:str):
    """
    select normalization type from 'batch' or 'instance'
    norm_type : str
    """
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError(f'{norm_type} normalization is not implemented')




if __name__ == '__main__':
    batch_size = 4
    H = 480
    W = 640
    input_nc = 3
    output_nc = 3

    input_2 = torch.randn((batch_size, input_nc, H, W))
    print(input_2[0][0])
    model = FEANet(input_nc=input_nc, output_nc=3, resnet_type='resnet34')
    model = FEANet2(verbose=True, resnet_type='resnet152')
    pred = model(input_2)
    #model_2 = FEANet2(resnet_type='resnet34')
    #pred = model_2(input_2)
    print(f"\npred_2 : {pred[0][0]}")
