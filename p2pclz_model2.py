import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDown(nn.Module):
    """U-Net のダウンサンプリングブロック"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """U-Net のアップサンプリングブロック"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorUNet(nn.Module):
    """U-Net形式のGenerator"""
    def __init__(self, in_channels=1, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # エンコーダー部分
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # デコーダー部分（スキップ接続を使用）
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, in_channels=4):  # 入力: グレースケール画像 + 生成されたカラー画像
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_gray, img_color):
        # 入力画像と出力画像を連結
        img_input = torch.cat((img_gray, img_color), 1)
        return self.model(img_input)


class Pix2PixColorization():
    """Pix2Pixを用いた画像着色モデル"""
    def __init__(self, device='cuda'):
        self.device = device
        # グレースケール画像（1チャンネル）からカラー画像（3チャンネル）を生成
        self.generator = GeneratorUNet(in_channels=1, out_channels=3).to(device)
        # 判別器は入力（1チャンネル）と出力（3チャンネル）を連結して4チャンネルを受け取る
        self.discriminator = Discriminator(in_channels=1+3).to(device)
        
        # 最適化アルゴリズム
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 損失関数
        self.criterion_GAN = nn.MSELoss()  # GAN損失
        self.criterion_pixelwise = nn.L1Loss()  # ピクセルワイズ損失（L1）
        
        # ピクセルワイズ損失の重み
        self.lambda_pixel = 100

    def train_step(self, gray_images, color_images):
        """1ステップの訓練を実行"""
        # 本物のラベルと偽物のラベル
        real = torch.ones((gray_images.size(0), 1, 16, 16), requires_grad=False).to(self.device)
        fake = torch.zeros((gray_images.size(0), 1, 16, 16), requires_grad=False).to(self.device)

        # ------------------
        # Generatorの訓練
        # ------------------
        self.optimizer_G.zero_grad()

        # 着色画像を生成
        gen_color = self.generator(gray_images)

        # Generatorの損失を計算
        pred_fake = self.discriminator(gray_images, gen_color)
        loss_GAN = self.criterion_GAN(pred_fake, real)
        
        # ピクセルワイズ損失
        loss_pixel = self.criterion_pixelwise(gen_color, color_images)
        
        # 合計損失
        loss_G = loss_GAN + self.lambda_pixel * loss_pixel

        loss_G.backward()
        self.optimizer_G.step()

        # ------------------
        # Discriminatorの訓練
        # ------------------
        self.optimizer_D.zero_grad()

        # 本物の画像に対する判別
        pred_real = self.discriminator(gray_images, color_images)
        loss_real = self.criterion_GAN(pred_real, real)

        # 生成された偽画像に対する判別
        pred_fake = self.discriminator(gray_images, gen_color.detach())
        loss_fake = self.criterion_GAN(pred_fake, fake)

        # 合計損失
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        self.optimizer_D.step()

        return {
            "loss_G": loss_G.item(),
            "loss_pixel": loss_pixel.item(),
            "loss_GAN": loss_GAN.item(),
            "loss_D": loss_D.item()
        }

    def save_models(self, path):
        """モデルの保存"""
        torch.save(self.generator.state_dict(), f"{path}/generator.pth")
        torch.save(self.discriminator.state_dict(), f"{path}/discriminator.pth")

    def load_models(self, path):
        """モデルの読み込み"""
        self.generator.load_state_dict(torch.load(f"{path}/generator.pth"))
        self.discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth"))

    def colorize(self, gray_image):
        """グレースケール画像を着色"""
        self.generator.eval()
        with torch.no_grad():
            colored = self.generator(gray_image)
        self.generator.train()
        return colored
    

if __name__ == '__main__':
    input = torch.randn(1, 1, 256, 256)
    model = Pix2PixColorization()
    output = model.colorize(input)
    print(output.size())
    # print(model.generator)
    # print(model.discriminator)