import torch
import torch.nn as nn

class RGBEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128):
        """
        Parameters:
            input_channels (int): 入力画像のチャネル数（RGB画像の場合は3）
            feature_dim (int): 最終的な特徴ベクトルの次元数
        """
        super(RGBEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # 入力サイズ：(B, input_channels, H, W)
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 空間サイズに依存せず1x1の出力に変換
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class VisionEncoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128):
        """
        Parameters:
            input_channels (int): 入力画像のチャネル数（深度画像の場合は通常1）
            feature_dim (int): 最終的な特徴ベクトルの次元数
        """
        super(VisionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # 入力サイズ：(B, input_channels, H, W)
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 空間サイズに依存せず1x1の出力に変換
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import numpy as np
    # 画像変換の定義：リサイズ、Tensor変換、（必要に応じて正規化）
    transform = transforms.Compose([
        transforms.Resize((320, 240)),  # 画像サイズを128x128にリサイズ
        transforms.ToTensor(),          # PIL画像をTensorに変換 (形状: [C, H, W])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])  # 必要に応じて正規化
    ])
    # 画像に変換を適用
    dummy_input = np.randn(3, 640, 480)
    img_tensor = transform(dummy_input)  # 形状: (3, 128, 128)

    # バッチ次元を追加：モデルの入力は通常 (B, C, H, W) となるので
    img_tensor = img_tensor.unsqueeze(0)  # 形状: (1, 3, 128, 128)
    # ダミーデータの作成：バッチサイズ4、チャネル3、画像サイズ128x128のRGB画像
    vision_encoder = VisionEncoder(input_channels=3, feature_dim=128)
    features = vision_encoder(img_tensor)
    print(features.shape)  # 期待される出力: torch.Size([4, 128])

    # ダミーデータの作成：バッチサイズ4、チャネル1、画像サイズ128x128の深度画像
    depth_encoder = VisionEncoder(input_channels=1, feature_dim=128)
    features = depth_encoder(img_tensor)
    print(features.shape)  # 期待される出力: torch.Size([4, 128])
