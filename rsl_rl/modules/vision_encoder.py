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


class DepthEncoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128):
        """
        Parameters:
            input_channels (int): 入力画像のチャネル数（深度画像の場合は通常1）
            feature_dim (int): 最終的な特徴ベクトルの次元数
        """
        super(DepthEncoder, self).__init__()
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
    # ダミーデータの作成：バッチサイズ4、チャネル3、画像サイズ128x128のRGB画像
    dummy_input = torch.randn(4, 3, 128, 128)
    model = RGBEncoder(input_channels=3, feature_dim=128)
    features = model(dummy_input)
    print(features.shape)  # 期待される出力: torch.Size([4, 128])

    # ダミーデータの作成：バッチサイズ4、チャネル1、画像サイズ128x128の深度画像
    dummy_input = torch.randn(4, 1, 128, 128)
    model = DepthEncoder(input_channels=1, feature_dim=128)
    features = model(dummy_input)
    print(features.shape)  # 期待される出力: torch.Size([4, 128])
