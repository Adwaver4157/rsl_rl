import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, mlp_input_dim, image_channels, critic_hidden_dims, activation=nn.ReLU()):
        super(CriticNetwork, self).__init__()

        # 画像の処理 (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(2, 2),  # Downsample: 128x128 → 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(2, 2),  # Downsample: 64x64 → 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(2, 2),  # Downsample: 32x32 → 16x16
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(2, 2),  # Downsample: 16x16 → 8x8
            nn.AdaptiveAvgPool2d((8, 8))  # Fixed-size output (8x8)
        )

        # CNN の出力サイズ
        self.cnn_output_dim = 128 * 8 * 8  # 128ch × 8 × 8 = 8192

        # 1次元の観測の処理 (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, critic_hidden_dims[0]),
            activation
        )

        # 結合後の MLP (Critic Head)
        concat_dim = self.cnn_output_dim + critic_hidden_dims[0]
        critic_layers = [nn.Linear(concat_dim, critic_hidden_dims[1]), activation]

        for l in range(1, len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
            critic_layers.append(activation)

        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))  # 状態の価値を出力
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, mlp_obs, image_obs):
        # 画像の特徴抽出
        image_features = self.cnn(image_obs)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten

        # 1次元観測の特徴抽出
        mlp_features = self.mlp(mlp_obs)

        # 結合
        concat_features = torch.cat((mlp_features, image_features), dim=1)

        # 状態の価値を出力
        return self.critic(concat_features)


if __name__ == "__main__":
    # 使用例
    mlp_input_dim = 10  # 1次元観測の次元
    image_channels = 3  # RGB画像
    image_size = 64  # 64x64の画像
    critic_hidden_dims = [128, 256, 128]  # 隠れ層のサイズ

    critic_model = CriticNetwork(mlp_input_dim, image_channels, critic_hidden_dims)
    mlp_obs = torch.randn(5, mlp_input_dim)  # バッチサイズ5の1次元観測
    image_obs = torch.randn(5, image_channels, image_size, image_size)  # バッチサイズ5の画像観測

    value = critic_model(mlp_obs, image_obs)
    print(value.shape)  # (5, 1) → 状態の価値
