import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, mlp_input_dim, image_channels, actor_hidden_dims, num_actions, activation=nn.ReLU()):
        super(ActorNetwork, self).__init__()

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

        # CNN の出力サイズを計算
        self.cnn_output_dim = 128 * 8 * 8  # 128ch × 8 × 8 = 8192

        # 1次元の観測の処理 (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, actor_hidden_dims[0]),
            activation
        )

        # 結合後の MLP (Actor Head)
        concat_dim = self.cnn_output_dim + actor_hidden_dims[0]
        actor_layers = [nn.Linear(concat_dim, actor_hidden_dims[1]), activation]

        for l in range(1, len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            actor_layers.append(activation)

        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, mlp_obs, image_obs):
        # 画像の特徴抽出
        print("before image_features.shape", image_obs.shape)
        print("before mlp_features.shape", mlp_obs.shape)
        image_features = self.cnn(image_obs)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten

        # 1次元観測の特徴抽出
        mlp_features = self.mlp(mlp_obs)

        # 結合
        print("after image_features.shape", image_features.shape)
        print("after mlp_features.shape", mlp_features.shape)
        concat_features = torch.cat((mlp_features, image_features), dim=1)

        # 行動の予測
        return self.actor(concat_features)

if __name__ == "__main__":
    # 使用例
    mlp_input_dim = 10  # 1次元観測の次元
    image_channels = 3  # RGB画像
    image_size = 64  # 64x64の画像
    actor_hidden_dims = [128, 256, 128]  # 隠れ層のサイズ
    num_actions = 4  # 出力する行動の数

    model = ActorNetwork(mlp_input_dim, image_channels, actor_hidden_dims, num_actions)
    mlp_obs = torch.randn(5, mlp_input_dim)  # バッチサイズ5の1次元観測
    image_obs = torch.randn(5, image_channels, image_size, image_size)  # バッチサイズ5の画像観測

    actions = model(mlp_obs, image_obs)
    print(actions.shape)  # (5, num_actions)
