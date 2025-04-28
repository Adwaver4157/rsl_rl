import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, mlp_input_dim, image_channels, critic_hidden_dims, activation=nn.ReLU(), use_vis=True, options=None):
        super(CriticNetwork, self).__init__()

        self.options = options
        # pure MLP critic if no image channels
        if not use_vis:
            layers = [nn.Linear(mlp_input_dim, critic_hidden_dims[0]), activation]
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    layers.append(activation)
            self.critic = nn.Sequential(*layers)
            self.is_mlp = True
        else:
            self.is_mlp = False
            # 画像の処理 (CNN)
            self.cnn = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
                activation,
                nn.MaxPool2d(2, 2),  # 例: 128x128 → 64x64
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                activation,
                nn.MaxPool2d(2, 2),  # 64x64 → 32x32
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                activation,
                nn.MaxPool2d(2, 2),  # 32x32 → 16x16
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                activation,
                nn.MaxPool2d(2, 2),  # 16x16 → 8x8
                nn.AdaptiveAvgPool2d((8, 8))  # 固定サイズの出力 (8x8)
            )
            # CNN の出力サイズ: 128ch × 8 × 8 = 8192
            self.cnn_output_dim = 128 * 8 * 8

            # 1次元観測の処理 (MLP) ※critic用
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, critic_hidden_dims[0]),
                activation
            )

            # CNN と MLP の結合後のネットワーク (critic Head)
            concat_dim = self.cnn_output_dim + critic_hidden_dims[0]
            critic_layers = [nn.Linear(concat_dim, critic_hidden_dims[1]), activation]

            for l in range(1, len(critic_hidden_dims) - 1):
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)

            critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))  # 状態の価値を出力
            self.critic = nn.Sequential(*critic_layers)

        if options is not None and "command_refiner" in options:
            # 追加: command refiner のネットワーク
            # mlp_obs 用の小さなMLP
            self.command_refiner_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 64),
                activation
            )

            self.command_refiner_cnn = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
                activation,
                nn.MaxPool2d(2, 2),  # 例: 128x128 → 64x64
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                activation,
                nn.MaxPool2d(2, 2),  # 64x64 → 32x32
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                activation,
                nn.MaxPool2d(2, 2),  # 32x32 → 16x16
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                activation,
                nn.MaxPool2d(2, 2),  # 16x16 → 8x8
                nn.AdaptiveAvgPool2d((8, 8))  # 固定サイズの出力 (8x8)
            )
            # CNN の出力サイズ: 128ch × 8 × 8 = 8192
            self.command_refiner_cnn_output_dim = 128 * 8 * 8
            # CNN 出力 (image_features) を低次元に変換
            self.command_refiner_image_fc = nn.Sequential(
                nn.Linear(self.command_refiner_cnn_output_dim, 64),
                activation
            )
            # 両者を結合して x, y, rotation の調整コマンドを出力
            self.command_refiner_head = nn.Sequential(
                nn.Linear(64 + 64, 32),
                activation,
                nn.Linear(32, 3)
            )

        # refiner と付いているネットワーク以外を重み固定
        if self.options == "command_refiner_fixed":
            for name, param in self.named_parameters():
                if "command_refiner" not in name:
                    param.requires_grad = False

    def forward(self, obs):
        if len(obs) == 2:
            mlp_obs = obs[0]
            image_obs = obs[1]
        else:
            mlp_obs = obs[0]
            image_obs = None

        if self.options is not None and "command_refiner" in self.options:
            # command refiner 用の特徴抽出
            command_refiner_input_image_features = self.command_refiner_cnn(image_obs)
            command_refiner_input_image_features = command_refiner_input_image_features.view(
                command_refiner_input_image_features.size(0), -1)
            refined_mlp_features = self.command_refiner_mlp(mlp_obs)
            refined_image_features = self.command_refiner_image_fc(command_refiner_input_image_features)
            # mlp_obs と image_features の両方から調整コマンドを生成
            concat_refiner_features = torch.cat((refined_mlp_features, refined_image_features), dim=1)
            refined_commands = self.command_refiner_head(concat_refiner_features)  # 出力は (x, y, rotation)

            # mlp_obs の7,8,9番目（Pythonではインデックス6～8）を refined_commands で置換
            # print(f"before command_refiner: {mlp_obs[0, 6:9]}")
            # print(f"before command_refiner: {mlp_obs[0, :]}")
            mlp_obs = mlp_obs.clone()
            mlp_obs[:, 6:9] = refined_commands
            # print(f"after command_refiner: {mlp_obs[0, 6:9]}")
            # print(f"after command_refiner: {mlp_obs[0, :]}")

        if self.is_mlp:
            return self.critic(mlp_obs)
        else:
            # 画像特徴の抽出
            image_features = self.cnn(image_obs)
            image_features = image_features.view(image_features.size(0), -1)  # Flatten
            # 1次元観測の処理
            mlp_features = self.mlp(mlp_obs)
            # 特徴の結合
            concat_features = torch.cat((mlp_features, image_features), dim=1)
            # 最終的な行動の予測
            return self.critic(concat_features)

if __name__ == "__main__":
    import torch
    options = None
    # pure MLP-only example
    mlp_input_dim = 10
    critic_hidden_dims = [128, 256, 128]
    use_vis = False
    image_channels = 1
    model_mlp = CriticNetwork(mlp_input_dim, image_channels, critic_hidden_dims, use_vis=use_vis, options=options)
    mlp_obs = torch.randn(5, mlp_input_dim)
    actions = model_mlp(mlp_obs, None)
    print("MLP-only actions:", actions.shape)

    # vision example
    image_size = 64
    model_vis = CriticNetwork(mlp_input_dim, image_channels, critic_hidden_dims, use_vis=use_vis, options=options)
    image_obs = torch.randn(5, image_channels, image_size, image_size)
    actions_vis = model_vis(mlp_obs, image_obs)
    print("Vision actions:", actions_vis.shape)

    # command_refiner no vision example
    # options = "command_refiner"
    options = "command_refiner_fixed"
    # pure MLP-only example
    mlp_input_dim = 10
    critic_hidden_dims = [128, 256, 128]
    use_vis = False
    image_channels = 1
    image_size = 64
    image_obs = torch.randn(5, image_channels, image_size, image_size)
    model_mlp = CriticNetwork(mlp_input_dim, image_channels, critic_hidden_dims, use_vis=use_vis, options=options)
    mlp_obs = torch.randn(5, mlp_input_dim)
    actions = model_mlp(mlp_obs, image_obs)
    print("MLP-only actions:", actions.shape)

    # vision example
    model_vis = CriticNetwork(mlp_input_dim, image_channels, critic_hidden_dims, use_vis=use_vis, options=options)
    image_obs = torch.randn(5, image_channels, image_size, image_size)
    actions_vis = model_vis(mlp_obs, image_obs)
    print("Vision actions:", actions_vis.shape)
