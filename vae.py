import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    编码器块（EncoderBlock）。

    该模块实现了一个卷积编码器块，用于逐步下采样和提取图像特征。
    """
    def __init__(self, base_channel):
        """
        初始化编码器块。

        参数:
            base_channel (int): 基础通道数，用于定义每个卷积层的输出通道数。
        """
        super().__init__()
        self.base_channel = base_channel

        # 定义卷积层序列
        self.conv = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(
                in_channels=3,
                out_channels=self.base_channel,
                kernel_size=4, padding=1, stride=2),  # 输出特征图大小为 16x16（假设输入为 32x32）
            nn.LeakyReLU(0.2, inplace=True),

            # 第二个卷积层
            nn.Conv2d(
                in_channels=self.base_channel,
                out_channels=self.base_channel*2,
                kernel_size=4, padding=1, stride=2),  # 输出特征图大小为 8x8
            nn.LeakyReLU(0.2, inplace=True),

            # 第三个卷积层
            nn.Conv2d(
                in_channels=self.base_channel*2,
                out_channels=self.base_channel*4,
                kernel_size=4, padding=1, stride=2),  # 输出特征图大小为 4x4
            nn.LeakyReLU(0.2, inplace=True),

            # 第四个卷积层
            nn.Conv2d(
                in_channels=self.base_channel*4,
                out_channels=self.base_channel*8,
                kernel_size=4, padding=1, stride=2),  # 输出特征图大小为 2x2
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        """
        前向传播函数，执行编码器块的操作。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 3, H, W)。

        返回:
            torch.Tensor: 编码器块的输出，形状为 (batch_size, base_channel*8, H/16, W/16)。
        """
        return self.conv(x)


class UpsampleDecoder(nn.Module):
    """
    上采样解码器（UpsampleDecoder）。

    该模块实现了一个使用上采样和卷积层的解码器，用于将潜在空间表示逐步上采样并转换为原始图像。
    """
    def __init__(self, latent_dim):
        """
        初始化上采样解码器。

        参数:
            latent_dim (int): 潜在空间的维度。
        """
        super().__init__()
        self.latent_dim = latent_dim
        base_channel = 64
        self.network = nn.Sequential(
            # 第一个上采样层，使用双线性插值，缩放因子为 4
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            # 第一个卷积层，输入通道数为 latent_dim，输出通道数为 base_channel*8
            nn.Conv2d(
                in_channels=latent_dim,
                out_channels=base_channel*8,
                bias=False,
                kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*8), # 第一个批量归一化层
            nn.ReLU(True),  # 输出特征图大小为 4x4

            # 第二个上采样层，使用双线性插值，缩放因子为 2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 第二个卷积层，输入通道数为 base_channel*8，输出通道数为 base_channel*4
            nn.Conv2d(in_channels=base_channel*8,
                      out_channels=base_channel*4,
                      bias=False,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*4), # 第二个批量归一化层
            nn.ReLU(True),  # 输出特征图大小为 8x8

            # 第三个上采样层，使用双线性插值，缩放因子为 2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 第三个卷积层，输入通道数为 base_channel*4，输出通道数为 base_channel*2
            nn.Conv2d(in_channels=base_channel*4,
                      out_channels=base_channel*2,
                      bias=False,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*2), # 第三个批量归一化层
            nn.ReLU(True),  # 输出特征图大小为 16x16

            # 第四个上采样层，使用双线性插值，缩放因子为 2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 第四个卷积层，输入通道数为 base_channel*2，输出通道数为 3（RGB 图像）
            nn.Conv2d(in_channels=base_channel*2,
                      out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()  # 输出特征图大小为 32x32
        )

    def forward(self, x):
        """
        前向传播函数，执行解码器的操作。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, latent_dim)。

        返回:
            torch.Tensor: 解码器的输出，形状为 (batch_size, 3, 32, 32)。
        """
        # 在最后两个维度上增加维度，以匹配卷积层的输入形状
        return self.network(x.unsqueeze(-1).unsqueeze(-1))


class VAE(nn.Module):
    """
    变分自编码器（VAE）。

    该模块实现了变分自编码器，包括编码器和解码器，以及重参数化技巧。
    """

    def __init__(self, latent_dim):
        """
        初始化变分自编码器。

        参数:
            latent_dim (int): 潜在空间的维度。
        """
        super().__init__()
        self.latent_dim = latent_dim
        base_channel = 64
        # 计算线性层的输入维度
        self.lin_in_dim = 2*2*base_channel*8

        # define encoder block
        # 定义编码器块
        self.encoder = EncoderBlock(base_channel)

        # 定义线性层序列，用于将编码器的输出映射到潜在空间
        self.lin1 = nn.Sequential(
            nn.Linear(self.lin_in_dim, latent_dim), # 全连接层，输入维度为 lin_in_dim，输出维度为 latent_dim
            nn.ReLU(),
        )

        # linear layers for mu and logvar prediction
        # 定义线性层，用于预测均值和标准差的对数
        self.lin11 = nn.Linear(latent_dim, latent_dim) # 均值预测层
        self.lin12 = nn.Linear(latent_dim, latent_dim) # 对数方差预测层

        # decoder block
        # 定义解码器块
        self.decoder = UpsampleDecoder(latent_dim)

    def reparametrize(self, mu, logvar):
        """
        重参数化技巧。

        参数:
            mu (torch.Tensor): 均值张量。
            logvar (torch.Tensor): 对数方差张量。

        返回:
            torch.Tensor: 重参数化后的潜在变量。
        """
        std = torch.exp(0.5 * logvar) # 计算标准差
        eps = torch.randn_like(std) # 从标准正态分布中采样
        return eps.mul(std).add_(mu) # 计算重参数化后的潜在变量

    def encode(self, x):
        """
        编码函数，将输入数据编码到潜在空间。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 重参数化后的潜在变量，均值和对数方差。
        """
        # 通过编码器提取特征
        z = self.encoder(x)
        # 展平特征
        z = z.view(-1, self.lin_in_dim)
        # 通过线性层映射到潜在空间
        z = self.lin1(z)
        # 预测均值
        mu = self.lin11(z)
        # 预测对数方差
        logvar = self.lin12(z)
        # 应用重参数化技巧
        z = self.reparametrize(mu, logvar)
        # 返回潜在变量，均值和对数方差
        return z, mu, logvar

    def forward(self, x):
        """
        前向传播函数，执行变分自编码器的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 解码后的输出，均值和对数方差。
        """
        # 编码输入数据
        z, mu, logvar = self.encode(x)
        # 解码潜在变量
        x_hat = self.decoder(z)
        # 返回解码后的输出，均值和对数方差
        return x_hat, mu, logvar
