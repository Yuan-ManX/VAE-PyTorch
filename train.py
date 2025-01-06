import torch
import torch.nn as nn
from torch.optim import Adam

from loader import get_loaders
from vae import VAE


def compute_loss(inputs, outputs, mu, logvar):
    """
    计算 VAE 的总损失，包括重建损失和 KL 散度损失。

    参数:
        inputs (torch.Tensor): 输入张量，形状为 (batch_size, C, H, W)。
        outputs (torch.Tensor): 解码器输出张量，形状与 inputs 相同。
        mu (torch.Tensor): 均值张量，形状为 (batch_size, latent_dim)。
        logvar (torch.Tensor): 对数方差张量，形状为 (batch_size, latent_dim)。

    返回:
        torch.Tensor: 总损失值。
    """
    # 计算重建损失，使用均方误差损失函数，并设置 reduction 为 'sum' 以对批次求和
    reconstruction_loss = nn.MSELoss(reduction='sum')(inputs, outputs)
    # 计算 KL 散度损失，公式为 -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    # 总损失为重建损失加上 KL 散度损失
    return kl_loss + reconstruction_loss


def train_vae():
    """
    训练变分自编码器（VAE）模型。

    该函数加载数据、初始化模型和优化器，并进行训练和验证。
    使用早停法（Early Stopping）来防止过拟合。
    """
    # 定义训练参数
    batch_size = 64  # 每个批次的样本数量
    epochs = 100  # 训练的总轮数
    latent_dimension = 100  # 潜在空间的维度
    patience = 10  # 早停法的容忍轮数

    # 设置设备，如果 CUDA 可用则使用 GPU，否则使用 CPU
    device = torch.device('cuda:0') \
        if torch.cuda.is_available() \
        else torch.device('cpu')

    # load data
    # 加载数据，使用 get_loaders 函数获取训练集和验证集的数据加载器
    train_loader, valid_loader = get_loaders('data', batch_size)

    # 初始化 VAE 模型，并将其移动到指定设备
    model = VAE(latent_dimension).to(device)

    # 初始化 Adam 优化器，指定模型的参数和学习率
    optim = Adam(model.parameters(), lr=1e-3)

    # intialize variables for early stopping
    # 初始化早停法的变量
    val_greater_count = 0  # 验证损失连续增加的次数
    last_val_loss = 0  # 上一次验证损失

    # 开始训练循环
    for e in range(epochs):
        # 初始化当前轮次的总损失
        running_loss = 0
        # 设置模型为训练模式
        model.train()
        for _, (images, _) in enumerate(train_loader):
            images = images.to(device)
            model.zero_grad()
            outputs, mu, logvar = model(images)  # 前向传播，获取输出和潜在变量
            loss = compute_loss(images, outputs, mu, logvar)  # 计算损失
            running_loss += loss  # 累加损失
            loss.backward()  # 反向传播
            optim.step()  # 更新模型参数

        # 计算平均损失
        running_loss = running_loss/len(train_loader)
        # 设置模型为评估模式
        model.eval()
        with torch.no_grad():
            val_loss = 0  # 初始化验证损失
            for images, _ in valid_loader:
                images = images.to(device)
                outputs, mu, logvar = model(images)  # 前向传播，获取输出和潜在变量
                loss = compute_loss(images, outputs, mu, logvar)  # 计算损失
                val_loss += loss  # 累加损失
            val_loss /= len(valid_loader)  # 计算平均验证损失

        # increment variable for early stopping
        # 早停法逻辑
        if val_loss > last_val_loss:
            # 验证损失增加，计数器加 1
            val_greater_count += 1
        else:
            # 验证损失减少，计数器重置
            val_greater_count = 0
        # 更新上一次验证损失
        last_val_loss = val_loss

        # save model
        torch.save({
            'epoch': e,
            'model': model.state_dict(),
            'running_loss': running_loss,
            'optim': optim.state_dict(),
        }, "checkpoint_{}.pth".format(e))
        print("Epoch: {} Train Loss: {}".format(e+1, running_loss.item()))
        print("Epoch: {} Val Loss: {}".format(e+1, val_loss.item()))

        # check early stopping condition
        if val_greater_count >= patience:
            break


if __name__ == '__main__':
    
    train_vae()
