from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import SVHN


def get_loaders(data_dir, batch_size, split=0.9):
    """
    加载 SVHN 数据集并创建训练集和验证集的数据加载器。

    参数:
        data_dir (str): 数据集存储的目录路径。
        batch_size (int): 每个批次中的样本数量。
        split (float, 可选): 训练集所占的比例，默认为 0.9（90%）。

    返回:
        tuple: 包含训练数据加载器和验证数据加载器的元组。
    """
    # 定义数据预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为张量，并将像素值缩放到 [0, 1]
        transforms.Normalize((.5, .5, .5),  # 对每个通道进行标准化，均值为 (0.5, 0.5, 0.5)
                             (.5, .5, .5))  # 标准差为 (0.5, 0.5, 0.5)
    ]) 

    # 加载 SVHN 数据集，指定为训练集，并应用预处理
    dataset = SVHN(data_dir,
                   split='train',
                   download=True,
                   transform=transform)
    
    # 计算训练集和验证集的大小
    train_len = int(len(dataset) * split)
    val_len = len(dataset) - train_len
    # 将数据集拆分为训练集和验证集
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # 创建训练集的数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4) # 使用 4 个子进程加载数据

    # 创建验证集的数据加载器
    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4)

    # 返回训练集和验证集的数据加载器
    return train_loader, valid_loader
