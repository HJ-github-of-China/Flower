import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import torch


class ChestDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_names=("NORMAL", "PNEUMONIA")):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names
        self.image_paths = []  # 存储所有图像路径
        self.labels = []  # 存储对应标签（0:NORMAL, 1:PNEUMONIA）

        # 遍历每个类别目录收集数据
        for label_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            # 支持jpg/jpeg/png等常见医学影像格式
            for img_path in glob(os.path.join(class_dir, "*.jpeg")):
                self.image_paths.append(img_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)  # 修改为图像列表长度

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # 关键修改：强制统一为RGB三通道
        if image.mode != "RGB":
            if image.mode == "L":
                # 灰度图→RGB（复制单通道到3个通道）
                image = Image.merge("RGB", (image, image, image))
            elif image.mode == "P":
                # 索引图→RGB（先转灰度，再复制）
                image = image.convert("L").convert("RGB")

        # 后续预处理（如Resize、ToTensor等）
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float).unsqueeze(-1)  # 确保标签是float类型


# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.4049, 0.0925, 0.3929], std=[0.4778, 0.5642, 0.4325])
])

val_transform = transforms.Compose([
    # transforms.Resize((48, 48)),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.4049, 0.0925, 0.3929], std=[0.4778, 0.5642, 0.4325])
])

# 定义全局变量
train_data = None


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    # 修改为Kaggle数据集的实际路径（假设用户下载后的结构）
    image_folder_path = r"F:\HJ_Flower\DataSet\chest_xray\train"  # 原数据集的train目录（含NORMAL/PNEUMONIA子目录）
    global train_data
    if train_data is None:
        # 初始化时使用新的Dataset参数（不再需要csv_file）
        train_data = ChestDataset(root_dir=image_folder_path, transform=train_transform)

    # 划分训练集为多个分区(IID划分)
    partition_sizes = [len(train_data) // num_partitions] * num_partitions
    remainder = len(train_data) % num_partitions
    for i in range(remainder):
        partition_sizes[i] += 1

    partitions = random_split(train_data, partition_sizes)

    # 获取当前分区的数据
    partition_data = partitions[partition_id]

    # 将当前分区划分为训练集和验证集(80%训练, 20%验证)
    train_size = int(0.8 * len(partition_data))
    val_size = int(0.2 * len(partition_data))
    t = len(partition_data) - train_size - val_size
    train_data, val_data, _ = random_split(partition_data, [train_size, val_size, t])

    # 创建数据加载器
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,  # 注意: 这里使用验证集作为"测试集"
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader
