import torchvision.models
import torch.nn as nn

"""
torchvision：这是 PyTorch 的一个子库，提供了许多计算机视觉任务常用的工具，比如预训练的模型、常用的数据集（像 MNIST、CIFAR - 10 等），还有图像变换函数（如裁剪、旋转、归一化等）。
nn：torch.nn 是 PyTorch 中用于构建神经网络的模块，包含了各种神经网络层（如全连接层、卷积层等）和损失函数。
"""


def get_resnet50():
    # 不要预训练的参数
    model = torchvision.models.resnet50(weights=None)
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 1)
    return model
