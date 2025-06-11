"""
ClientConfig.py 是「客户端核心」：定义了客户端的具体行为（如何训练、评估、同步参数）。
"""
import torchvision
import torch.nn as nn
import torch
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient

from Utils.DatasetUtil import load_data
from Utils.DeeplearningUtils import train, validate


class FlowerClient(NumPyClient):
    def __init__(self, train_load, val_load, local_epochs, learning_rate):
        self.model = self._get_model()
        self.train_load = train_load
        self.val_load = val_load
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.local_epochs = local_epochs

    def _get_model(self) -> torch.nn.Module:
        """初始化ResNet50模型"""
        model = torchvision.models.resnet50(weights=None)  # 使用pretrained=False避免自动下载
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 1)
        # model = Model()
        return model

    def fit(
            self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self.set_parameters(parameters)
        all_epoch_results = train(self.model, self.train_load, self.local_epochs, self.lr, self.device)
        # 取最后一个 epoch 的指标（本地训练完成后的最终结果）
        final_results = all_epoch_results[-1]  # 关键修改：列表转单个字典
        return self.get_parameters(config), len(self.train_load.dataset), final_results

    def evaluate(
            self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.set_parameters(self.model)
        loss, results = validate(self.model, self.val_load, self.device)
        return loss, len(self.val_load.dataset), results

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if isinstance(parameters, list):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.load_state_dict(parameters.state_dict())


# TODO 增加指定数据集的功能
# TODO 这个num_partitions是什么 和本身的数据分区有没有区别哦
def client_fn(partition_id):
    num_partitions = 3
    batch_size = 32
    train_loader, val_loader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = 1
    learning_rate = 0.01

    # Return Client instance
    return FlowerClient(train_loader, val_loader, local_epochs, learning_rate).to_client()


"""
加载参数：调用 set_parameters 把服务器的全局参数加载到本地模型。
本地训练：用自己的训练数据（self.train_load）训练模型（train 函数是具体训练逻辑）。
local_epochs=1：只训练1轮（本地训练次数）。
lr=0.01：学习率（控制参数更新的速度）。
返回结果：
self.get_parameters(config)：提取更新后的模型参数（要上传给服务器）。
len(self.train_load.dataset)：本地训练数据的总量（服务器用于计算加权平均）。
results：训练指标（如损失值、准确率等）。
"""
