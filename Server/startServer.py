"""
1. 启动服务
2.选择模型
3.选择数据集
4.选择训练参数
5.选择训练方式
6.参数的配置
"""
import flwr as fl
from pathlib import Path
import torch
import Model.ResnetModel as ResnetModel
from Server.Mapper.SetAndGet import FED_CONFIG
from Server.ServerUtils.fitUtils import weighted_average, fit_config
from Server.Strategies import FedAvg, FedTrimmedAvg, FedProx, FedMedian


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建一个模型实例
net = ResnetModel.get_resnet50().to(device)

"""
从文件系统根目录开始的完整路径。在 Windows 系统中，绝对路径通常以盘符（如 C:、D:）开头；
在 Linux 或 macOS 系统中，绝对路径以 / 开头。
"""
Path("result/saved_models").mkdir(exist_ok=True)
Path("result/metrics").mkdir(exist_ok=True)


def start_server():
    # TODO 实现其他聚合策略
    strategy_mapping = {
        "FedAvg": FedAvg,
        "FedTrimmedAvg": FedTrimmedAvg,
        "FedProx": FedProx,
        "FedMedian": FedMedian
    }

    selected_strategy = FED_CONFIG["strategy"]
    strategy_class = strategy_mapping.get(selected_strategy, FedAvg)

    # TODO 清理这些参数是什么
    common_args = {
        "fraction_fit": FED_CONFIG['fraction_fit'],
        "fraction_evaluate": FED_CONFIG['fraction_evaluate'],
        "min_fit_clients": FED_CONFIG['min_fit_clients'],
        "min_evaluate_clients": FED_CONFIG['min_evaluate_clients'],
        "min_available_clients": FED_CONFIG['min_available_clients'],
        "fit_metrics_aggregation_fn": weighted_average,
        "evaluate_metrics_aggregation_fn": weighted_average,
        # TODO 删除了一个配置
        "on_fit_config_fn": fit_config
    }

    if selected_strategy == "FedTrimmedAvg":
        strategy1 = strategy_class.FedTrimmedAvg(
            **common_args
        )
    elif selected_strategy == "FedProx":
        strategy1 = strategy_class.FedProx(
            proximal_mu=FED_CONFIG['proximal_mu'],
            **common_args
        )
    else:
        strategy1 = strategy_class.FedAvg(**common_args)

        #server_address config strategy
        #就是flwr服务器的IP地址，用于与客户端进行通信。

    fl.server.start_server(
        server_address=FED_CONFIG['server_address'],
        config=fl.server.ServerConfig(num_rounds=FED_CONFIG['num_epochs']),
        strategy=strategy1
    )


if __name__ == "__main__":
    start_server()
