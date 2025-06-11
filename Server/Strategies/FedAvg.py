import flwr as fl

from Server.ServerUtils.saveUtils import save_model, save_results

# 定义策略名称
clazz = "FedAvg"


class FedAvg(fl.server.strategy.FedAvg):
    """自定义FedAvg策略, 包含模型保存和指标跟踪"""

    def __init__(self, *args, **kwargs):
        """初始化方法，调用父类的初始化"""
        super().__init__(*args, **kwargs)

    # 可选：自定义全局模型参数初始化（这里注释掉，使用默认初始化）
    # def initialize_parameters(self, client_manager):
    #     model = get_resnet50()  # 假设你有一个函数返回初始模型
    #     params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    #     return fl.common.ndarrays_to_parameters(params)

    def aggregate_fit(self, server_round, results, failures):
        """聚合每一轮的训练结果"""
        # 定义前缀，用于区分训练和评估的保存文件
        prefix = "Training"

        # （可选）保存每个客户端的参数，用于调试或分析
        # save_results_by_client(server_round, results, prefix, clazz)

        # 调用父类的aggregate_fit方法，执行FedAvg聚合
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # 保存聚合后的模型参数和指标
        if aggregated_parameters is not None:  # 确保聚合成功
            save_model(aggregated_parameters, server_round)
            save_results(server_round, metrics, prefix, clazz)

        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """聚合每一轮的评估结果"""
        prefix = "Validation"

        # （可选）保存每个客户端的评估结果
        # save_results_by_client(server_round, results, prefix, clazz)

        # 调用父类的aggregate_evaluate方法，聚合评估指标
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # 保存聚合后的评估指标
        if loss_aggregated is not None:  # 确保聚合成功
            save_results(server_round, metrics_aggregated, prefix, clazz)

        return loss_aggregated, metrics_aggregated