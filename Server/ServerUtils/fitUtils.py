#TODO 调研一下这个工具类是什么哦
import re
from typing import List, Tuple, Dict

import numpy as np
from flwr.common import Scalar


# TODO 了解一下re的一些函数
def parse_matrix_string(matrix_str: str) -> np.ndarray:
    cleaned = re.sub(r"[^\d.\-\s]", " ", matrix_str)
    # 提取所有数字（包括浮点数和负数）
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", cleaned)
    # 转换为 int 并 reshape
    return np.array([int(x) for x in numbers]).reshape(2, 2)


"""
功能概述
这个函数用于计算一组指标的加权平均值。在联邦学习场景里，不同客户端可能会提供带有各自样本数量和指标的信息，此函数会根据样本数量对这些指标进行加权计算，得出最终的平均指标。

详细步骤
计算总样本数：把所有客户端提供的样本数量相加，得到总的样本数。
初始化加权指标字典：创建一个字典，里面包含 accuracy（准确率）、loss（损失值）、recall（召回率）、precision（精确率）、F1（F1 分数）和 matrix（混淆矩阵）这些指标，并将它们初始化为 0 或者空矩阵。
计算加权和：遍历每个客户端提供的样本数量和指标，对于 accuracy、loss、recall、precision、F1 这些指标，用样本数量乘以指标值，累加到对应的加权指标中；对于混淆矩阵，先把字符串解析成矩阵，再累加到加权指标的矩阵里。
计算加权平均值：用每个指标的加权和除以总样本数，得到加权平均值。
处理混淆矩阵格式：把最终的混淆矩阵转换为字符串格式。
返回结果：返回包含所有加权平均指标的字典。
"""


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_metrics = {
        'accuracy': 0.0,
        'loss': 0.0,
        'recall': 0.0,
        'precision': 0.0,
        'F1': 0.0,
        'matrix': np.zeros((2, 2))  # 初始化混淆矩阵（数值格式）
    }

    for num_examples, metric in metrics:
        # 计算常规指标的加权和
        for key in ['accuracy', 'loss', 'recall', 'precision', 'F1']:
            if key in metric:
                weighted_metrics[key] += num_examples * metric[key]

        # 解析混淆矩阵字符串并加权求和
        if 'matrix' in metric:
            try:
                matrix = parse_matrix_string(metric['matrix'])  # 安全解析
                weighted_metrics['matrix'] += matrix
            except Exception as e:
                print(f"解析矩阵失败: {e}")
                continue

    # 计算加权平均值
    for key in ['accuracy', 'loss', 'recall', 'precision', 'F1']:
        if key in weighted_metrics:
            weighted_metrics[key] /= total_examples

    # 将混淆矩阵转换为字符串格式（保持原格式）
    weighted_metrics['matrix'] = str(weighted_metrics['matrix'].tolist())

    return weighted_metrics


"""
这个函数的作用是返回训练配置信息。在联邦学习中，服务器需要给客户端发送训练相关的配置参数，此函数会从配置文件里获取训练轮数，将其作为配置信息返回。

详细步骤
导入配置：从 Config.config 模块里导入 FEDERATED_CONFIG 配置字典。
返回配置信息：创建一个字典，里面包含 num_epochs（训练轮数）这个配置项，其值从 FEDERATED_CONFIG 中获取，最后返回这个字典。
"""


def fit_config(round: int) -> Dict[str, Scalar]:
    from Server.Mapper.SetAndGet import FED_CONFIG
    return {
        'num_epochs': FED_CONFIG['num_epochs']
    }
