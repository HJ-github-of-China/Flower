"""
这个文件里的 Matrix 类是一个 分类任务评估工具，专门用于计算和分析模型的预测效果。它通过「混淆矩阵」这个核心工具，能算出准确率、精确率、召回率、F1分数等常用的分类指标。
2.混淆矩阵是机器学习中评估分类模型性能的核心工具，尤其适用于监督学习的分类任务（如二分类、多分类）。它通过表格形式，直观展示模型预测结果与真实标签的匹配情况，帮助分析模型在不同类别上的“混淆”（错误预测）模式。

"""
import torch
import numpy as np


class Matrix:

    # 初始化方法，接收模型输出、真实标签、可选的混淆矩阵和类别数
    def __init__(self, outputs, labels, matrix=None, num_classes=2):
        self.outputs = outputs.squeeze()  # 模型输出（去掉多余维度）
        self.labels = labels  # 真实标签
        self.num_classes = num_classes  # 类别数（默认2类，比如「患病/健康」）
        # 初始化混淆矩阵（如果没传，就创建全零矩阵）
        if matrix is None:
            self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        else:
            self.matrix = matrix

    def confusion_matrix(self):
        # 二分类问题，使用 sigmoid 后判断阈值
        if self.num_classes == 2:
            outputs = (torch.sigmoid(self.outputs) > 0.5).float()
            for p, t in zip(outputs, self.labels):
                self.matrix[int(t.item()), int(p.item())] += 1  # 修正索引顺序
        else:
            # 多分类问题，使用 argmax 获取预测类别
            _, predicted = torch.max(self.outputs, 1)
            for p, t in zip(predicted, self.labels):
                self.matrix[int(t.item()), int(p.item())] += 1

    def get_matrix(self):
        return self.matrix

    def get_accuracy(self):
        """计算整体准确率"""
        matrix = self.matrix.cpu().numpy()
        return matrix.diagonal().sum() / matrix.sum() * 100 if matrix.sum() > 0 else 0

    def get_precision(self, class_idx=None):
        """计算精确率
        如果指定class_idx，则返回该类别的精确率
        否则返回所有类别的平均精确率
        """
        matrix = self.matrix.cpu().numpy()
        if class_idx is not None:
            tp = matrix[class_idx, class_idx]
            fp = matrix[:, class_idx].sum() - tp
            return tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

        # 计算所有类别的平均精确率
        precisions = []
        for i in range(self.num_classes):
            tp = matrix[i, i]
            fp = matrix[:, i].sum() - tp
            precisions.append(tp / (tp + fp) * 100 if (tp + fp) > 0 else 0)
        return np.mean(precisions)

    def get_recall(self, class_idx=None):
        """计算召回率
        如果指定class_idx，则返回该类别的召回率
        否则返回所有类别的平均召回率
        """
        matrix = self.matrix.cpu().numpy()
        if class_idx is not None:
            tp = matrix[class_idx, class_idx]
            fn = matrix[class_idx, :].sum() - tp
            return tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

        # 计算所有类别的平均召回率
        recalls = []
        for i in range(self.num_classes):
            tp = matrix[i, i]
            fn = matrix[i, :].sum() - tp
            recalls.append(tp / (tp + fn) * 100 if (tp + fn) > 0 else 0)
        return np.mean(recalls)

    def get_f1(self, class_idx=None):
        """计算F1分数
        如果指定class_idx，则返回该类别的F1分数
        否则返回所有类别的平均F1分数
        """
        p = self.get_precision(class_idx) / 100
        r = self.get_recall(class_idx) / 100
        return 2 * p * r / (p + r) * 100 if (p + r) > 0 else 0

    def get_string_matrix(self):
        matrix = np.array(self.matrix.cpu())
        return str(matrix)