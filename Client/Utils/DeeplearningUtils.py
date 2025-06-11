import torch
import torch.nn as nn
from tqdm import tqdm
from Client.Utils.BinaryMatrixUtil import Matrix


def _compute_metrics(matrix_obj):
    """公共指标计算函数，减少代码重复"""
    return {
        "loss": None,  # 由调用方填充
        "accuracy": matrix_obj.get_accuracy(),
        "precision": matrix_obj.get_precision(),
        "recall": matrix_obj.get_recall(),
        "F1": matrix_obj.get_f1(),
        "matrix": matrix_obj.get_string_matrix()
    }


def train(
        model,
        train_loader,
        epochs,
        learning_rate,
        device,
        criterion=None,  # 可自定义损失函数
        optimizer_cls=torch.optim.SGD  # TODO 可自定义优化器类型
):
    model.to(device)
    # 初始化损失函数（默认BCEWithLogitsLoss）
    criterion = criterion or nn.BCEWithLogitsLoss().to(device)
    # 初始化优化器（支持自定义优化器）
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    model.train()
    total_results = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        # 初始化混淆矩阵（跨batch累加）
        epoch_matrix = torch.zeros(2, 2, dtype=torch.int64)

        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播 + 损失计算
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加混淆矩阵（跨batch统计）
            batch_matrix = Matrix(outputs, labels, epoch_matrix)
            batch_matrix.confusion_matrix()

            # 更新进度条（计算当前平均loss）
            avg_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix({"loss": avg_loss, "acc": batch_matrix.get_accuracy()})

        # 计算epoch级指标
        epoch_avg_loss = epoch_loss / len(train_loader)
        metrics = _compute_metrics(batch_matrix)
        metrics["loss"] = epoch_avg_loss  # 填充loss

        total_results.append(metrics)
        print(f"Train Epoch {epoch + 1}: Loss={epoch_avg_loss:.4f}, Acc={metrics['accuracy']:.2f}%")

    return total_results  # 返回每个epoch的结果


def validate(
        model,
        test_loader,
        device,
        criterion=None  # 可自定义损失函数
):
    model.to(device)
    criterion = criterion or nn.BCEWithLogitsLoss().to(device)  # 修复设备一致性
    model.eval()

    total_loss = 0.0
    total_matrix = torch.zeros(2, 2, dtype=torch.int64)

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Validation")
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 累加混淆矩阵（跨batch统计）
            batch_matrix = Matrix(outputs, labels, total_matrix)
            batch_matrix.confusion_matrix()

            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({"loss": avg_loss, "acc": batch_matrix.get_accuracy()})

    # 计算最终指标
    avg_loss = total_loss / len(test_loader)
    metrics = _compute_metrics(batch_matrix)
    metrics["loss"] = avg_loss

    print(f"Validation: Loss={avg_loss:.4f}, Acc={metrics['accuracy']:.2f}%")
    return avg_loss, metrics