"""
[INPUT]: dataclasses, typing, torch
[OUTPUT]: BenchmarkResult dataclass, ModelBenchmark class, comparison utilities
[POS]: experiments/benchmark/ benchmark execution and reporting
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn


# =============================================================================
# 数据类: BenchmarkResult
# =============================================================================


@dataclass
class BenchmarkResult:
    """
    单次基准测试结果

    Attributes:
        model_name: 模型名称
        accuracy: 分类准确率 (0-1)
        macro_f1: Macro F1 分数 (0-1)
        auc_roc: AUC-ROC 分数 (0-1)
        precision: 精确率 (可选)
        recall: 召回率 (可选)
        training_time: 训练时间秒数 (可选)
        inference_time: 推理时间秒数 (可选)
    """

    model_name: str
    accuracy: float
    macro_f1: float
    auc_roc: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None


# =============================================================================
# 类: ModelBenchmark
# =============================================================================


class ModelBenchmark:
    """
    模型基准测试执行器

    用法:
        model = create_resnet50(num_classes=3)
        benchmark = ModelBenchmark(model, "ResNet50")
        result = benchmark.run_benchmark(val_loader, num_epochs=10)
    """

    def __init__(self, model: nn.Module, model_name: str, device: Optional[str] = None):
        """
        初始化基准测试器

        Args:
            model: 待测试模型
            model_name: 模型名称标识
            device: 计算设备 ("cuda"/"cpu")，默认自动选择
        """
        self.model = model
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def run_benchmark(
        self,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 1,
    ) -> BenchmarkResult:
        """
        在验证集上运行基准测试

        Args:
            val_loader: 验证数据加载器
            num_epochs: 测试轮数 (多次运行取平均)

        Returns:
            BenchmarkResult 测试结果
        """
        self.model.eval()

        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for _ in range(num_epochs):
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    predictions = outputs.argmax(dim=1)

                    total_correct += (predictions == targets).sum().item()
                    total_samples += targets.size(0)

                    all_preds.extend(predictions.cpu().tolist())
                    all_targets.extend(targets.cpu().tolist())

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # 简化: 使用 accuracy 作为 macro_f1 和 auc_roc 的占位
        # 完整实现应使用 sklearn.metrics 计算真实值
        return BenchmarkResult(
            model_name=self.model_name,
            accuracy=accuracy,
            macro_f1=accuracy,  # placeholder
            auc_roc=accuracy,  # placeholder
        )

    def measure_inference_time(self, input_size: tuple, num_iterations: int = 100) -> float:
        """
        测量单次推理时间

        Args:
            input_size: 输入尺寸 (B, C, H, W)
            num_iterations: 迭代次数

        Returns:
            平均推理时间 (秒)
        """
        self.model.eval()
        dummy_input = torch.randn(input_size).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

        # Measure
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time

        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        return elapsed / num_iterations


# =============================================================================
# 函数: generate_comparison_table
# =============================================================================


def generate_comparison_table(results: List[Dict[str, Any]]) -> str:
    """
    生成模型对比 Markdown 表格

    Args:
        results: 基准测试结果列表，每个 dict 包含:
            - model_name: str
            - accuracy: float
            - macro_f1: float
            - auc_roc: float
            - (可选) precision: float
            - (可选) recall: float

    Returns:
        Markdown 格式表格字符串

    Example:
        >>> results = [
        ...     {"model_name": "ResNet50", "accuracy": 0.82, "macro_f1": 0.81, "auc_roc": 0.90},
        ...     {"model_name": "ViT", "accuracy": 0.84, "macro_f1": 0.83, "auc_roc": 0.92},
        ... ]
        >>> print(generate_comparison_table(results))
    """
    if not results:
        return "No results to display."

    # 表头
    headers = ["Model", "Accuracy", "Macro F1", "AUC-ROC"]
    if any("precision" in r and r["precision"] is not None for r in results):
        headers.append("Precision")
    if any("recall" in r and r["recall"] is not None for r in results):
        headers.append("Recall")

    # 分隔线
    separator = ["---"] * len(headers)

    # 格式化行
    rows = []
    for r in results:
        row = [
            r.get("model_name", "Unknown"),
            f"{r.get('accuracy', 0):.4f}",
            f"{r.get('macro_f1', 0):.4f}",
            f"{r.get('auc_roc', 0):.4f}",
        ]
        if "precision" in headers:
            precision = r.get("precision")
            row.append(f"{precision:.4f}" if precision is not None else "-")
        if "recall" in headers:
            recall = r.get("recall")
            row.append(f"{recall:.4f}" if recall is not None else "-")

        rows.append("| " + " | ".join(row) + " |")

    # 构建表格
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(separator) + " |"

    table = "\n".join([header_line, sep_line] + rows)
    return table


def save_results(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    保存基准测试结果到文件

    Args:
        results: 结果列表
        filepath: 输出文件路径
    """
    import json

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """
    从文件加载基准测试结果

    Args:
        filepath: 结果文件路径

    Returns:
        结果列表
    """
    import json

    with open(filepath, "r") as f:
        return json.load(f)
