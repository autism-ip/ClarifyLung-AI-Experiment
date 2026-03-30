"""
[INPUT]: torch, ptflops
[OUTPUT]: ModelComplexityAnalyzer
[POS]: experiments/ 模型复杂度分析
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import time


# =============================================================================
# 函数: count_parameters
# =============================================================================


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数量

    Args:
        model: PyTorch 模型

    Returns:
        包含 total 和 trainable 数量的字典

    Example:
        >>> model = nn.Linear(100, 10)
        >>> count_parameters(model)
        {'total': 1010, 'trainable': 1010}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# =============================================================================
# 函数: measure_latency
# =============================================================================


def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    测量模型推理延迟

    Args:
        model: PyTorch 模型
        input_size: 输入尺寸，如 (1, 3, 224, 224)
        num_iterations: 测量迭代次数
        warmup_iterations: 预热迭代次数
        device: 计算设备

    Returns:
        包含 mean_ms, std_ms, min_ms, max_ms 的字典

    Example:
        >>> model = nn.Linear(100, 10)
        >>> measure_latency(model, (1, 100), num_iterations=50)
        {'mean_ms': 0.123, 'std_ms': 0.045, 'min_ms': 0.089, 'max_ms': 0.234}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # 创建输入
    dummy_input = torch.randn(input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    # 测量
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
            times.append(elapsed)

    times_tensor = torch.tensor(times)
    return {
        "mean_ms": times_tensor.mean().item(),
        "std_ms": times_tensor.std().item(),
        "min_ms": times_tensor.min().item(),
        "max_ms": times_tensor.max().item(),
    }


# =============================================================================
# 函数: calculate_flops (使用 ptflops)
# =============================================================================


def calculate_flops(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, float]:
    """
    计算模型 FLOPs (使用 ptflops)

    Args:
        model: PyTorch 模型
        input_size: 输入尺寸，如 (1, 3, 224, 224)

    Returns:
        包含 flops 和 macs 的字典

    Note:
        需要安装 ptflops: pip install ptflops
    """
    try:
        import ptflops

        model.eval()
        macs, params = ptflops.get_model_complexity_info(
            model,
            input_size,
            as_strings=False,
            verbose=False,
            print_per_layer_stat=False,
        )
        # FLOPs = 2 * MACs (乘法加法)
        flops = macs * 2
        return {"flops": flops, "macs": macs, "params": params}
    except ImportError:
        return {"flops": 0, "macs": 0, "params": 0}


# =============================================================================
# 函数: analyze_model_complexity
# =============================================================================


def analyze_model_complexity(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    num_latency_iterations: int = 100,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    综合分析模型复杂度

    Args:
        model: PyTorch 模型
        input_size: 输入尺寸
        num_latency_iterations: 延迟测量迭代次数
        device: 计算设备

    Returns:
        包含 parameters, latency, flops 的完整分析字典

    Example:
        >>> model = nn.Linear(100, 10)
        >>> result = analyze_model_complexity(model, (1, 100))
        >>> print(result['parameters']['total'])
        1010
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # 参数量
    parameters = count_parameters(model)

    # 延迟
    latency = measure_latency(
        model,
        input_size,
        num_iterations=num_latency_iterations,
        device=device,
    )

    # FLOPs
    flops_info = calculate_flops(model, input_size)

    return {
        "parameters": parameters,
        "latency": latency,
        "flops": flops_info.get("flops", 0),
        "macs": flops_info.get("macs", 0),
        "model_size_mb": parameters["total"] * 4 / (1024**2),  # float32 = 4 bytes
    }


# =============================================================================
# 函数: generate_complexity_table
# =============================================================================


def generate_complexity_table(results: list) -> str:
    """
    生成模型复杂度对比 Markdown 表格

    Args:
        results: 复杂度分析结果列表，每个元素包含:
            - model_name: str
            - parameters: Dict with 'total' and 'trainable'
            - flops: float
            - latency: Dict with 'mean_ms'

    Returns:
        Markdown 表格字符串

    Example:
        >>> results = [{
        ...     'model_name': 'ResNet50',
        ...     'parameters': {'total': 25e6, 'trainable': 25e6},
        ...     'flops': 4e9,
        ...     'latency': {'mean_ms': 10.5}
        ... }]
        >>> print(generate_complexity_table(results))
    """
    if not results:
        return "No results to display."

    # 表头
    headers = ["Model", "Params (M)", "FLOPs (G)", "Latency (ms)", "Size (MB)"]
    separator = ["---"] * len(headers)

    # 数据行
    rows = []
    for r in results:
        params_m = r.get("parameters", {}).get("total", 0) / 1e6
        flops_g = r.get("flops", 0) / 1e9
        latency = r.get("latency", {}).get("mean_ms", 0)
        size_mb = r.get("model_size_mb", 0)

        row = [
            r.get("model_name", "Unknown"),
            f"{params_m:.2f}",
            f"{flops_g:.2f}",
            f"{latency:.2f}",
            f"{size_mb:.2f}",
        ]
        rows.append("| " + " | ".join(row) + " |")

    # 构建表格
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(separator) + " |"

    return "\n".join([header_line, sep_line] + rows)


# =============================================================================
# 类: ModelComplexityAnalyzer (可选的高级封装)
# =============================================================================


class ModelComplexityAnalyzer:
    """
    模型复杂度分析器 (封装常用功能)

    用法:
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.analyze(my_model, input_size=(1, 3, 224, 224))
        print(analyzer.generate_report(result))
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def analyze(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """分析模型复杂度"""
        return analyze_model_complexity(model, input_size, device=self.device)

    def generate_report(self, result: Dict[str, Any]) -> str:
        """生成分析报告"""
        params_m = result["parameters"]["total"] / 1e6
        flops_g = result["flops"] / 1e9
        latency = result["latency"]["mean_ms"]
        size_mb = result["model_size_mb"]

        lines = [
            "=" * 50,
            "Model Complexity Report",
            "=" * 50,
            f"Parameters: {params_m:.2f}M",
            f"FLOPs: {flops_g:.2f}G",
            f"Latency: {latency:.2f}ms",
            f"Model Size: {size_mb:.2f}MB",
            "=" * 50,
        ]
        return "\n".join(lines)
