# =============================================================================
# Ablation Study Runner Module
# =============================================================================
"""
[INPUT]: torch, typing, numpy
[OUTPUT]: AblationStudy
[POS]: experiments/ablation/ Ablation study runner
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Callable, Optional
import numpy as np

from .configs import AblationConfig, ABLATION_CONFIGS, get_ablation_config, list_available_configs


# =============================================================================
# Ablation Study
# =============================================================================
class AblationStudy:
    """
    消融研究框架类
    用于系统性地评估模型各组件的贡献度

    [功能]:
    - 定义标准消融配置组合
    - 支持自定义配置扩展
    - 自动计算各组件的相对提升

    [消融组件]:
    1. baseline_cnn: 仅CNN特征提取器
    2. no_multi_scale: 移除多尺度特征融合
    3. no_gate: 移除门控机制
    4. no_cross_attention: 移除交叉注意力
    5. no_transformer: 移除Transformer模块
    6. full_hybrid: 完整混合架构
    """

    def __init__(
        self,
        model_class: type,
        configs: Optional[List[AblationConfig]] = None
    ):
        """
        初始化消融研究

        Args:
            model_class: 模型类，必须接受 multi_scale, gate, transformer, cross_attention 参数
            configs: 自定义配置列表，None则使用ABLATION_CONFIGS
        """
        self.model_class = model_class
        if configs is not None:
            self.configs = configs
        else:
            # 使用预定义配置
            self.configs = [cfg for cfg in ABLATION_CONFIGS.values()]

    def run_single_config(
        self,
        config: AblationConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_fn: Callable,
        evaluate_fn: Callable,
        epochs: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Dict[str, float]:
        """
        运行单个消融配置实验

        Args:
            config: 消融配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            train_fn: 训练函数签名 (model, train_loader, epochs) -> train_history
            evaluate_fn: 评估函数签名 (model, val_loader) -> metrics_dict
            epochs: 训练轮数
            device: 设备

        Returns:
            评估指标字典
        """
        # 实例化模型
        model = self.model_class(
            multi_scale=config.multi_scale,
            gate=config.gate,
            transformer=config.transformer,
            cross_attention=config.cross_attention
        )
        model = model.to(device)

        # 训练
        train_fn(model, train_loader, epochs=epochs)

        # 评估
        metrics = evaluate_fn(model, val_loader)

        # 添加配置信息
        metrics['config'] = config.name

        return metrics

    def run_ablation(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_fn: Callable,
        evaluate_fn: Callable,
        epochs: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Dict[str, Dict[str, float]]:
        """
        运行完整消融研究

        [流程]:
        1. 按顺序运行每个配置
        2. 记录每个配置的评估指标
        3. 返回配置名称到指标的映射

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            train_fn: 训练函数
            evaluate_fn: 评估函数
            epochs: 每配置训练轮数
            device: 设备

        Returns:
            配置名称 -> 指标字典 的映射
        """
        results = {}

        for config in self.configs:
            print(f"Running ablation config: {config.name}")

            metrics = self.run_single_config(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                train_fn=train_fn,
                evaluate_fn=evaluate_fn,
                epochs=epochs,
                device=device
            )

            results[config.name] = metrics

        return results

    @staticmethod
    def compute_summary(
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        计算消融研究摘要

        [功能]:
        - 计算各配置相对baseline的提升
        - 计算平均指标和标准差
        - 生成可读性强的摘要报告

        Args:
            results: run_ablation返回的结果字典

        Returns:
            摘要字典，包含各配置详情和改进幅度
        """
        summary = {}

        # 获取baseline指标
        baseline = results.get('baseline_cnn', None)
        if baseline is None:
            # 如果没有baseline，使用第一个结果
            keys = list(results.keys())
            baseline = results[keys[0]]

        baseline_acc = baseline.get('accuracy', 0)
        baseline_f1 = baseline.get('macro_f1', 0)
        baseline_auc = baseline.get('auc_roc', 0)

        # 计算每个配置的改进
        for config_name, metrics in results.items():
            config_summary = {
                'accuracy': metrics.get('accuracy', 0),
                'macro_f1': metrics.get('macro_f1', 0),
                'auc_roc': metrics.get('auc_roc', 0),
                'improvement': {
                    'accuracy': metrics.get('accuracy', 0) - baseline_acc,
                    'macro_f1': metrics.get('macro_f1', 0) - baseline_f1,
                    'auc_roc': metrics.get('auc_roc', 0) - baseline_auc,
                }
            }
            summary[config_name] = config_summary

        # 添加汇总统计
        all_accs = [m.get('accuracy', 0) for m in results.values()]
        summary['_stats'] = {
            'mean_accuracy': np.mean(all_accs),
            'std_accuracy': np.std(all_accs),
            'best_config': max(results.keys(), key=lambda k: results[k].get('accuracy', 0))
        }

        return summary

    def add_config(self, config: AblationConfig) -> None:
        """
        添加自定义消融配置

        Args:
            config: 消融配置
        """
        # 避免重复名称
        existing_names = [c.name for c in self.configs]
        if config.name not in existing_names:
            self.configs.append(config)

    def get_config(self, name: str) -> Optional[AblationConfig]:
        """
        按名称获取配置

        Args:
            name: 配置名称

        Returns:
            配置对象，不存在则返回None
        """
        for config in self.configs:
            if config.name == name:
                return config
        return None