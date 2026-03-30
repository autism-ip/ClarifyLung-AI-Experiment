# =============================================================================
# TDD Phase 1: RED - 失败的测试用例
# =============================================================================
"""
[INPUT]: torch, numpy, typing
[OUTPUT]: AblationStudy
[POS]: experiments/ablation/ Ablation study framework
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any


# -----------------------------------------------------------------------------
# Test: AblationStudy Initialization
# -----------------------------------------------------------------------------
def test_ablation_study_init():
    """AblationStudy should initialize with model class and configs"""
    from experiments.ablation.ablator import AblationStudy
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self, multi_scale=True, gate='se', transformer=True, cross_attention=True):
            super().__init__()
            self.multi_scale = multi_scale
            self.gate = gate
            self.transformer = transformer
            self.cross_attention = cross_attention

        def forward(self, x):
            return x.mean(dim=-1)

    study = AblationStudy(model_class=DummyModel)
    assert study.model_class == DummyModel
    # ABLATION_CONFIGS has 6 configs
    assert len(study.configs) == 6


# -----------------------------------------------------------------------------
# Test: Ablation Study Has Correct Configurations
# -----------------------------------------------------------------------------
def test_ablation_configs_count():
    """AblationStudy should have 6 ablation configurations"""
    from experiments.ablation.ablator import AblationStudy
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self, multi_scale=True, gate='se', transformer=True, cross_attention=True):
            super().__init__()
            self.multi_scale = multi_scale
            self.gate = gate
            self.transformer = transformer
            self.cross_attention = cross_attention

        def forward(self, x):
            return x.mean(dim=-1)

    study = AblationStudy(model_class=DummyModel)

    # Expected configs from ABLATION_CONFIGS
    expected_configs = [
        'baseline_cnn',
        'no_multi_scale',
        'no_gate',
        'no_cross_attention',
        'no_transformer',
        'full_hybrid'
    ]

    config_names = [c.name for c in study.configs]
    for expected in expected_configs:
        assert expected in config_names, f"Missing config: {expected}"


# -----------------------------------------------------------------------------
# Test: Compute Summary Has Required Columns
# -----------------------------------------------------------------------------
def test_compute_summary_has_required_columns():
    """compute_summary should return dict with metrics"""
    from experiments.ablation.ablator import AblationStudy
    import numpy as np

    # Mock results
    results = {
        'baseline_cnn': {'accuracy': 0.80, 'macro_f1': 0.79, 'auc_roc': 0.88},
        'full_hybrid': {'accuracy': 0.85, 'macro_f1': 0.84, 'auc_roc': 0.92},
    }

    summary = AblationStudy.compute_summary(results)

    assert 'baseline_cnn' in summary
    assert 'full_hybrid' in summary
    assert 'improvement' in summary['full_hybrid']


# -----------------------------------------------------------------------------
# Test: Compute Summary Includes Improvement Metrics
# -----------------------------------------------------------------------------
def test_compute_summary_includes_improvement():
    """compute_summary should compute improvement over baseline"""
    from experiments.ablation.ablator import AblationStudy

    results = {
        'baseline_cnn': {'accuracy': 0.80, 'macro_f1': 0.79},
        'full_hybrid': {'accuracy': 0.85, 'macro_f1': 0.84},
    }

    summary = AblationStudy.compute_summary(results)

    # Full hybrid should have improvement over baseline
    assert summary['full_hybrid']['improvement']['accuracy'] == pytest.approx(0.05, abs=0.01)
    assert summary['full_hybrid']['improvement']['macro_f1'] == pytest.approx(0.05, abs=0.01)


# -----------------------------------------------------------------------------
# Test: Run Ablation Returns Results Dict
# -----------------------------------------------------------------------------
def test_run_ablation_returns_dict():
    """run_ablation should return dictionary of results"""
    from experiments.ablation.ablator import AblationStudy
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class DummyModel(nn.Module):
        def __init__(self, multi_scale=True, gate='se', transformer=True, cross_attention=True):
            super().__init__()
            self.multi_scale = multi_scale
            self.gate = gate
            self.transformer = transformer
            self.cross_attention = cross_attention
            self.fc = nn.Linear(100, 3)

        def forward(self, x):
            return self.fc(x.mean(dim=1))

    # Create dummy dataset
    X = torch.randn(20, 100)
    y = torch.randint(0, 3, (20,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=5)

    study = AblationStudy(model_class=DummyModel)

    # Mock train and evaluate functions since we don't have real training setup
    def mock_train(model, train_loader, epochs=1):
        return {'train_loss': 0.5}

    def mock_evaluate(model, val_loader):
        return {'accuracy': 0.8, 'macro_f1': 0.78, 'auc_roc': 0.87}

    results = study.run_ablation(
        train_loader=dataloader,
        val_loader=dataloader,
        train_fn=mock_train,
        evaluate_fn=mock_evaluate,
        epochs=1
    )

    assert isinstance(results, dict)
    assert len(results) == 6  # 6 ablation experiments


# -----------------------------------------------------------------------------
# Test: Single Ablation Config Run
# -----------------------------------------------------------------------------
def test_run_single_config():
    """run_single_config should train and evaluate one configuration"""
    from experiments.ablation.ablator import AblationStudy
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class DummyModel(nn.Module):
        def __init__(self, multi_scale=True, gate='se', transformer=True, cross_attention=True):
            super().__init__()
            self.multi_scale = multi_scale
            self.gate = gate
            self.transformer = transformer
            self.cross_attention = cross_attention
            self.fc = nn.Linear(100, 3)

        def forward(self, x):
            return self.fc(x.mean(dim=1))

    X = torch.randn(20, 100)
    y = torch.randint(0, 3, (20,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=5)

    study = AblationStudy(model_class=DummyModel)
    config = study.configs[0]  # baseline_cnn

    def mock_train(model, train_loader, epochs=1):
        return {'train_loss': 0.5}

    def mock_evaluate(model, val_loader):
        return {'accuracy': 0.8, 'macro_f1': 0.78, 'auc_roc': 0.87}

    result = study.run_single_config(
        config=config,
        train_loader=dataloader,
        val_loader=dataloader,
        train_fn=mock_train,
        evaluate_fn=mock_evaluate,
        epochs=1
    )

    assert isinstance(result, dict)
    assert 'accuracy' in result
    assert 'macro_f1' in result
    assert 'auc_roc' in result


# -----------------------------------------------------------------------------
# Test: Get Config By Name
# -----------------------------------------------------------------------------
def test_get_config_by_name():
    """get_config should return the correct configuration"""
    from experiments.ablation.ablator import AblationStudy
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self, multi_scale=True, gate='se', transformer=True, cross_attention=True):
            super().__init__()

        def forward(self, x):
            return x.mean(dim=-1)

    study = AblationStudy(model_class=DummyModel)

    config = study.get_config('baseline_cnn')
    assert config is not None
    assert config.name == 'baseline_cnn'
    assert config.multi_scale is False

    # Test non-existent config
    assert study.get_config('non_existent') is None