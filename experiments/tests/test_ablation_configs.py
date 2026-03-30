"""
测试消融实验配置模块
[INPUT]: 模拟消融配置参数
[OUTPUT]: AblationConfig数据类验证
[POS]: experiments/tests/消融实验配置测试
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from experiments.ablation.configs import (
    AblationConfig,
    ABLATION_CONFIGS,
    get_ablation_config,
    list_available_configs,
)


# =============================================================================
# TDD Phase 1: Tests defined BEFORE implementation
# All tests should FAIL initially (RED phase)
# =============================================================================


class TestAblationConfigDataclass:
    """测试AblationConfig数据类"""

    def test_dataclass_creation(self):
        """测试dataclass创建与默认值"""
        config = AblationConfig()

        assert hasattr(config, 'name')
        assert hasattr(config, 'multi_scale')
        assert hasattr(config, 'gate')
        assert hasattr(config, 'transformer')
        assert hasattr(config, 'cross_attention')

    def test_full_config(self):
        """测试完整配置"""
        config = AblationConfig(
            name='full_hybrid',
            multi_scale=True,
            gate='se',
            transformer=True,
            cross_attention=True,
        )

        assert config.name == 'full_hybrid'
        assert config.multi_scale is True
        assert config.gate == 'se'
        assert config.transformer is True
        assert config.cross_attention is True

    def test_minimal_config(self):
        """测试最小配置 (全部禁用)"""
        config = AblationConfig(
            name='baseline_cnn',
            multi_scale=False,
            gate=None,
            transformer=False,
            cross_attention=False,
        )

        assert config.name == 'baseline_cnn'
        assert config.multi_scale is False
        assert config.gate is None
        assert config.transformer is False
        assert config.cross_attention is False

    def test_immutability(self):
        """测试不可变性 (frozen=True)"""
        config = AblationConfig(
            name='test',
            multi_scale=True,
            gate='se',
            transformer=True,
            cross_attention=True,
        )
        with pytest.raises(AttributeError):
            config.multi_scale = False


class TestAblationConfigs:
    """测试预定义消融配置字典"""

    def test_ablation_configs_exists(self):
        """测试ABLATION_CONFIGS字典存在"""
        assert isinstance(ABLATION_CONFIGS, dict)
        assert len(ABLATION_CONFIGS) >= 5

    def test_required_keys_in_all_configs(self):
        """验证所有配置都有必需的键"""
        required_keys = ['multi_scale', 'gate', 'transformer', 'cross_attention']
        for name, config in ABLATION_CONFIGS.items():
            for key in required_keys:
                assert hasattr(config, key), f"{name} missing {key}"

    def test_baseline_cnn_is_minimal(self):
        """Baseline应该全部禁用 (纯CNN基线)"""
        config = ABLATION_CONFIGS['baseline_cnn']

        assert config.multi_scale is False
        assert config.gate is None
        assert config.transformer is False
        assert config.cross_attention is False

    def test_full_hybrid_has_all(self):
        """Full hybrid应该全部启用"""
        config = ABLATION_CONFIGS['full_hybrid']

        assert config.multi_scale is True
        assert config.gate == 'se'
        assert config.transformer is True
        assert config.cross_attention is True

    def test_no_multi_scale_config(self):
        """测试no_multi_scale配置"""
        config = ABLATION_CONFIGS['no_multi_scale']

        assert config.multi_scale is False
        assert config.gate == 'se'
        assert config.transformer is True
        assert config.cross_attention is True

    def test_no_gate_config(self):
        """测试no_gate配置"""
        config = ABLATION_CONFIGS['no_gate']

        assert config.multi_scale is True
        assert config.gate is None
        assert config.transformer is True
        assert config.cross_attention is True

    def test_no_cross_attention_config(self):
        """测试no_cross_attention配置"""
        config = ABLATION_CONFIGS['no_cross_attention']

        assert config.multi_scale is True
        assert config.gate == 'se'
        assert config.transformer is True
        assert config.cross_attention is False

    def test_no_transformer_config(self):
        """测试no_transformer配置 (纯CNN基线)"""
        config = ABLATION_CONFIGS['no_transformer']

        assert config.multi_scale is True
        assert config.gate == 'se'
        assert config.transformer is False
        assert config.cross_attention is False


class TestHelperFunctions:
    """测试辅助函数"""

    def test_get_ablation_config_valid(self):
        """测试获取有效配置"""
        config = get_ablation_config('baseline_cnn')
        assert isinstance(config, AblationConfig)
        assert config.multi_scale is False

    def test_get_ablation_config_full_hybrid(self):
        """测试获取full_hybrid配置"""
        config = get_ablation_config('full_hybrid')
        assert isinstance(config, AblationConfig)
        assert config.multi_scale is True
        assert config.gate == 'se'

    def test_get_ablation_config_invalid(self):
        """测试获取无效配置"""
        with pytest.raises(KeyError):
            get_ablation_config('nonexistent_config')

    def test_list_available_configs(self):
        """测试列出所有可用配置"""
        configs = list_available_configs()
        assert isinstance(configs, list)
        assert len(configs) >= 5
        assert 'baseline_cnn' in configs
        assert 'full_hybrid' in configs
        assert 'no_transformer' in configs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
