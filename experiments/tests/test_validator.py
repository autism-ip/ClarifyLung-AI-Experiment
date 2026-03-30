"""
[INPUT]: numpy, typing
[OUTPUT]: KFoldCrossValidator class
[POS]: experiments/cross_validation/ K-fold cross validation splitter
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import pytest
import numpy as np


def test_kfold_creates_correct_number_of_folds():
    """5-fold should create exactly 5 (train, val) pairs"""
    from experiments.cross_validation.validator import KFoldCrossValidator

    # Mock dataset
    class MockDataset:
        def __len__(self):
            return 50

        def __getitem__(self, idx):
            return np.random.rand(3, 224, 224), idx % 3

    validator = KFoldCrossValidator(num_folds=5)
    folds = validator.create_folds(MockDataset())

    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert len(train_idx) + len(val_idx) == 50


def test_stratified_split_preserves_class_distribution():
    """Each fold should have similar class distribution"""
    from experiments.cross_validation.validator import KFoldCrossValidator

    # 30 samples: 10 of each class
    class MockDataset:
        def __len__(self):
            return 30

        def __getitem__(self, idx):
            return np.random.rand(3, 224, 224), idx % 3  # classes 0, 1, 2

    validator = KFoldCrossValidator(num_folds=5)
    folds = validator.create_folds(MockDataset())

    # Check all folds have all classes represented
    for train_idx, val_idx in folds:
        assert len(np.unique(train_idx % 3)) == 3
        assert len(np.unique(val_idx % 3)) == 3
