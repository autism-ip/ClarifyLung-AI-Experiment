"""
[INPUT]: numpy, typing
[OUTPUT]: KFoldCrossValidator class
[POS]: experiments/cross_validation/ K-fold cross validation splitter
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import dataclass
from typing import List, Tuple, Protocol, Iterable

import numpy as np


class DatasetProtocol(Protocol):
    """Dataset protocol for cross-validation"""

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> tuple:
        ...


@dataclass(frozen=True)
class FoldSplit:
    """Single fold split container"""
    train_indices: np.ndarray
    val_indices: np.ndarray
    fold_number: int


class KFoldCrossValidator:
    """
    K-Fold Cross-Validation with stratified split support

    Splits dataset into K folds, ensuring each fold has balanced
    class distribution when labels are available.
    """

    def __init__(self, num_folds: int = 5, random_seed: int = 42):
        """
        [INPUT]: num_folds - number of folds, random_seed - reproducibility
        """
        if num_folds < 2:
            raise ValueError("num_folds must be >= 2")
        self.num_folds = num_folds
        self.random_seed = random_seed

    def create_folds(self, dataset: DatasetProtocol) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create K stratified folds from dataset

        [INPUT]: dataset with __len__ and __getitem__ (returns label at index 1)
        [OUTPUT]: List of (train_indices, val_indices) tuples
        """
        n_samples = len(dataset)
        indices = np.arange(n_samples)

        # Collect labels if available
        labels = self._extract_labels(dataset, n_samples)

        # Stratified split if labels available
        if labels is not None:
            return self._stratified_split(indices, labels)
        else:
            return self._random_split(indices)

    def _extract_labels(self, dataset: DatasetProtocol, n_samples: int) -> np.ndarray | None:
        """Extract labels from dataset if available"""
        try:
            # Try to get first item to check if labels are available
            _ = dataset[0]
            labels = np.array([dataset[i][1] for i in range(n_samples)])
            # Check if labels are numeric
            if labels.dtype.kind in ('i', 'f'):
                return labels
            return None
        except (IndexError, TypeError, KeyError):
            return None

    def _stratified_split(
        self,
        indices: np.ndarray,
        labels: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split indices maintaining class distribution"""
        rng = np.random.default_rng(self.random_seed)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []

        # Group indices by class
        unique_labels = np.unique(labels)
        class_indices = {label: indices[labels == label] for label in unique_labels}

        # Shuffle each class
        for label in unique_labels:
            rng.shuffle(class_indices[label])

        # Distribute samples to folds
        fold_sizes = len(indices) // self.num_folds
        fold_assignments = [[] for _ in range(self.num_folds)]

        for label in unique_labels:
            label_indices = class_indices[label]
            n_label = len(label_indices)

            # Round-robin assignment to folds
            for i, idx in enumerate(label_indices):
                fold_idx = i % self.num_folds
                fold_assignments[fold_idx].append(idx)

        # Create train/val splits
        for fold_idx in range(self.num_folds):
            val_indices = np.array(fold_assignments[fold_idx])

            # All other indices for training
            train_indices = np.array([
                idx for fold in fold_assignments[:fold_idx] + fold_assignments[fold_idx + 1:]
                for idx in fold
            ])

            folds.append((train_indices, val_indices))

        return folds

    def _random_split(self, indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Simple random K-fold split"""
        rng = np.random.default_rng(self.random_seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        fold_size = len(shuffled) // self.num_folds

        for fold_idx in range(self.num_folds):
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < self.num_folds - 1 else len(shuffled)

            val_indices = shuffled[start:end]
            train_indices = np.concatenate([shuffled[:start], shuffled[end:]])

            folds.append((train_indices, val_indices))

        return folds
