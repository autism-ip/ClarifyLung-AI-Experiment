# =============================================================================
# model_comparison Package
# =============================================================================
"""
[INPUT]: pandas, matplotlib, numpy
[OUTPUT]: plot_model_comparison, create_comparison_table
[POS]: experiments/visualization/ model comparison visualization
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def create_comparison_table(results: Dict[str, Dict[str, float]], metric_names: List[str]) -> pd.DataFrame:
    """
    Create a comparison table from benchmark results.

    Args:
        results: Dictionary mapping model names to their metric dictionaries
                 e.g., {"ResNet50": {"accuracy": 0.85, "f1_score": 0.84}}
        metric_names: List of metric names to include in the table

    Returns:
        pandas DataFrame with model names as index and metrics as columns

    Raises:
        ValueError: If results is empty
    """
    if not results:
        raise ValueError("Results cannot be empty")

    data = {}
    for model_name, metrics in results.items():
        row = {}
        for metric in metric_names:
            row[metric] = metrics.get(metric, 0.0)
        data[model_name] = row

    df = pd.DataFrame(data).T
    df.index.name = "model"
    return df


def plot_model_comparison(
    benchmark_results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot grouped bar charts comparing models across multiple metrics.

    Args:
        benchmark_results: Dictionary mapping model names to metric dictionaries
                          e.g., {"ResNet50": {"accuracy": 0.85, "f1_score": 0.84}}
        metrics: List of metric names to plot. If None, uses all metrics found in results
        save_path: Optional path to save the figure. If None, figure is not saved

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If metrics list is empty
    """
    if metrics is None:
        # Use all metrics found in results
        all_metrics = set()
        for model_metrics in benchmark_results.values():
            all_metrics.update(model_metrics.keys())
        metrics = list(all_metrics)
    elif len(metrics) == 0:
        raise ValueError("At least one metric must be provided")

    if not metrics:
        raise ValueError("At least one metric must be provided")

    # Create DataFrame for easier handling
    df = create_comparison_table(benchmark_results, metrics)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar dimensions
    n_models = len(df)
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics  # Width of each bar within a group
    group_width = 1.0  # Width of each model group
    group_positions = np.arange(n_models) * group_width

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        positions = group_positions + i * bar_width
        values = df[metric].values
        bars = ax.bar(positions, values, bar_width * 0.9, label=metric, color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Configure axes
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(group_positions + (n_metrics - 1) * bar_width / 2)
    ax.set_xticklabels(df.index, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Example usage
    sample_results = {
        "ResNet50": {"accuracy": 0.85, "f1_score": 0.84, "precision": 0.83},
        "ViT": {"accuracy": 0.87, "f1_score": 0.86, "precision": 0.85},
        "Hybrid-Basic": {"accuracy": 0.88, "f1_score": 0.87, "precision": 0.86},
        "Hybrid-Advanced": {"accuracy": 0.90, "f1_score": 0.89, "precision": 0.88},
    }

    print("=" * 60)
    print("Model Comparison Table")
    print("=" * 60)
    table = create_comparison_table(sample_results, ["accuracy", "f1_score", "precision"])
    print(table)

    print("\n" + "=" * 60)
    print("Generating Plot...")
    print("=" * 60)
    fig = plot_model_comparison(sample_results, metrics=["accuracy", "f1_score", "precision"])
    plt.show()
