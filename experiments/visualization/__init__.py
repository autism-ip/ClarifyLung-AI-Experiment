# =============================================================================
# visualization Package
# =============================================================================
"""
[INPUT]: 依赖 gradcam, torch, numpy, matplotlib, seaborn, sklearn, class_distribution
[OUTPUT]: GradCAMVisualizer, AttentionVisualizer, overlay_heatmap, visualize_attention, compute_confusion_matrix, plot_confusion_matrix, plot_class_distribution, plot_pie_chart
[POS]: experiments/visualization/ 可解释性可视化模块
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

# Optional imports - grad-cam package may not be installed
try:
    from .gradcam import GradCAMVisualizer, overlay_heatmap
except ImportError:
    GradCAMVisualizer = None
    overlay_heatmap = None

try:
    from .attention_maps import AttentionVisualizer, visualize_attention, visualize_multihead_attention
except ImportError:
    AttentionVisualizer = None
    visualize_attention = None
    visualize_multihead_attention = None

try:
    from .model_comparison import plot_model_comparison, create_comparison_table
except ImportError:
    plot_model_comparison = None
    create_comparison_table = None

try:
    from .training_curves import plot_training_curves, save_training_plot
except ImportError:
    plot_training_curves = None
    save_training_plot = None

try:
    from .confusion_matrix import compute_confusion_matrix, plot_confusion_matrix
except ImportError:
    compute_confusion_matrix = None
    plot_confusion_matrix = None

try:
    from .class_distribution import plot_class_distribution, plot_pie_chart
except ImportError:
    plot_class_distribution = None
    plot_pie_chart = None

__all__ = [
    'GradCAMVisualizer',
    'overlay_heatmap',
    'AttentionVisualizer',
    'visualize_attention',
    'visualize_multihead_attention',
    'plot_model_comparison',
    'create_comparison_table',
    'plot_training_curves',
    'save_training_plot',
    'compute_confusion_matrix',
    'plot_confusion_matrix',
    'plot_class_distribution',
    'plot_pie_chart',
]