# =============================================================================
# 数据集配置
# =============================================================================
"""
[INPUT]: 环境变量 LUNG_DATASET_DIR (可选), 或本地 datasets/ 目录
[OUTPUT]: 数据集路径配置字典 DATASET_PATHS, DATASET_INFO, 标签映射
[POS]: configs/ 数据集路径配置中心, 被 data/custom_dataset.py 和 scripts/* 消费
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import os
from pathlib import Path

# =============================================================================
# 数据集根目录解析
# =============================================================================
# 优先从环境变量读取，未设置则回退到项目本地 datasets/
# 远程部署时只需: export LUNG_DATASET_DIR=/your/path

_BASE_DIR = os.environ.get("LUNG_DATASET_DIR", str(Path(__file__).parent.parent / "datasets"))

# =============================================================================
# 数据集路径配置
# =============================================================================

DATASET_PATHS = {
    # Dataset 1: IQ-OTHNCCD Lung Cancer Dataset
    # 结构: root/Augmented IQ-OTHNCCD lung cancer dataset/{Normal cases,Malignant cases,Benign cases}/
    "dataset1": os.path.join(_BASE_DIR, "IQ-OTHNCCD"),

    # Dataset 2: Lung and Colon Cancer Histopathological Images
    # 结构: root/lung_colon_image_set/lung_image_sets/{lung_n,lung_aca,lung_scc}/
    "dataset2": os.path.join(_BASE_DIR, "LungColon"),

    # Dataset 3: Lung Cancer 4 Types Image Dataset
    # 结构: root/Data/{train,valid,test}/{class_subdir}/  (class_subdir含肿瘤位置信息)
    "dataset3": os.path.join(_BASE_DIR, "Lung4Types"),
}

# =============================================================================
# 数据集信息
# =============================================================================

DATASET_INFO = {
    "dataset1": {
        "name": "IQ-OTHNCCD Lung Cancer Dataset",
        "description": "胸部X光片",
        "source": "IQ-OTHNCCD",
        "modalities": ["X-ray"],
    },
    "dataset2": {
        "name": "Lung and Colon Cancer Histopathological Images",
        "description": "肺组织切片",
        "source": "Kaggle",
        "modalities": ["Histopathology"],
    },
    "dataset3": {
        "name": "Lung Cancer 4 Types Image Dataset",
        "description": "4种肺癌类型",
        "source": "Kaggle",
        "modalities": ["Histopathology"],
    },
}

# =============================================================================
# 标签映射配置 (与 data/custom_dataset.py 保持一致)
# =============================================================================

LABEL_TO_INDEX = {
    "normal": 0,
    "benign": 1,
    "malignant": 2,
}

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

CLASS_NAMES = ["normal", "benign", "malignant"]

# =============================================================================
# 验证配置
# =============================================================================

def validate_paths():
    """验证所有数据集路径是否存在"""
    import os

    errors = []
    for name, path in DATASET_PATHS.items():
        if not Path(path).exists():
            errors.append(f"{name}: {path} 不存在")

    if errors:
        print("数据集路径验证失败:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("✓ 所有数据集路径验证通过")
    return True


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    print("数据集配置验证")
    print("=" * 60)

    validate_paths()

    print("\n数据集路径:")
    for name, path in DATASET_PATHS.items():
        info = DATASET_INFO.get(name, {})
        print(f"  {name}: {info.get('name', name)}")
        print(f"    路径: {path}")
