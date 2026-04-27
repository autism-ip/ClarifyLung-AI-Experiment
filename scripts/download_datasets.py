"""
Kaggle数据集下载脚本
[INPUT]: Kaggle credentials, dataset slugs
[OUTPUT]: Downloaded and extracted datasets
[POS]: scripts/ Kaggle数据集下载脚本，被 tests/test_download_datasets.py 验证
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

================================================================================
手动下载指南 (当Kaggle API不可用时)
================================================================================

本脚本优先使用 kagglehub 自动下载。若因网络、认证或地区限制导致自动下载失败，
请按以下步骤手动下载并放置到对应目录。

--- Dataset 0: IQ-OTHNCCD Lung Cancer Dataset ----------------------------------
来源: https://www.kaggle.com/datasets/subhajeetdas/iq-othnccd-lung-cancer-dataset-augmented
步骤:
  1. 访问上述Kaggle页面
  2. 点击 "Download" 获取压缩包
  3. 解压到: datasets/IQ-OTHNCCD/
期望目录结构:
  datasets/IQ-OTHNCCD/
  └── Augmented IQ-OTHNCCD lung cancer dataset/
      ├── Normal cases/
      ├── Malignant cases/
      └── Benign cases/

--- Dataset 1: Lung and Colon Cancer Histopathological Images ------------------
来源: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images
步骤:
  1. 访问上述Kaggle页面
  2. 点击 "Download" 获取压缩包
  3. 解压到: datasets/LungColon/
期望目录结构:
  datasets/LungColon/
  └── lung_colon_image_set/
      └── lung_image_sets/
          ├── lung_n/          (正常肺组织)
          ├── lung_aca/        (肺腺癌)
          └── lung_scc/        (肺鳞状细胞癌)
      └── colon_image_sets/    (本项目不使用结肠部分)

--- Dataset 2: Lung Cancer 4 Types Image Dataset -------------------------------
来源: https://www.kaggle.com/datasets/sanjeevjangir/lung-cancer-4-types-image-dataset
步骤:
  1. 访问上述Kaggle页面
  2. 点击 "Download" 获取压缩包
  3. 解压到: datasets/Lung4Types/
期望目录结构:
  datasets/Lung4Types/
  └── Data/
      ├── train/
      │   ├── normal/
      │   ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
      │   ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
      │   └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
      ├── valid/
      │   └── ... (同上4类)
      └── test/
          ├── normal/
          ├── adenocarcinoma/
          ├── large.cell.carcinoma/
          └── squamous.cell.carcinoma/

================================================================================
兼容性说明
================================================================================
data/custom_dataset.py 已做自适应路径探测，兼容以下两种传入方式:
  A) 传入数据集根目录 (如 datasets/IQ-OTHNCCD/)
  B) 传入深层目录 (如 datasets/IQ-OTHNCCD/Augmented IQ-OTHNCCD lung cancer dataset/)
推荐方式 A，让加载器自动探测子目录。
================================================================================
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional

import kagglehub


# =============================================================================
# 数据集配置
# =============================================================================

DATASETS: List[Dict[str, str]] = [
    {
        "name": "IQ-OTHNCCD Lung Cancer Dataset",
        "slug": "subhajeetdas/iq-othnccd-lung-cancer-dataset-augmented",
        "target_dir": "datasets/IQ-OTHNCCD"
    },
    {
        "name": "Lung and Colon Cancer Histopathological Images",
        "slug": "andrewmvd/lung-and-colon-cancer-histopathological-images",
        "target_dir": "datasets/LungColon"
    },
    {
        "name": "Lung Cancer 4 Types Image Dataset",
        "slug": "sanjeevjangir/lung-cancer-4-types-image-dataset",
        "target_dir": "datasets/Lung4Types"
    }
]


# =============================================================================
# Kaggle凭证检查
# =============================================================================

def check_kaggle_credentials() -> bool:
    """
    检查Kaggle API凭证是否配置正确

    [OUTPUT]: bool - 凭证是否存在且有效
    """
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if not kaggle_json.exists():
        print(f"[ERROR] Kaggle凭证文件不存在: {kaggle_json}")
        print("请先配置Kaggle API凭证:")
        print("  1. 访问 https://www.kaggle.com/account")
        print("  2. 点击 'Create New API Token' 下载 kaggle.json")
        print(f"  3. 将文件移动到: {kaggle_json.parent}/")
        return False

    # 检查文件权限（Unix系统）
    if os.name == 'posix':
        stat_info = os.stat(kaggle_json)
        mode = stat_info.st_mode & 0o777
        if mode & 0o077:  # 检查组或其他用户是否有读权限
            print(f"[WARNING] Kaggle凭证文件权限过宽: {oct(mode)}")
            print("建议运行: chmod 600 ~/.kaggle/kaggle.json")

    return True


# =============================================================================
# 进度显示工具
# =============================================================================

def print_progress(iteration: int, total: int, prefix: str = '', suffix: str = '',
                   length: int = 50, fill: str = '#') -> None:
    """打印进度条"""
    if total <= 0:
        return
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


# =============================================================================
# 数据集下载
# =============================================================================

def download_dataset(slug: str, target: Path, force: bool = False) -> bool:
    """
    下载并解压Kaggle数据集

    [INPUT]:
        slug: Kaggle数据集slug (owner/dataset-name)
        target: 目标目录路径
        force: 是否强制重新下载

    [OUTPUT]: bool - 下载是否成功
    """
    target = Path(target)

    # 检查是否已存在且非强制下载
    if target.exists() and not force:
        if any(target.iterdir()):
            print(f"[SKIP] 数据集已存在: {target} (使用 --force 重新下载)")
            return True

    # 创建目标目录
    target.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] 开始下载: {slug}")
    print(f"[TARGET] 目标路径: {target}")

    try:
        # 使用 kagglehub 下载
        downloaded_path = kagglehub.dataset_download(slug)

        if downloaded_path and Path(downloaded_path).exists():
            # 移动到目标目录
            downloaded_path = Path(downloaded_path)

            # 如果下载的内容是目录且与target不同，则移动
            if downloaded_path != target:
                # 移动所有内容到目标目录
                for item in downloaded_path.iterdir():
                    dest = target / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)

            print(f"[SUCCESS] 下载完成: {slug}")
            return True
        else:
            print(f"[ERROR] 下载失败: kagglehub返回无效路径")
            return False

    except Exception as e:
        print(f"[ERROR] 下载异常: {e}")
        return False


def validate_dataset(dataset_path: Path) -> bool:
    """
    验证数据集目录是否包含有效数据

    [INPUT]: dataset_path - 数据集根目录
    [OUTPUT]: bool - 数据集是否有效
    """
    if not dataset_path.exists():
        return False

    if not dataset_path.is_dir():
        return False

    # 检查是否为空目录
    if not any(dataset_path.iterdir()):
        return False

    # 统计文件数量（递归）
    files = list(dataset_path.rglob('*'))
    data_files = [f for f in files if f.is_file() and not f.name.startswith('.')]

    if len(data_files) == 0:
        return False

    print(f"[VALIDATE] 数据集 {dataset_path.name}: {len(data_files)} 个文件")
    return True


def cleanup_downloads(download_dir: Path) -> int:
    """
    清理下载过程中的临时文件

    [INPUT]: download_dir - 下载目录
    [OUTPUT]: int - 清理的文件数量
    """
    if not download_dir.exists():
        return 0

    removed = 0
    patterns = ['*.zip', '*.tar.gz', '*.tgz', '*.csv']

    for pattern in patterns:
        for file in download_dir.rglob(pattern):
            try:
                file.unlink()
                removed += 1
                print(f"[CLEANUP] 删除临时文件: {file}")
            except OSError:
                pass

    return removed


# =============================================================================
# 主入口
# =============================================================================

def main(argv: List[str] = None) -> int:
    """
    主入口函数

    [INPUT]: argv - 命令行参数列表（可选）
    [OUTPUT]: int - 退出码（0成功，1失败）
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='下载Kaggle医学影像数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_datasets.py                    # 下载所有数据集
  python scripts/download_datasets.py --dataset 0        # 只下载第一个数据集
  python scripts/download_datasets.py --force            # 强制重新下载
  python scripts/download_datasets.py --validate-only    # 仅验证已下载数据
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        type=int,
        choices=[0, 1, 2],
        help='指定数据集索引 (0=IQ-OTHNCCD, 1=Lung-Colon, 2=Lung4Types)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制重新下载已存在的数据集'
    )

    parser.add_argument(
        '--validate-only', '-v',
        action='store_true',
        help='仅验证已下载的数据集，不进行下载'
    )

    parser.add_argument(
        '--cleanup', '-c',
        action='store_true',
        help='下载完成后清理临时文件'
    )

    parser.add_argument(
        '--check-credentials',
        action='store_true',
        help='仅检查Kaggle凭证是否配置正确'
    )

    args = parser.parse_args(argv)

    # 仅检查凭证
    if args.check_credentials:
        if check_kaggle_credentials():
            print("[OK] Kaggle凭证配置正确")
            return 0
        else:
            print("[FAIL] Kaggle凭证配置不正确")
            return 1

    # 检查凭证
    if not check_kaggle_credentials():
        return 1

    # 确定要处理的数据集
    if args.dataset is not None:
        selected_datasets = [DATASETS[args.dataset]]
    else:
        selected_datasets = DATASETS

    print("=" * 60)
    print("Kaggle 医学影像数据集下载")
    print("=" * 60)
    print(f"数据集数量: {len(selected_datasets)}")
    print()

    results = {}

    for i, ds in enumerate(selected_datasets):
        target = Path(ds['target_dir'])

        print(f"[{i+1}/{len(selected_datasets)}] {ds['name']}")
        print("-" * 40)

        if args.validate_only:
            # 仅验证模式
            if validate_dataset(target):
                print(f"[PASS] 数据集有效\n")
                results[ds['name']] = True
            else:
                print(f"[FAIL] 数据集无效或不存在\n")
                results[ds['name']] = False
        else:
            # 下载模式
            if download_dataset(ds['slug'], target, force=args.force):
                print(f"[PASS] 下载成功\n")
                results[ds['name']] = True

                # 验证下载的数据
                if not validate_dataset(target):
                    print(f"[WARN] 下载完成但验证失败: {target}")
                    results[ds['name']] = False

                # 清理临时文件
                if args.cleanup:
                    cleaned = cleanup_downloads(target)
                    if cleaned > 0:
                        print(f"[CLEANUP] 清理了 {cleaned} 个临时文件")
            else:
                print(f"[FAIL] 下载失败\n")
                results[ds['name']] = False

    # 打印汇总
    print("=" * 60)
    print("下载汇总")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print()
    print(f"成功: {success_count}/{total_count}")

    if success_count == total_count:
        print("\n[ALL DONE] 所有数据集下载完成!")
        return 0
    else:
        print("\n[PARTIAL] 部分数据集下载失败，请检查上述错误信息")
        return 1


if __name__ == '__main__':
    sys.exit(main())
