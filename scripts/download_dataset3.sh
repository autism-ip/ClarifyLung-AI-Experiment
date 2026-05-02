#!/bin/bash
source activate lung_cancer

echo "开始下载第三个数据集..."
echo "开始时间: $(date)"
echo "PID: $$"
echo $$ > /tmp/download_dataset3.pid

python -c "
import kagglehub
import shutil
from pathlib import Path
import time

print('开始下载第三个数据集...')
start_time = time.time()

try:
    path = kagglehub.dataset_download('kabil007/lungcancer4types-imagedataset')
    end_time = time.time()
    print(f'下载完成，耗时: {end_time - start_time:.2f}秒')
    print(f'下载路径: {path}')
    
    target = Path('datasets/Lung4Types')
    target.mkdir(parents=True, exist_ok=True)
    
    downloaded_path = Path(path)
    if downloaded_path != target:
        print('正在移动文件到目标目录...')
        for item in downloaded_path.iterdir():
            dest = target / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    
    print('下载和移动完成!')
except Exception as e:
    print(f'下载失败: {e}')

"

echo "结束时间: $(date)"
rm -f /tmp/download_dataset3.pid
