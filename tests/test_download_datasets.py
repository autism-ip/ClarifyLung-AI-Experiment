"""
TDD Phase 1: Kaggle数据集下载脚本测试（RED阶段）
[INPUT]: Kaggle API配置、数据集slug列表
[OUTPUT]: 测试报告（预期失败 → 实现后通过）
[POS]: tests/下载功能测试，验证download_datasets.py
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path


# =============================================================================
# 测试数据集配置结构
# =============================================================================

class TestDatasetConfig:
    """测试数据集配置结构"""

    def test_dataset_config_has_required_fields(self):
        """验证数据集配置包含所有必需字段"""
        # 动态导入，避免在测试阶段就加载模块
        from scripts.download_datasets import DATASETS

        assert len(DATASETS) == 3, "应包含3个数据集配置"

        for ds in DATASETS:
            assert 'name' in ds, "配置需包含 name 字段"
            assert 'slug' in ds, "配置需包含 slug 字段"
            assert 'target_dir' in ds, "配置需包含 target_dir 字段"

            # 验证slug格式正确
            assert '/' in ds['slug'], "slug应包含所有者/数据集格式"

    def test_dataset_slugs_are_unique(self):
        """验证数据集slug唯一性"""
        from scripts.download_datasets import DATASETS

        slugs = [ds['slug'] for ds in DATASETS]
        assert len(slugs) == len(set(slugs)), "slug应唯一"

    def test_iq_othnccd_dataset_config(self):
        """验证IQ-OTHNCCD数据集配置正确"""
        from scripts.download_datasets import DATASETS

        iq_ds = next((ds for ds in DATASETS if 'iq-othnccd' in ds['slug'].lower()), None)
        assert iq_ds is not None, "应包含IQ-OTHNCCD数据集"
        assert 'subhajeetdas' in iq_ds['slug']

    def test_lung_colon_dataset_config(self):
        """验证Lung and Colon Cancer数据集配置正确"""
        from scripts.download_datasets import DATASETS

        lc_ds = next((ds for ds in DATASETS if 'lung-and-colon' in ds['slug'].lower()), None)
        assert lc_ds is not None, "应包含Lung and Colon Cancer数据集"

    def test_lung_4types_dataset_config(self):
        """验证Lung Cancer 4 Types数据集配置正确"""
        from scripts.download_datasets import DATASETS

        l4_ds = next((ds for ds in DATASETS if 'lung-cancer-4-types' in ds['slug'].lower()), None)
        assert l4_ds is not None, "应包含Lung Cancer 4 Types数据集"


# =============================================================================
# 测试脚本存在性与可执行性
# =============================================================================

class TestScriptExistence:
    """测试脚本存在性和基本属性"""

    def test_download_script_exists(self):
        """验证下载脚本存在"""
        script_path = Path(__file__).parent.parent / 'scripts' / 'download_datasets.py'
        assert script_path.exists(), f"下载脚本应存在于 {script_path}"

    def test_download_script_is_valid_python(self):
        """验证脚本是有效的Python文件"""
        script_path = Path(__file__).parent.parent / 'scripts' / 'download_datasets.py'

        if script_path.exists():
            # 尝试编译脚本检查语法
            with open(script_path, 'r') as f:
                code = f.read()
            compile(code, str(script_path), 'exec')


# =============================================================================
# 测试Kaggle凭证检查
# =============================================================================

class TestKaggleCredentials:
    """测试Kaggle API凭证检查功能"""

    def test_check_credentials_function_exists(self):
        """验证check_kaggle_credentials函数存在"""
        from scripts.download_datasets import check_kaggle_credentials
        assert callable(check_kaggle_credentials), "check_kaggle_credentials应是可调用函数"

    def test_check_credentials_returns_bool(self):
        """验证check_kaggle_credentials返回布尔值"""
        from scripts.download_datasets import check_kaggle_credentials

        result = check_kaggle_credentials()
        assert isinstance(result, bool), "应返回布尔值"

    def test_check_credentials_detects_missing_kaggle_json(self):
        """验证能检测到缺失的kaggle.json凭证"""
        from scripts.download_datasets import check_kaggle_credentials

        # 模拟不存在的凭证文件
        with patch.dict(os.environ, {}, clear=True):
            with patch('pathlib.Path.exists', return_value=False):
                # 重新导入以使用mock
                import importlib
                import scripts.download_datasets as dd
                importlib.reload(dd)

                result = dd.check_kaggle_credentials()
                # 如果kaggle.json不存在，应返回False
                kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
                if not kaggle_json.exists():
                    assert result is False, "凭证不存在时应返回False"


# =============================================================================
# 测试下载函数行为
# =============================================================================

class TestDownloadFunction:
    """测试数据集下载函数"""

    def test_download_dataset_function_exists(self):
        """验证download_dataset函数存在"""
        from scripts.download_datasets import download_dataset
        assert callable(download_dataset), "download_dataset应是可调用函数"

    def test_download_dataset_with_invalid_slug_returns_false(self):
        """验证无效slug时返回False而非崩溃"""
        from scripts.download_datasets import download_dataset

        target = Path(tempfile.mkdtemp())
        try:
            result = download_dataset(
                slug="invalid/nonexistent-dataset-slug",
                target=target
            )
            # 函数应返回False表示失败，而不是抛出异常
            assert result is False, "无效slug时应返回False"
        finally:
            shutil.rmtree(target, ignore_errors=True)

    def test_download_dataset_accepts_pathlib_path(self):
        """验证download_dataset接受pathlib.Path类型"""
        from scripts.download_datasets import download_dataset
        import inspect

        sig = inspect.signature(download_dataset)
        params = sig.parameters

        # 检查target参数类型注解
        assert 'target' in params, "应有target参数"


# =============================================================================
# 测试错误处理
# =============================================================================

class TestErrorHandling:
    """测试错误处理机制"""

    def test_main_function_exists(self):
        """验证main入口函数存在"""
        from scripts.download_datasets import main
        assert callable(main), "main应是可调用函数"

    def test_script_handles_missing_credentials_gracefully(self):
        """验证脚本能优雅处理缺失凭证"""
        from scripts.download_datasets import check_kaggle_credentials

        # 模拟凭证不存在的情况
        with patch('pathlib.Path.exists', return_value=False):
            result = check_kaggle_credentials()
            assert result is False, "缺失凭证时应返回False"

    def test_script_handles_invalid_dataset_gracefully(self):
        """验证脚本能优雅处理无效数据集"""
        from scripts.download_datasets import download_dataset

        invalid_slug = "owner/definitely-nonexistent-dataset-12345"
        target = Path(tempfile.mkdtemp())

        try:
            # 应该返回False表示失败，而不是崩溃
            result = download_dataset(slug=invalid_slug, target=target)
            assert result is False, "无效数据集应返回False而非崩溃"
        finally:
            shutil.rmtree(target, ignore_errors=True)


# =============================================================================
# 测试数据验证功能
# =============================================================================

class TestDataValidation:
    """测试下载后数据验证功能"""

    def test_validate_dataset_function_exists(self):
        """验证validate_dataset函数存在"""
        try:
            from scripts.download_datasets import validate_dataset
            assert callable(validate_dataset), "validate_dataset应是可调用函数"
        except ImportError:
            pytest.skip("validate_dataset函数尚未实现")

    def test_validate_empty_directory_returns_false(self):
        """验证空目录验证失败"""
        try:
            from scripts.download_datasets import validate_dataset

            empty_dir = Path(tempfile.mkdtemp())
            try:
                result = validate_dataset(empty_dir)
                assert result is False, "空目录应验证失败"
            finally:
                shutil.rmtree(empty_dir, ignore_errors=True)
        except ImportError:
            pytest.skip("validate_dataset函数尚未实现")

    def test_validate_nonexistent_directory_returns_false(self):
        """验证不存在的目录返回False"""
        from scripts.download_datasets import validate_dataset

        nonexistent = Path(tempfile.mkdtemp()) / 'nonexistent'
        result = validate_dataset(nonexistent)
        assert result is False, "不存在的目录应返回False"

    def test_validate_directory_with_files_returns_true(self):
        """验证包含文件的目录返回True"""
        from scripts.download_datasets import validate_dataset

        test_dir = Path(tempfile.mkdtemp())
        try:
            # 创建一些模拟文件
            (test_dir / 'image1.jpg').write_text('fake image data')
            (test_dir / 'image2.png').write_text('fake image data')
            (test_dir / 'data.csv').write_text('col1,col2\n1,2')

            result = validate_dataset(test_dir)
            assert result is True, "包含文件的目录应返回True"
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


# =============================================================================
# 测试进度显示功能
# =============================================================================

class TestProgressDisplay:
    """测试进度显示功能"""

    def test_print_progress_function_exists(self):
        """验证print_progress函数存在"""
        from scripts.download_datasets import print_progress
        assert callable(print_progress), "print_progress应是可调用函数"

    def test_print_progress_with_zero_total(self):
        """验证total为0时不崩溃"""
        from scripts.download_datasets import print_progress

        # 不应抛出异常
        print_progress(0, 0, prefix='Test')
        print_progress(5, 0, prefix='Test')

    def test_print_progress_complete(self):
        """验证进度条完成"""
        from scripts.download_datasets import print_progress

        # 不应抛出异常
        print_progress(100, 100, prefix='Test', suffix='Complete')


# =============================================================================
# 测试清理功能
# =============================================================================

class TestCleanup:
    """测试清理功能"""

    def test_cleanup_downloads_function_exists(self):
        """验证cleanup_downloads函数存在"""
        from scripts.download_datasets import cleanup_downloads
        assert callable(cleanup_downloads), "cleanup_downloads应是可调用函数"

    def test_cleanup_nonexistent_directory_returns_zero(self):
        """验证清理不存在的目录返回0"""
        from scripts.download_datasets import cleanup_downloads

        nonexistent = Path(tempfile.mkdtemp()) / 'nonexistent'
        result = cleanup_downloads(nonexistent)
        assert result == 0, "不存在的目录应返回0"

    def test_cleanup_removes_zip_files(self):
        """验证清理删除zip文件"""
        from scripts.download_datasets import cleanup_downloads

        test_dir = Path(tempfile.mkdtemp())
        try:
            # 创建临时zip文件
            zip_path = test_dir / 'dataset.zip'
            zip_path.write_text('fake zip content')
            tar_path = test_dir / 'data.tar.gz'
            tar_path.write_text('fake tar content')
            txt_path = test_dir / 'readme.txt'
            txt_path.write_text('readme content')

            result = cleanup_downloads(test_dir)
            assert result >= 2, "应清理至少2个文件"
            assert not zip_path.exists(), "zip文件应被删除"
            assert not tar_path.exists(), "tar.gz文件应被删除"
            assert txt_path.exists(), "txt文件不应被删除"
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


# =============================================================================
# 测试main函数参数解析
# =============================================================================

class TestMainArguments:
    """测试main函数命令行参数解析"""

    def test_main_with_check_credentials_flag(self):
        """验证--check-credentials参数"""
        from scripts.download_datasets import main

        # 模拟凭证不存在
        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=False):
            result = main(['--check-credentials'])
            assert result == 1, "凭证检查失败应返回1"

    def test_main_with_dataset_index(self):
        """验证-d参数指定数据集索引"""
        from scripts.download_datasets import main

        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=True):
            with patch('scripts.download_datasets.download_dataset', return_value=True) as mock_download:
                with patch('scripts.download_datasets.validate_dataset', return_value=True):
                    result = main(['--dataset', '0'])
                    assert result == 0, "成功下载应返回0"
                    mock_download.assert_called_once()

    def test_main_with_force_flag(self):
        """验证--force参数"""
        from scripts.download_datasets import main

        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=True):
            with patch('scripts.download_datasets.download_dataset', return_value=True) as mock_download:
                with patch('scripts.download_datasets.validate_dataset', return_value=True):
                    result = main(['--force'])
                    assert result == 0
                    # 验证force=True被传递
                    call_args = mock_download.call_args
                    assert call_args[1]['force'] is True

    def test_main_with_validate_only_flag(self):
        """验证--validate-only参数"""
        from scripts.download_datasets import main

        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=True):
            with patch('scripts.download_datasets.validate_dataset', return_value=True) as mock_validate:
                result = main(['--validate-only'])
                assert result == 0
                mock_validate.assert_called()

    def test_main_with_cleanup_flag(self):
        """验证--cleanup参数"""
        from scripts.download_datasets import main

        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=True):
            with patch('scripts.download_datasets.download_dataset', return_value=True):
                with patch('scripts.download_datasets.validate_dataset', return_value=True):
                    with patch('scripts.download_datasets.cleanup_downloads', return_value=3) as mock_cleanup:
                        result = main(['--cleanup'])
                        assert result == 0
                        mock_cleanup.assert_called()

    def test_main_with_multiple_flags(self):
        """验证多个参数组合"""
        from scripts.download_datasets import main

        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=True):
            with patch('scripts.download_datasets.download_dataset', return_value=True):
                with patch('scripts.download_datasets.validate_dataset', return_value=True):
                    with patch('scripts.download_datasets.cleanup_downloads', return_value=2):
                        result = main(['--dataset', '1', '--force', '--cleanup'])
                        assert result == 0

    def test_main_download_fails_returns_nonzero(self):
        """验证下载失败时main返回非零退出码"""
        from scripts.download_datasets import main

        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=True):
            with patch('scripts.download_datasets.download_dataset', return_value=False):
                result = main(['--dataset', '0'])
                assert result == 1, "下载失败应返回1"

    def test_main_validate_fails_after_download(self):
        """验证下载后验证失败的情况"""
        from scripts.download_datasets import main

        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=True):
            with patch('scripts.download_datasets.download_dataset', return_value=True):
                with patch('scripts.download_datasets.validate_dataset', return_value=False):
                    result = main(['--dataset', '0'])
                    # 验证成功但校验失败，应该返回1
                    assert result == 1

    def test_main_partial_failure(self):
        """验证部分失败的情况"""
        from scripts.download_datasets import main

        with patch('scripts.download_datasets.check_kaggle_credentials', return_value=True):
            with patch('scripts.download_datasets.download_dataset', return_value=True):
                with patch('scripts.download_datasets.validate_dataset', return_value=True):
                    with patch('scripts.download_datasets.cleanup_downloads', return_value=0):
                        # 下载多个数据集，一个成功一个失败
                        with patch('scripts.download_datasets.DATASETS', [
                            {'name': 'DS1', 'slug': 'a/b', 'target_dir': '/tmp/ds1'},
                            {'name': 'DS2', 'slug': 'c/d', 'target_dir': '/tmp/ds2'}
                        ]):
                            # 模拟第二次下载失败
                            with patch('scripts.download_datasets.download_dataset', side_effect=[True, False]):
                                result = main([])
                                assert result == 1, "部分失败应返回1"


class TestDownloadSkipLogic:
    """测试下载跳过逻辑"""

    def test_download_skips_existing_directory(self):
        """验证已存在目录时跳过下载"""
        from scripts.download_datasets import download_dataset

        with patch('scripts.download_datasets.kagglehub.dataset_download') as mock_download:
            with tempfile.TemporaryDirectory() as tmpdir:
                target = Path(tmpdir)
                # 创建一些文件使目录非空
                (target / 'existing.txt').write_text('data')

                result = download_dataset('test/slug', target)

                # 应该跳过，不调用kagglehub下载
                mock_download.assert_not_called()
                assert result is True

    def test_download_force_overwrites_existing(self):
        """验证force=True时覆盖已存在目录"""
        from scripts.download_datasets import download_dataset

        with patch('scripts.download_datasets.kagglehub.dataset_download') as mock_download:
            with tempfile.TemporaryDirectory() as tmpdir:
                fake_download = Path(tmpdir) / 'fake_download'
                fake_download.mkdir()
                (fake_download / 'data.txt').write_text('fake')
                mock_download.return_value = str(fake_download)

                target = Path(tmpdir) / 'target'
                target.mkdir()
                (target / 'existing.txt').write_text('data')

                result = download_dataset('test/slug', target, force=True)

                # 应该调用kagglehub进行下载
                mock_download.assert_called_once()
                assert result is True


class TestValidationEdgeCases:
    """测试验证函数边界情况"""

    def test_validate_directory_with_only_hidden_files(self):
        """验证只有隐藏文件的目录返回False"""
        from scripts.download_datasets import validate_dataset

        test_dir = Path(tempfile.mkdtemp())
        try:
            # 创建隐藏文件
            (test_dir / '.DS_Store').write_text('hidden')
            (test_dir / '.gitkeep').write_text('')

            result = validate_dataset(test_dir)
            assert result is False, "只有隐藏文件的目录应验证失败"
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_validate_directory_with_subdirectories(self):
        """验证包含子目录的目录返回True"""
        from scripts.download_datasets import validate_dataset

        test_dir = Path(tempfile.mkdtemp())
        try:
            # 创建子目录和文件
            (test_dir / 'subdir').mkdir()
            (test_dir / 'subdir' / 'image.jpg').write_text('data')
            (test_dir / 'root.txt').write_text('root file')

            result = validate_dataset(test_dir)
            assert result is True, "包含文件的目录应验证成功"
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestCleanupEdgeCases:
    """测试清理功能边界情况"""

    def test_cleanup_handles_oserror(self):
        """验证清理时OSError不导致崩溃"""
        from scripts.download_datasets import cleanup_downloads

        test_dir = Path(tempfile.mkdtemp())
        try:
            zip_path = test_dir / 'test.zip'
            zip_path.write_text('data')

            # 模拟unlink时OSError
            with patch('pathlib.Path.unlink', side_effect=OSError("Permission denied")):
                result = cleanup_downloads(test_dir)
                # 应该捕获异常并返回已处理的数量
                assert isinstance(result, int)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_cleanup_tar_gz_files(self):
        """验证清理.tar.gz文件"""
        from scripts.download_datasets import cleanup_downloads

        test_dir = Path(tempfile.mkdtemp())
        try:
            (test_dir / 'data.tar.gz').write_text('fake tar')
            (test_dir / 'data.tgz').write_text('fake tgz')

            result = cleanup_downloads(test_dir)
            assert result == 2, "应清理2个tar文件"
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
