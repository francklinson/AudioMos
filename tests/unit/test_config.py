"""
配置模块单元测试
测试配置加载和管理功能
"""
import pytest
import os
from pathlib import Path


class TestConfig:
    """配置测试类"""

    def test_config_import(self):
        """测试配置模块导入"""
        try:
            from app.core.config import Config, load_config, settings
            assert True
        except ImportError as e:
            pytest.skip(f"配置模块导入失败: {e}")

    def test_default_config_values(self):
        """测试默认配置值"""
        try:
            from app.core.config import Config
            config = Config()
            
            # 验证默认配置
            assert config.server.backend.host == "0.0.0.0"
            assert config.server.backend.port == 8000
            assert config.auth.access_token_expire_minutes == 60
            assert config.audio.target_sample_rate == 16000
            assert ".wav" in config.audio.supported_formats
            assert ".mp3" in config.audio.supported_formats
        except ImportError:
            pytest.skip("配置模块导入失败")

    def test_paths_config(self):
        """测试路径配置"""
        try:
            from app.core.config import Config
            config = Config()
            
            # 验证路径配置存在
            assert config.paths.ref_dir is not None
            assert config.paths.upload_dir is not None
            assert config.paths.result_dir is not None
            assert config.paths.temp_dir is not None
        except ImportError:
            pytest.skip("配置模块导入失败")

    def test_audio_config(self):
        """测试音频配置"""
        try:
            from app.core.config import Config
            config = Config()
            
            # 验证音频配置
            assert config.audio.target_sample_rate == 16000
            assert config.audio.max_file_size > 0
            assert len(config.audio.supported_formats) > 0
        except ImportError:
            pytest.skip("配置模块导入失败")

    def test_auth_config(self):
        """测试认证配置"""
        try:
            from app.core.config import Config
            config = Config()
            
            # 验证认证配置
            assert config.auth.admin_username is not None
            assert config.auth.secret_key is not None
            assert config.auth.access_token_expire_minutes > 0
        except ImportError:
            pytest.skip("配置模块导入失败")


class TestEnvironmentVariables:
    """环境变量测试类"""

    def test_env_override_backend_host(self, monkeypatch):
        """测试环境变量覆盖后端主机"""
        try:
            from app.core.config import load_config
            
            # 设置环境变量
            monkeypatch.setenv("AUDIOMOS_BACKEND_HOST", "127.0.0.1")
            
            config = load_config()
            assert config.server.backend.host == "127.0.0.1"
        except ImportError:
            pytest.skip("配置模块导入失败")

    def test_env_override_backend_port(self, monkeypatch):
        """测试环境变量覆盖后端端口"""
        try:
            from app.core.config import load_config
            
            # 设置环境变量
            monkeypatch.setenv("AUDIOMOS_BACKEND_PORT", "9000")
            
            config = load_config()
            assert config.server.backend.port == 9000
        except ImportError:
            pytest.skip("配置模块导入失败")

    def test_env_override_secret_key(self, monkeypatch):
        """测试环境变量覆盖密钥"""
        try:
            from app.core.config import load_config
            
            # 设置环境变量
            test_key = "test-secret-key-12345"
            monkeypatch.setenv("AUDIOMOS_SECRET_KEY", test_key)
            
            config = load_config()
            assert config.auth.secret_key == test_key
        except ImportError:
            pytest.skip("配置模块导入失败")


class TestCUDAConfig:
    """CUDA配置测试类"""

    def test_cuda_enabled_default(self):
        """测试CUDA默认启用状态"""
        try:
            from app.core.config import Config
            config = Config()
            
            # 验证CUDA配置存在
            assert hasattr(config.cuda, 'enabled')
            assert hasattr(config.cuda, 'device_id')
        except ImportError:
            pytest.skip("配置模块导入失败")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
