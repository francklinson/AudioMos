"""
配置管理模块
从YAML配置文件加载配置
"""
import os
import yaml
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class BackendConfig(BaseSettings):
    """后端服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000


class FrontendConfig(BaseSettings):
    """前端服务配置"""
    host: str = "0.0.0.0"
    port: int = 3000


class ServerConfig(BaseSettings):
    """服务器配置"""
    backend: BackendConfig = BackendConfig()
    frontend: FrontendConfig = FrontendConfig()
    debug: bool = False


class AuthConfig(BaseSettings):
    """认证配置"""
    secret_key: str = "your-secret-key-change-this-in-production"
    access_token_expire_minutes: int = 60
    admin_username: str = "admin"
    admin_password: str = "tp123456"


class PathsConfig(BaseSettings):
    """路径配置"""
    ref_dir: str = "./data/ref"
    upload_dir: str = "./data/uploads"
    result_dir: str = "./data/results"
    temp_dir: str = "./data/temp"
    models_dir: str = "./models"


class CUDAConfig(BaseSettings):
    """CUDA配置"""
    enabled: bool = True
    device_id: int = 0


class LoggingConfig(BaseSettings):
    """日志配置"""
    level: str = "INFO"
    file: str = "./logs/backend.log"
    max_size: int = 10
    backup_count: int = 5


class AudioConfig(BaseSettings):
    """音频配置"""
    target_sample_rate: int = 16000
    supported_formats: List[str] = [".wav", ".mp3"]
    max_file_size: int = 100


class Config(BaseSettings):
    """全局配置"""
    server: ServerConfig = ServerConfig()
    auth: AuthConfig = AuthConfig()
    paths: PathsConfig = PathsConfig()
    cuda: CUDAConfig = CUDAConfig()
    logging: LoggingConfig = LoggingConfig()
    audio: AudioConfig = AudioConfig()


def load_config(config_path: str = None) -> Config:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径,默认为项目根目录的config.yaml
        
    Returns:
        Config对象
    """
    if config_path is None:
        # 查找配置文件
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "config" / "config.yaml",
            Path(__file__).parent.parent.parent.parent / "config.yaml",
            Path.cwd() / "config" / "config.yaml",
            Path.cwd() / "config.yaml",
            Path("/app/config.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    config = Config()

    # 确定项目根目录
    if config_path:
        config_file_path = Path(config_path)
        if config_file_path.parent.name == "config":
            project_root = config_file_path.parent.parent
        else:
            project_root = config_file_path.parent
    else:
        project_root = Path(__file__).parent.parent.parent.parent
    
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        
        if yaml_config:
            # 服务器配置
            if "server" in yaml_config:
                server_data = yaml_config["server"]
                if "backend" in server_data:
                    config.server.backend = BackendConfig(**server_data["backend"])
                if "frontend" in server_data:
                    config.server.frontend = FrontendConfig(**server_data["frontend"])
                if "debug" in server_data:
                    config.server.debug = server_data["debug"]
            
            if "auth" in yaml_config:
                config.auth = AuthConfig(**yaml_config["auth"])
            if "paths" in yaml_config:
                config.paths = PathsConfig(**yaml_config["paths"])
            if "cuda" in yaml_config:
                config.cuda = CUDAConfig(**yaml_config["cuda"])
            if "logging" in yaml_config:
                config.logging = LoggingConfig(**yaml_config["logging"])
            if "audio" in yaml_config:
                config.audio = AudioConfig(**yaml_config["audio"])
    
    # 将相对路径转换为基于项目根目录的绝对路径
    def resolve_path(path_str: str) -> str:
        """将路径解析为基于项目根目录的绝对路径"""
        if not path_str:
            return path_str
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str((project_root / path).resolve())
    
    # 解析所有路径配置
    config.paths.ref_dir = resolve_path(config.paths.ref_dir)
    config.paths.upload_dir = resolve_path(config.paths.upload_dir)
    config.paths.result_dir = resolve_path(config.paths.result_dir)
    config.paths.temp_dir = resolve_path(config.paths.temp_dir)
    config.paths.models_dir = resolve_path(config.paths.models_dir)
    config.logging.file = resolve_path(config.logging.file)
    
    # 从环境变量覆盖配置 - 后端
    if os.getenv("AUDIOMOS_BACKEND_HOST"):
        config.server.backend.host = os.getenv("AUDIOMOS_BACKEND_HOST")
    if os.getenv("AUDIOMOS_BACKEND_PORT"):
        config.server.backend.port = int(os.getenv("AUDIOMOS_BACKEND_PORT"))
    
    # 从环境变量覆盖配置 - 前端
    if os.getenv("AUDIOMOS_FRONTEND_HOST"):
        config.server.frontend.host = os.getenv("AUDIOMOS_FRONTEND_HOST")
    if os.getenv("AUDIOMOS_FRONTEND_PORT"):
        config.server.frontend.port = int(os.getenv("AUDIOMOS_FRONTEND_PORT"))
    
    # 向后兼容旧的环境变量
    if os.getenv("AUDIOMOS_HOST"):
        config.server.backend.host = os.getenv("AUDIOMOS_HOST")
    if os.getenv("AUDIOMOS_PORT"):
        config.server.backend.port = int(os.getenv("AUDIOMOS_PORT"))
    
    if os.getenv("AUDIOMOS_SECRET_KEY"):
        config.auth.secret_key = os.getenv("AUDIOMOS_SECRET_KEY")
    if os.getenv("AUDIOMOS_REF_DIR"):
        config.paths.ref_dir = os.getenv("AUDIOMOS_REF_DIR")
    if os.getenv("AUDIOMOS_UPLOAD_DIR"):
        config.paths.upload_dir = os.getenv("AUDIOMOS_UPLOAD_DIR")
    if os.getenv("AUDIOMOS_RESULT_DIR"):
        config.paths.result_dir = os.getenv("AUDIOMOS_RESULT_DIR")
    if os.getenv("AUDIOMOS_CUDA_ENABLED"):
        config.cuda.enabled = os.getenv("AUDIOMOS_CUDA_ENABLED").lower() == "true"
    if os.getenv("AUDIOMOS_LOG_LEVEL"):
        config.logging.level = os.getenv("AUDIOMOS_LOG_LEVEL")
    
    return config


# 全局配置实例
settings = load_config()
