"""
日志配置模块
提供统一的日志配置
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import settings


def setup_logging() -> logging.Logger:
    """
    设置日志配置
    
    Returns:
        配置好的logger实例
    """
    # 创建logger
    logger = logging.getLogger("audiomos")
    logger.setLevel(getattr(logging, settings.logging.level.upper()))
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    log_file = Path(settings.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=settings.logging.max_size * 1024 * 1024,
        backupCount=settings.logging.backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, settings.logging.level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# 全局logger实例
logger = setup_logging()
