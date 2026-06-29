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

    # 如果已经配置过，直接返回（防止重复初始化）
    if logger.handlers:
        return logger

    # 阻止日志向父logger传播（防止Uvicorn等重复输出）
    logger.propagate = False

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件处理器 - audiomos.log（主日志文件）
    log_dir = Path(settings.logging.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "audiomos.log"

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.logging.max_size * 1024 * 1024,
        backupCount=settings.logging.backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, settings.logging.level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器 - 仅开发环境输出到stdout
    if settings.logging.level.upper() == "DEBUG":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"[日志配置] 日志系统初始化完成，输出到: {log_path}")

    return logger


# 全局logger实例
logger = setup_logging()
