"""
日志配置模块
提供统一的日志配置 - 前后端一体架构专用
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import settings


def setup_logging() -> logging.Logger:
    """
    设置日志配置 - 只输出到 unified.log

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

    # 控制台处理器 - 输出到stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器 - unified.log（前后端一体架构唯一日志文件）
    log_dir = Path(settings.logging.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    unified_log_path = log_dir / "unified.log"

    file_handler = RotatingFileHandler(
        unified_log_path,
        maxBytes=settings.logging.max_size * 1024 * 1024,
        backupCount=settings.logging.backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, settings.logging.level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"[日志配置] 日志系统初始化完成，输出到: {unified_log_path}")

    return logger


# 全局logger实例
logger = setup_logging()
