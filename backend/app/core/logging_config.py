"""
日志配置模块
提供统一的日志配置，优化格式化输出和可读性
"""
import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import settings


class ColorFormatter(logging.Formatter):
    """带颜色的日志格式化器（控制台用）"""
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[1;31m',  # 亮红
        'RESET': '\033[0m',
    }
    # 特殊标记高亮
    HIGHLIGHTS = {
        '✅': '\033[32m',
        '❌': '\033[31m',
        '⚠️': '\033[33m',
        '✓': '\033[32m',
        '✗': '\033[31m',
    }

    def format(self, record):
        # 先让父类处理消息
        s = super().format(record)
        # 仅控制台添加颜色
        if hasattr(record, '_console') and record._console:
            color = self.COLORS.get(record.levelname, '')
            if color:
                s = f"{color}{s}{self.COLORS['RESET']}"
        return s


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

    # 统一格式: 时间 [级别] 模块.函数:行号 - 消息
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 更简洁的控制台格式
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S"
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
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台处理器 - 直接输出到stdout（不在DEBUG时才启用，始终启用）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if settings.logging.level.upper() != "DEBUG" else logging.DEBUG)
    # 使用标准格式（不依赖自定义属性）包装颜色
    original_format = console_handler.format

    def _console_format(record):
        record._console = True
        return console_formatter.format(record)

    console_handler.format = _console_format
    logger.addHandler(console_handler)

    logger.info(f"[日志配置] 日志系统初始化完成")
    logger.info(f"[日志配置] 日志文件: {log_path}")
    logger.info(f"[日志配置] 日志级别: {settings.logging.level.upper()}")

    return logger


# 全局logger实例
logger = setup_logging()


def log_request(method: str, path: str, status_code: int, duration_ms: float, extra: str = ""):
    """
    记录HTTP请求日志（统一格式）

    Args:
        method: HTTP方法
        path: 请求路径
        status_code: HTTP状态码
        duration_ms: 耗时（毫秒）
        extra: 额外信息
    """
    level = logging.WARNING if status_code >= 400 else logging.INFO
    logger.log(level, f"[请求] {method} {path} → {status_code} ({duration_ms:.0f}ms){' | ' + extra if extra else ''}")
