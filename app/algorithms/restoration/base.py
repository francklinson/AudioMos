"""
音频修复算法基类

定义音频修复（去混响、超分辨率等）的统一接口和数据类
"""

import time
import numpy as np
import soundfile as sf
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class RestorationResult:
    """音频修复结果数据类"""

    audio: np.ndarray  # 修复后的音频数据
    sample_rate: int  # 采样率
    processing_time: float  # 处理耗时（秒）
    algorithm_name: str  # 算法名称
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    @property
    def duration(self) -> float:
        """音频时长（秒）"""
        return len(self.audio) / self.sample_rate

    @property
    def rtf(self) -> float:
        """实时因子（Real-Time Factor）"""
        if self.duration > 0:
            return self.processing_time / self.duration
        return float("inf")


class BaseRestorer(ABC):
    """
    音频修复算法抽象基类

    所有音频修复算法（去混响、超分辨率等）需继承此类，
    并实现 initialize() 和 restore() 方法。
    """

    def __init__(self, name: str, sample_rate: int = 16000, device: str = "cuda"):
        """
        初始化修复器

        Args:
            name: 算法名称
            sample_rate: 目标采样率
            device: 计算设备 (cuda/cpu)
        """
        self.name = name
        self.sample_rate = sample_rate
        self.device = device
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化模型

        Returns:
            是否初始化成功
        """
        pass

    @abstractmethod
    def restore(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> RestorationResult:
        """
        执行音频修复

        Args:
            audio: 输入音频数据
            sample_rate: 输入采样率（None则使用默认）

        Returns:
            RestorationResult 对象
        """
        pass

    def restore_file(self, input_path: str, output_path: str) -> RestorationResult:
        """
        从文件读取音频并修复，保存结果

        Args:
            input_path: 输入音频文件路径
            output_path: 输出音频文件路径

        Returns:
            RestorationResult 对象
        """
        audio, sr = sf.read(input_path)
        result = self.restore(audio, sr)
        sf.write(output_path, result.audio, result.sample_rate)
        return result

    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._is_initialized

    def get_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "initialized": self._is_initialized,
        }


class RestorationRegistry:
    """
    音频修复算法注册表

    管理所有已注册的音频修复算法，支持按名称获取实例。
    """

    _algorithms: Dict[str, type] = {}
    _instances: Dict[str, BaseRestorer] = {}

    @classmethod
    def register(cls, name: str, algorithm_class: type):
        """
        注册算法

        Args:
            name: 算法标识名
            algorithm_class: 算法类（需继承BaseRestorer）
        """
        cls._algorithms[name] = algorithm_class

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseRestorer:
        """
        获取算法实例（带缓存）

        Args:
            name: 算法标识名
            **kwargs: 传递给构造函数

        Returns:
            算法实例
        """
        if name not in cls._algorithms:
            raise ValueError(f"未注册的算法: {name}")

        cache_key = f"{name}_{str(kwargs)}"
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls._algorithms[name](**kwargs)

        return cls._instances[cache_key]

    @classmethod
    def list_algorithms(cls) -> list:
        """列出所有已注册的算法名称"""
        return list(cls._algorithms.keys())

    @classmethod
    def clear_instances(cls):
        """清除实例缓存"""
        cls._instances.clear()


# 音频修复算法描述信息
RESTORATION_DESCRIPTIONS = {
    "dereverberation": {
        "name": "去混响",
        "description": "使用深度学习模型去除音频中的混响效果，提高语音清晰度",
        "type": "深度学习",
        "advantages": [
            "有效去除房间混响",
            "提升语音可懂度",
            "支持多种混响场景",
        ],
        "limitations": [
            "处理速度相对较慢",
            "极端混响场景效果有限",
            "需要GPU加速",
        ],
        "recommended_scenarios": [
            "会议室录音增强",
            "远场语音处理",
            "智能音箱前处理",
        ],
    },
    "super_resolution": {
        "name": "音频超分辨率",
        "description": "将低采样率音频重建为高采样率（带宽扩展），恢复高频成分",
        "type": "深度学习",
        "advantages": [
            "恢复丢失的高频信息",
            "提升音频整体质量",
            "适合老旧录音修复",
        ],
        "limitations": [
            "重建的高频可能不完美",
            "处理时间较长",
            "需要大量计算资源",
        ],
        "recommended_scenarios": [
            "老旧录音增强",
            "电话音频宽频扩展",
            "压缩音频还原",
        ],
    },
}


def get_restoration_description(name: str) -> Optional[Dict[str, Any]]:
    """获取算法描述信息"""
    return RESTORATION_DESCRIPTIONS.get(name)
