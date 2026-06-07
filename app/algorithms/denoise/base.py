"""
降噪算法基类
定义统一的降噪算法接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DenoiseResult:
    """降噪结果数据类"""
    audio: np.ndarray  # 降噪后的音频数据
    sample_rate: int   # 采样率
    processing_time: float  # 处理耗时(秒)
    algorithm_name: str  # 算法名称
    
    # 可选的额外信息
    snr_before: Optional[float] = None  # 降噪前信噪比
    snr_after: Optional[float] = None   # 降噪后信噪比
    noise_reduction_db: Optional[float] = None  # 降噪量(dB)


class BaseDenoiser(ABC):
    """
    降噪算法基类
    所有降噪算法都应继承此类并实现抽象方法
    """
    
    def __init__(self, name: str, sample_rate: int = 16000, device: str = "cuda"):
        """
        初始化降噪器
        
        Args:
            name: 算法名称
            sample_rate: 目标采样率
            device: 计算设备 (cuda/cpu)
        """
        self.name = name
        self.sample_rate = sample_rate
        self.device = device
        self._model = None
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
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """
        执行降噪
        
        Args:
            audio: 输入音频数据
            sample_rate: 音频采样率(如果为None则使用默认采样率)
            
        Returns:
            DenoiseResult对象
        """
        pass
    
    def denoise_file(self, input_path: str, output_path: str) -> DenoiseResult:
        """
        对音频文件进行降噪
        
        Args:
            input_path: 输入音频文件路径
            output_path: 输出音频文件路径
            
        Returns:
            DenoiseResult对象
        """
        import soundfile as sf
        import time
        
        # 读取音频
        audio, sr = sf.read(input_path)
        
        # 执行降噪
        start_time = time.time()
        result = self.denoise(audio, sr)
        result.processing_time = time.time() - start_time
        
        # 保存结果
        sf.write(output_path, result.audio, result.sample_rate)
        
        return result
    
    def is_initialized(self) -> bool:
        """检查模型是否已初始化"""
        return self._is_initialized
    
    def get_info(self) -> dict:
        """
        获取算法信息
        
        Returns:
            包含算法信息的字典
        """
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "initialized": self._is_initialized
        }
