"""
ASR算法基类
定义统一的语音识别算法接口，对齐项目降噪模块BaseDenoiser设计模式
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import numpy as np


@dataclass
class ASRSegment:
    """单段识别结果"""
    start: float           # 起始时间(秒)
    end: float             # 结束时间(秒)
    text: str              # 识别文本
    confidence: Optional[float] = None  # 置信度(0-1)


@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str                                   # 完整识别文本
    language: str = "zh"                        # 识别语言
    confidence: Optional[float] = None          # 整体置信度(0-1)
    processing_time: float = 0.0                # 处理耗时(秒)
    rtf: Optional[float] = None                 # 实时因子(处理时间/音频时长)
    segments: Optional[List[ASRSegment]] = None # 分段结果
    algorithm_name: str = ""                    # 算法名称

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "processing_time": round(self.processing_time, 3),
            "rtf": round(self.rtf, 3) if self.rtf else None,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "confidence": s.confidence}
                for s in (self.segments or [])
            ],
            "algorithm_name": self.algorithm_name,
        }


class BaseASR(ABC):
    """
    ASR算法基类
    所有ASR算法都应继承此类并实现抽象方法
    """

    def __init__(
        self,
        name: str,
        sample_rate: int = 16000,
        device: str = "cuda",
        language: str = "zh",
        model_dir: Optional[str] = None,
        offline: bool = True,
    ):
        self.name = name
        self.sample_rate = sample_rate
        self.device = device
        self.language = language
        self.model_dir = model_dir
        self.offline = offline
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
    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        """
        识别音频

        Args:
            audio: 音频数据(numpy数组)
            sample_rate: 采样率(None则使用默认)

        Returns:
            ASRResult对象
        """
        pass

    # ── 流式转录接口（可选实现） ──

    def supports_streaming(self) -> bool:
        """是否支持流式转录"""
        return False

    def init_streaming_state(self, **kwargs) -> Any:
        """
        初始化流式转录状态

        Args:
            **kwargs: 算法特定参数，如 chunk_size_sec, unfixed_chunk_num 等

        Returns:
            流式状态对象
        """
        raise NotImplementedError(f"{self.name} 不支持流式转录")

    def streaming_transcribe(self, audio_chunk: np.ndarray, state: Any) -> dict:
        """
        流式转录：送入一个音频块，返回增量识别结果

        Args:
            audio_chunk: 音频块(numpy float32, 16kHz)
            state: 流式状态对象（由 init_streaming_state 返回）

        Returns:
            dict: {"text": str, "language": str, "is_final": bool}
        """
        raise NotImplementedError(f"{self.name} 不支持流式转录")

    def finish_streaming_transcribe(self, state: Any) -> dict:
        """
        结束流式转录，获取最终结果

        Args:
            state: 流式状态对象

        Returns:
            dict: {"text": str, "language": str, "is_final": True}
        """
        raise NotImplementedError(f"{self.name} 不支持流式转录")

    def transcribe_file(self, audio_path: str) -> ASRResult:
        """
        识别音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            ASRResult对象
        """
        import soundfile as sf
        import time

        audio, sr = sf.read(audio_path)
        # 转单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        # 重采样
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        start_time = time.time()
        result = self.transcribe(audio, sr)
        result.processing_time = time.time() - start_time
        audio_duration = len(audio) / sr
        result.rtf = result.processing_time / audio_duration if audio_duration > 0 else 0
        result.algorithm_name = self.name
        return result

    def is_initialized(self) -> bool:
        """检查模型是否已初始化"""
        return self._is_initialized

    def get_info(self) -> dict:
        """获取算法信息"""
        return {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "language": self.language,
            "initialized": self._is_initialized,
            "model_dir": self.model_dir,
        }

    def unload(self):
        """卸载模型释放显存"""
        import gc
        import torch

        self._model = None
        self._is_initialized = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
