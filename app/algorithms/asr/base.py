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

    # ── 滑动窗口流式转录共享逻辑 ──

    @staticmethod
    def _find_text_overlap(prev_text: str, curr_text: str,
                           min_overlap: int = 3, max_lookback: int = 80) -> int:
        """
        找到 prev_text 尾部与 curr_text 头部的最长匹配长度。

        策略:
        1. 优先后缀-前缀精确匹配（O(max_lookback²)，但常数很小）
        2. 失败则回退到最长公共子串（prev 尾部 60 字符窗口 × curr 头部 60 字符窗口）

        Returns:
            curr_text 中与 prev_text 尾部重叠的字符数（即 curr_text 的偏移量）
        """
        if not prev_text or not curr_text:
            return 0

        # 策略1: 后缀-前缀精确匹配
        check_len = min(len(prev_text), len(curr_text), max_lookback)
        for n in range(check_len, min_overlap - 1, -1):
            if prev_text[-n:] == curr_text[:n]:
                return n

        # 策略2: 回退 — 最长公共子串（prev 尾部 × curr 头部）
        search_prev = prev_text[-60:] if len(prev_text) > 60 else prev_text
        search_curr = curr_text[:60] if len(curr_text) > 60 else curr_text
        best_offset = 0  # curr_text 中匹配起始位置 + 匹配长度

        for pl in range(len(search_prev)):
            for cl in range(len(search_curr)):
                match = 0
                while (pl + match < len(search_prev) and
                       cl + match < len(search_curr) and
                       search_prev[pl + match] == search_curr[cl + match]):
                    match += 1
                if match >= min_overlap:
                    # curr 中的绝对位置 = (curr 头部偏移) + cl + match
                    curr_abs_offset = cl + match
                    if curr_abs_offset > best_offset:
                        best_offset = curr_abs_offset

        return best_offset

    def _streaming_sliding_transcribe(self, audio_chunk: "np.ndarray",
                                       state: dict) -> dict:
        """
        滑动窗口流式转录：只推理最近 window_size_sec 秒音频，推理时间恒定。

        供模拟流式适配器（WeNet/Paraformer/Fun-ASR-Nano）的 streaming_transcribe 委托调用。

        要求 state 包含:
          - audio_buffer: np.ndarray (float32)
          - confirmed_text: str (已确认不会回退的文本)
          - last_window_text: str (上一窗口的转录结果)
          - chunk_size_sec: float
          - min_chunk_samples: int
          - window_size_sec: float (默认 5.0)
          - window_samples: int (= window_size_sec * sample_rate)
          - last_transcribe_buffer_len: int
        """
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # 追加 chunk
        state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_chunk])
        total_samples = len(state["audio_buffer"])
        window_samples = state["window_samples"]
        min_chunk = state["min_chunk_samples"]

        # 阶段1: 音频总量不足一个窗口 → 全量转录（与旧逻辑一致）
        if total_samples < window_samples:
            if total_samples < min_chunk:
                return {
                    "text": state.get("confirmed_text", "") or state.get("last_text", ""),
                    "language": self.language,
                    "is_final": False,
                }
            new_samples = total_samples - state.get("last_transcribe_buffer_len", 0)
            if new_samples < min_chunk:
                return {
                    "text": state.get("confirmed_text", "") or state.get("last_text", ""),
                    "language": self.language,
                    "is_final": False,
                }
            try:
                result = self.transcribe(state["audio_buffer"], self.sample_rate)
                text = result.text
                state["last_text"] = text
                state["confirmed_text"] = text
                state["last_window_text"] = text
                state["last_transcribe_buffer_len"] = total_samples
            except Exception as e:
                import logging
                logging.getLogger("audiomos").warning(
                    f"[{self.name}] 流式转录（阶段1）失败: {e}"
                )
            return {
                "text": state.get("confirmed_text", ""),
                "language": self.language,
                "is_final": False,
            }

        # 阶段2: 滑动窗口模式
        new_samples = total_samples - state.get("last_transcribe_buffer_len", 0)
        if new_samples < min_chunk:
            return {
                "text": state.get("confirmed_text", ""),
                "language": self.language,
                "is_final": False,
            }

        try:
            # 只取最后 window_samples 个样本推理
            window_audio = state["audio_buffer"][-window_samples:]
            result = self.transcribe(window_audio, self.sample_rate)
            new_window_text = result.text

            prev_window = state.get("last_window_text", "")
            if prev_window and new_window_text:
                overlap = self._find_text_overlap(prev_window, new_window_text)
                if overlap > 0:
                    new_part = new_window_text[overlap:]
                    if new_part:
                        state["confirmed_text"] = (state.get("confirmed_text", "") + new_part)
                else:
                    # 完全没有重叠，保守地追加新文本（用空格分隔）
                    if new_window_text != prev_window:
                        state["confirmed_text"] = (state.get("confirmed_text", "")
                                                   + new_window_text)
            else:
                # 首次进入窗口模式，confirmed_text 设为窗口结果
                # 但如果阶段1已经设置了 confirmed_text，保留它
                if not state.get("confirmed_text"):
                    state["confirmed_text"] = new_window_text

            state["last_window_text"] = new_window_text
            state["last_transcribe_buffer_len"] = total_samples

        except Exception as e:
            import logging
            logging.getLogger("audiomos").warning(
                f"[{self.name}] 滑动窗口转录失败: {e}"
            )

        return {
            "text": state.get("confirmed_text", ""),
            "language": self.language,
            "is_final": False,
        }
