"""
ASR算法适配器模块
"""

from .funasr_adapter import ParaformerAdapter, SenseVoiceAdapter
from .wenet_adapter import WeNetAdapter
from .whisper_adapter import WhisperAdapter

__all__ = [
    "ParaformerAdapter",
    "SenseVoiceAdapter",
    "WeNetAdapter",
    "WhisperAdapter",
]
