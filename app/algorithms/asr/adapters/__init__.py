"""
ASR算法适配器模块
"""

from .funasr_adapter import ParaformerAdapter, SenseVoiceAdapter
from .wenet_adapter import WeNetAdapter
from .whisper_adapter import WhisperAdapter
from .firered_adapter import FireRedASR2Adapter
from .qwen3_asr_adapter import Qwen3ASRAdapter
from .funasr_llm_adapter import FunASRLLMAdapter
from .step_audio_adapter import StepAudioAdapter
from .vibevoice_adapter import VibeVoiceAdapter

__all__ = [
    "ParaformerAdapter",
    "SenseVoiceAdapter",
    "WeNetAdapter",
    "WhisperAdapter",
    "FireRedASR2Adapter",
    "Qwen3ASRAdapter",
    "FunASRLLMAdapter",
    "StepAudioAdapter",
    "VibeVoiceAdapter",
]
