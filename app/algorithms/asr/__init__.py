"""
ASR评测模块
集成9个前沿中文ASR开源SOTA算法，支持手动上传解析和Benchmark评测
"""

from .base import BaseASR, ASRResult, ASRSegment
from .registry import ASRRegistry, ASR_ALGORITHM_DESCRIPTIONS
from .evaluator import ASRMetrics, compute_cer, compute_wer, evaluate_asr
from .adapters import (
    ParaformerAdapter, SenseVoiceAdapter, WeNetAdapter, WhisperAdapter,
    FireRedASR2Adapter, Qwen3ASRAdapter, FunASRLLMAdapter,
    StepAudioAdapter, VibeVoiceAdapter,
)


def register_all_asr_algorithms():
    """注册所有ASR算法到注册表"""
    for name, desc in ASR_ALGORITHM_DESCRIPTIONS.items():
        adapter_class = _get_adapter_class(name)
        if adapter_class:
            ASRRegistry.register(name, adapter_class, desc)


def _get_adapter_class(name: str):
    """根据算法名获取适配器类"""
    adapter_map = {
        "paraformer-large": ParaformerAdapter,
        "sensevoice-small": SenseVoiceAdapter,
        "wenet-u2pp": WeNetAdapter,
        "whisper-large-v3-turbo": WhisperAdapter,
        "firered-asr2": FireRedASR2Adapter,
        "qwen3-asr": Qwen3ASRAdapter,
        "funasr-llm": FunASRLLMAdapter,
        "step-audio-2-mini": StepAudioAdapter,
        "vibevoice-asr": VibeVoiceAdapter,
    }
    return adapter_map.get(name)


def get_available_asr_algorithms() -> list:
    """获取所有可用的ASR算法列表"""
    return ASRRegistry.list_available()


# 模块加载时自动注册
register_all_asr_algorithms()
