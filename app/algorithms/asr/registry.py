"""
ASR算法注册表
管理所有可用的ASR算法，对齐DenoiserRegistry设计模式
"""

import os
import logging
from typing import Dict, Type, List, Optional

from .base import BaseASR

logger = logging.getLogger("audiomos")


class ASRRegistry:
    """ASR算法注册表"""

    _asrs: Dict[str, Type[BaseASR]] = {}
    _instances: Dict[str, BaseASR] = {}
    _descriptions: Dict[str, dict] = {}

    @classmethod
    def register(cls, name: str, asr_class: Type[BaseASR], description: Optional[dict] = None):
        """
        注册ASR算法

        Args:
            name: 算法名称
            asr_class: ASR算法类
            description: 算法描述信息
        """
        cls._asrs[name] = asr_class
        if description:
            cls._descriptions[name] = description
        logger.info(f"ASR算法已注册: {name} ({asr_class.__name__})")

    @classmethod
    def get(cls, name: str, **kwargs) -> Optional[BaseASR]:
        """
        获取ASR算法实例（带缓存）

        Args:
            name: 算法名称
            **kwargs: 传递给ASR算法的参数

        Returns:
            ASR算法实例或None
        """
        cache_key = f"{name}_{kwargs.get('device', 'cuda')}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        if name in cls._asrs:
            try:
                instance = cls._asrs[name](**kwargs)
                cls._instances[cache_key] = instance
                return instance
            except Exception as e:
                logger.error(f"创建ASR算法实例失败 {name}: {e}")
                return None
        else:
            logger.error(f"未注册的ASR算法: {name}")
            return None

    @classmethod
    def get_initialized(cls, name: str) -> Optional[BaseASR]:
        """获取已初始化的ASR算法实例"""
        instance = cls.get(name)
        if instance and instance.is_initialized():
            return instance
        return None

    @classmethod
    def list_available(cls) -> List[dict]:
        """列出所有可用的ASR算法"""
        result = []
        for name, asr_class in cls._asrs.items():
            desc = cls._descriptions.get(name, {})
            instance = cls._instances.get(f"{name}_cuda") or cls._instances.get(f"{name}_cpu")
            result.append({
                "name": name,
                "class": asr_class.__name__,
                "initialized": instance.is_initialized() if instance else False,
                "description": desc,
            })
        return result

    @classmethod
    def list_initialized(cls) -> List[str]:
        """列出所有已初始化的算法名称"""
        return [
            name for name, inst in cls._instances.items()
            if inst.is_initialized()
        ]

    @classmethod
    def initialize_all(cls, model_dir: str = "./models/asr", device: str = "cuda"):
        """初始化所有预加载的ASR算法"""
        for name, desc in cls._descriptions.items():
            if desc.get("preload", False):
                try:
                    instance = cls.get(name, device=device, model_dir=os.path.join(model_dir, name))
                    if instance:
                        instance.initialize()
                        logger.info(f"  ✅ ASR算法 {name} 初始化成功")
                except Exception as e:
                    logger.error(f"  ❌ ASR算法 {name} 初始化失败: {e}")

    @classmethod
    def unload(cls, name: str):
        """卸载指定算法"""
        for key in list(cls._instances.keys()):
            if key.startswith(f"{name}_"):
                instance = cls._instances.pop(key)
                instance.unload()
                logger.info(f"ASR算法 {name} 已卸载")


# ==================== 算法描述定义 ====================

ASR_ALGORITHM_DESCRIPTIONS = {
    "paraformer-large": {
        "display_name": "Paraformer-Large",
        "description": "非自回归端到端ASR模型，推理速度极快，中文工业级首选",
        "architecture": "非自回归(NAR) - 预测校验双通路",
        "params": "220M",
        "cer_aishell1": "1.95%",
        "streaming": True,
        "languages": ["zh", "en"],
        "license": "MIT",
        "preload": True,
        "tags": ["快速", "工业级", "流式"],
    },
    "sensevoice-small": {
        "display_name": "SenseVoice-Small",
        "description": "多任务语音基础模型，支持ASR+语种识别+情感识别+音频事件检测",
        "architecture": "多任务语音基础模型",
        "params": "234M",
        "cer_aishell1": "~3.0%",
        "streaming": False,
        "languages": ["zh", "en", "ja", "ko", "50+"],
        "license": "MIT",
        "preload": True,
        "tags": ["多任务", "情感识别", "轻量"],
    },
    "wenet-u2pp": {
        "display_name": "WeNet U2++",
        "description": "统一CTC/Attention混合模型，单模型同时支持流式和非流式识别",
        "architecture": "CTC/Attention混合 (Conformer)",
        "params": "~100M",
        "cer_aishell1": "~5.3%",
        "streaming": True,
        "languages": ["zh", "en"],
        "license": "Apache 2.0",
        "preload": True,
        "tags": ["流式", "生产部署", "C++运行时"],
    },
    "whisper-large-v3-turbo": {
        "display_name": "Whisper Large-v3 Turbo",
        "description": "OpenAI多语言ASR模型，支持99种语言，生态最成熟",
        "architecture": "Encoder-Decoder Transformer",
        "params": "809M",
        "cer_aishell1": "~5.14%",
        "streaming": False,
        "languages": ["zh", "en", "99种语言"],
        "license": "MIT",
        "preload": False,
        "tags": ["多语言", "参考基线", "生态成熟"],
    },
    "firered-asr2": {
        "display_name": "FireRedASR2-AED",
        "description": "高性能与效率平衡，支持20+方言和字级时间戳",
        "architecture": "Attention-based Encoder-Decoder",
        "params": "1.1B",
        "cer_aishell1": "3.05%",
        "streaming": False,
        "languages": ["zh", "en", "20+方言"],
        "license": "Apache 2.0",
        "preload": False,
        "tags": ["高效", "时间戳", "方言"],
    },
    "qwen3-asr": {
        "display_name": "Qwen3-ASR-1.7B",
        "description": "通义千问ASR，支持22种中文方言，统一流式/非流式架构",
        "architecture": "AuT音频编码器 + Qwen3 LLM",
        "params": "1.7B",
        "cer_aishell1": "~3.76%",
        "streaming": True,
        "languages": ["zh", "en", "52种语言"],
        "license": "Apache 2.0",
        "preload": False,
        "tags": ["方言", "流式", "千问"],
    },
    "funasr-llm": {
        "display_name": "Fun-ASR-Nano (800M)",
        "description": "轻量LLM-based ASR，支持31种语言+7种中文方言，低计算资源友好",
        "architecture": "音频编码器(0.2B) + Qwen3 LLM解码器(0.6B)",
        "params": "800M",
        "cer_aishell1": "~4.16%",
        "streaming": True,
        "languages": ["zh", "en", "31种语言"],
        "license": "MIT",
        "preload": False,
        "tags": ["轻量", "LLM", "多语言", "方言"],
    },
}
