"""
音频修复算法模块
包含去混响、超分辨率等音频修复功能

提供：
- BaseRestorer: 音频修复算法基类
- RestorationResult: 修复结果数据类
- RestorationRegistry: 算法注册表
- get_available_restorers: 获取可用算法列表
"""

from .base import BaseRestorer, RestorationResult, RestorationRegistry

# 尝试导入各算法实现
try:
    from .dereverberation import DereverbRestorer
    DEREVERB_AVAILABLE = True
except ImportError as e:
    DEREVERB_AVAILABLE = False
    DereverbRestorer = None

try:
    from .super_resolution import SuperResolutionRestorer
    SUPERRES_AVAILABLE = True
except ImportError as e:
    SUPERRES_AVAILABLE = False
    SuperResolutionRestorer = None


def get_available_restorers() -> dict:
    """获取所有可用的音频修复算法"""
    available = {}

    if DEREVERB_AVAILABLE:
        available["dereverberation"] = {
            "name": "去混响",
            "class": DereverbRestorer,
            "description": "使用深度学习模型去除音频中的混响效果",
        }

    if SUPERRES_AVAILABLE:
        available["super_resolution"] = {
            "name": "音频超分辨率",
            "class": SuperResolutionRestorer,
            "description": "将低采样率音频重建为高采样率（带宽扩展）",
        }

    return available


# 音频修复算法描述信息
RESTORATION_DESCRIPTIONS = {
    "dereverberation": {
        "name": "去混响 (Dereverberation)",
        "description": "使用WPE (Weighted Prediction Error) 算法去除音频中的混响效果，提升语音清晰度",
        "type": "信号处理+深度学习",
        "paper": "WPE: Weighted Prediction Error for Speech Dereverberation",
        "pros": ["有效减少混响", "盲去混响（无需参考）", "适用多种场景"],
        "cons": ["长混响场景效果有限", "计算复杂度中等"],
    },
    "super_resolution": {
        "name": "音频超分辨率 (Bandwidth Extension)",
        "description": "将低采样率音频（如8kHz）重建为高采样率（如16kHz/48kHz），扩展音频带宽",
        "type": "深度学习",
        "paper": "NVSR: Neural Voice Super Resolution",
        "pros": ["显著提升音频质量", "恢复高频细节", "真实现场录音效果显著"],
        "cons": ["模型较大", "推理速度受限", "极端低质量输入效果有限"],
    },
}


def get_restoration_description(name: str) -> dict:
    """获取音频修复算法的描述信息"""
    return RESTORATION_DESCRIPTIONS.get(name, {})


__all__ = [
    "BaseRestorer",
    "RestorationResult",
    "RestorationRegistry",
    "get_available_restorers",
    "get_restoration_description",
    "RESTORATION_DESCRIPTIONS",
    "DereverbRestorer",
    "SuperResolutionRestorer",
]
