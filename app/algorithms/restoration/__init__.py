"""
音频修复算法模块
包含去混响、超分辨率、以及所有降噪算法的统一修复入口

提供：
- BaseRestorer: 音频修复算法基类
- RestorationResult: 修复结果数据类
- RestorationRegistry: 算法注册表
- get_available_restorers: 获取可用算法列表
- get_restoration_description: 获取算法描述
"""

from .base import BaseRestorer, RestorationResult, RestorationRegistry

# ── 原有修复算法 ──
try:
    from .dereverberation import DereverbRestorer, DereverbWienerRestorer
    DEREVERB_AVAILABLE = True
    import logging
    logging.getLogger('audiomos').info("[音频修复模块] DereverbRestorer / DereverbWienerRestorer 加载成功")
except ImportError as e:
    DEREVERB_AVAILABLE = False
    DereverbRestorer = None
    DereverbWienerRestorer = None
    import logging
    import traceback
    logging.getLogger('audiomos').warning(f"[音频修复模块] DereverbRestorer 加载失败: {e}")
    logging.getLogger('audiomos').debug(f"[音频修复模块] DereverbRestorer 错误详情: {traceback.format_exc()}")

try:
    from .super_resolution import SuperResolutionRestorer
    SUPERRES_AVAILABLE = True
    import logging
    logging.getLogger('audiomos').info("[音频修复模块] SuperResolutionRestorer 加载成功")
except ImportError as e:
    SUPERRES_AVAILABLE = False
    SuperResolutionRestorer = None
    import logging
    import traceback
    logging.getLogger('audiomos').warning(f"[音频修复模块] SuperResolutionRestorer 加载失败: {e}")
    logging.getLogger('audiomos').debug(f"[音频修复模块] SuperResolutionRestorer 错误详情: {traceback.format_exc()}")

# ── 降噪算法适配器 ──
try:
    from .denoise_adapter import DenoiseRestorerAdapter
    DENOISE_ADAPTER_AVAILABLE = True
    import logging
    logging.getLogger('audiomos').info("[音频修复模块] DenoiseRestorerAdapter 加载成功")
except ImportError as e:
    DENOISE_ADAPTER_AVAILABLE = False
    DenoiseRestorerAdapter = None
    import logging
    import traceback
    logging.getLogger('audiomos').error(f"[音频修复模块] DenoiseRestorerAdapter 加载失败: {e}")
    logging.getLogger('audiomos').error(f"[音频修复模块] 错误详情: {traceback.format_exc()}")


# ── 降噪算法 → 修复算法工厂 ──

def _make_denoise_restorer_class(denoiser_name: str, default_sr: int = 16000):
    """创建一个预配置了降噪算法名称的适配器类（工厂函数）"""
    class _ConfiguredDenoiseRestorer(DenoiseRestorerAdapter):
        def __init__(self, sample_rate=default_sr, device="cuda"):
            super().__init__(
                denoiser_name=denoiser_name,
                sample_rate=sample_rate,
                device=device,
            )
    # 设置类名以便调试
    _ConfiguredDenoiseRestorer.__name__ = f"DenoiseRestorer_{denoiser_name}"
    _ConfiguredDenoiseRestorer.__qualname__ = f"DenoiseRestorer_{denoiser_name}"
    return _ConfiguredDenoiseRestorer


# ── 降噪算法采样率配置 (名称 → {sample_rate}) ──
# ClearVoice 模型采样率从 CLEARVOICE_MODEL_SPECS 动态读取（单一数据源）
# 注意: clearvoice_mossformer2_sr_48k 是超分辨率模型，不作为降噪器暴露
#       超分能力通过原生 super_resolution 算法提供
_DENOISER_SR_CONFIG = {
    "speechbrain_metricgan": 16000,
    "speechbrain_sepformer": 16000,
    "spectral_subtraction": 16000,
    "wiener_filtering": 16000,
}

# 动态注入 ClearVoice 降噪模型（排除 SR 超分模型）
def _inject_clearvoice_models():
    try:
        from denoise.clearervoice_denoiser import CLEARVOICE_MODEL_SPECS
        for key, spec in CLEARVOICE_MODEL_SPECS.items():
            # 跳过超分辨率模型（通过原生 super_resolution 暴露）
            if spec.get("task") == "speech_super_resolution":
                continue
            _DENOISER_SR_CONFIG[key] = spec["sample_rate"]
    except ImportError:
        import logging
        logging.getLogger('audiomos').warning("[音频修复] 无法导入 CLEARVOICE_MODEL_SPECS，ClearVoice 降噪模型不可用")

_inject_clearvoice_models()


def get_available_restorers() -> dict:
    """获取所有可用的音频修复算法（原有修复 + 所有降噪算法）"""
    available = {}

    # ── 原有修复算法 ──
    if DEREVERB_AVAILABLE:
        available["dereverberation"] = {
            "name": "去混响",
            "class": DereverbRestorer,
            "description": "使用深度学习模型去除音频中的混响效果",
        }
        available["dereverberation_wiener"] = {
            "name": "去混响（传统）",
            "class": DereverbWienerRestorer,
            "description": "基于晚期混响抑制的传统信号处理方法，无模型加载，快速处理",
        }

    if SUPERRES_AVAILABLE:
        available["super_resolution"] = {
            "name": "音频超分辨率",
            "class": SuperResolutionRestorer,
            "description": "使用 MossFormer2_SR_48K 模型将低采样率音频重建为高采样率（带宽扩展）",
        }

    # ── 降噪算法（通过适配器，描述从 denoise 模块获取避免重复）──
    if DENOISE_ADAPTER_AVAILABLE:
        try:
            from denoise.registry import DENOISER_DESCRIPTIONS
            for key, sr in _DENOISER_SR_CONFIG.items():
                desc = DENOISER_DESCRIPTIONS.get(key, {})
                available[key] = {
                    "name": desc.get("name", key),
                    "class": _make_denoise_restorer_class(key, sr),
                    "description": desc.get("description", ""),
                }
        except ImportError:
            import logging
            logging.getLogger('audiomos').warning("[音频修复] 无法从 denoise 模块加载算法描述，降噪算法不可用")

    return available


# ── 算法描述（供前端展示）──
RESTORATION_DESCRIPTIONS: dict = {
    "dereverberation": {
        "name": "去混响 (Dereverberation)",
        "description": "使用 SpeechBrain SepFormer (WHAMR!) 深度学习模型去除房间混响效果，基于 Transformer 架构，可同时处理噪声和混响",
        "type": "深度学习",
        "paper": "Subakan et al., 'Attention is All You Need in Speech Separation' (ICASSP 2021)",
        "advantages": [
            "SpeechBrain SepFormer 架构",
            "WHAMR! 数据集训练",
            "联合降噪 + 去混响",
            "适合录音室/会议室录音",
        ],
        "limitations": [
            "模型较大",
            "推理速度较慢",
            "需重采样到 8kHz",
        ],
    },
    "dereverberation_wiener": {
        "name": "去混响-传统 (Wiener-style)",
        "description": "基于晚期混响抑制的传统信号处理方法，通过 STFT 时间平滑估计并减去晚期混响分量，无模型加载，快速处理",
        "type": "传统方法",
        "advantages": [
            "无需模型加载",
            "处理速度极快",
            "无采样率限制",
            "适合轻度混响场景",
        ],
        "limitations": [
            "对重度混响效果有限",
            "可能损伤早期反射",
            "非端到端学习",
        ],
    },
    "super_resolution": {
        "name": "音频超分辨率 (Bandwidth Extension)",
        "description": "使用 ClearVoice MossFormer2_SR_48K 模型（MossFormer2 + HiFiGAN 架构），将低采样率音频 (8k/16k/24k/32k) 重建为 48kHz 高保真音频，恢复高频细节",
        "type": "深度学习",
        "paper": "ClearerVoice-Studio: Bridging Advanced Speech Processing Research and Practical Deployment (INTERSPEECH 2025)",
        "advantages": [
            "MossFormer2 + HiFiGAN 架构",
            "支持 8k/16k/24k/32k → 48k 超分",
            "恢复高频细节",
            "提升听感质量",
        ],
        "limitations": [
            "模型较大",
            "推理时间较长",
            "需低于 48kHz 输入",
        ],
    },
}

# 合并降噪算法描述（从 denoise 模块引用，避免重复维护）
def _load_denoise_descriptions():
    """运行时从 denoise 模块加载降噪算法描述"""
    try:
        from denoise.registry import DENOISER_DESCRIPTIONS
        for key, info in DENOISER_DESCRIPTIONS.items():
            if key in _DENOISER_SR_CONFIG and key not in RESTORATION_DESCRIPTIONS:
                RESTORATION_DESCRIPTIONS[key] = {
                    "name": info.get("name", key),
                    "description": info.get("description", ""),
                    "type": info.get("type", "未知"),
                    "advantages": info.get("pros", []),
                    "limitations": info.get("cons", []),
                }
    except ImportError:
        pass

_load_denoise_descriptions()


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
    "DereverbWienerRestorer",
    "SuperResolutionRestorer",
    "DenoiseRestorerAdapter",
    "DENOISE_ADAPTER_AVAILABLE",
]
