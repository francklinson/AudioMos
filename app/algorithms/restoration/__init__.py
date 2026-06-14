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
    from .dereverberation import DereverbRestorer
    DEREVERB_AVAILABLE = True
    import logging
    logging.getLogger('audiomos').info("[音频修复模块] DereverbRestorer 加载成功")
except ImportError as e:
    DEREVERB_AVAILABLE = False
    DereverbRestorer = None
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


# ── 所有降噪算法配置 (名称 → 适配器类) ──
_DENOISE_ALGORITHMS = {
    # ClearVoice-Studio 系列
    "clearvoice_frcrn_se_16k": {
        "name": "ClearVoice FRCRN 降噪 (16K)",
        "description": "阿里巴巴FRCRN实时语音增强模型，16kHz，轻量高效，支持流式处理",
        "type": "深度学习",
        "advantages": [
            "实时处理，RTF仅0.086",
            "模型轻量仅154MB",
            "300万+生产环境验证",
            "支持流式处理",
        ],
        "limitations": [
            "仅支持16kHz输入",
            "极端噪声场景效果有限",
        ],
        "denoiser_name": "clearvoice_frcrn_se_16k",
        "sample_rate": 16000,
    },
    "clearvoice_mossformer2_se_48k": {
        "name": "ClearVoice MossFormer2 降噪 (48K)",
        "description": "MossFormer2架构48kHz高保真语音增强，最优降噪质量，性能优于Resemble-enhance和DeepFilterNet",
        "type": "深度学习",
        "advantages": [
            "48kHz高保真输出",
            "SOTA降噪性能",
            "支持多种音频格式",
            "专业音频处理级别",
        ],
        "limitations": [
            "模型较大(212MB)",
            "需要较多GPU内存",
            "推理速度较慢",
        ],
        "denoiser_name": "clearvoice_mossformer2_se_48k",
        "sample_rate": 48000,
    },
    "clearvoice_mossformer_gan_se_16k": {
        "name": "ClearVoice MossFormerGAN 降噪 (16K)",
        "description": "基于GAN的MossFormer语音增强，VoiceBank+DEMAND上PESQ=3.47/STOI=0.96，DNS Challenge PESQ=3.57",
        "type": "深度学习",
        "advantages": [
            "SOTA客观指标",
            "GAN增强语音自然度",
            "VoiceBank+DEMAND最优",
            "16kHz快速推理",
        ],
        "limitations": [
            "模型约131MB",
            "GAN推理有随机性",
        ],
        "denoiser_name": "clearvoice_mossformer_gan_se_16k",
        "sample_rate": 16000,
    },
    "clearvoice_mossformer2_ss_16k": {
        "name": "ClearVoice MossFormer2 语音分离 (16K)",
        "description": "MossFormer2语音分离模型，WSJ0-2Mix上SI-SNRi=22.0dB，支持2人语音分离",
        "type": "深度学习",
        "advantages": [
            "SOTA分离性能",
            "WSJ0-2Mix最优",
            "多说话人场景适用",
        ],
        "limitations": [
            "模型较大(640MB)",
            "仅支持2人分离",
            "长音频需分段处理",
        ],
        "denoiser_name": "clearvoice_mossformer2_ss_16k",
        "sample_rate": 16000,
    },
    "clearvoice_mossformer2_sr_48k": {
        "name": "ClearVoice MossFormer2 超分辨率 (48K)",
        "description": "MossFormer2语音超分辨率模型，16kHz→48kHz高保真重建，恢复高频细节",
        "type": "深度学习",
        "advantages": [
            "16k→48k超分",
            "恢复高频细节",
            "提升听感质量",
        ],
        "limitations": [
            "模型巨大(2.1GB)",
            "推理时间较长",
            "需16kHz以上输入",
        ],
        "denoiser_name": "clearvoice_mossformer2_sr_48k",
        "sample_rate": 48000,
    },
    # SpeechBrain 系列
    "speechbrain_metricgan": {
        "name": "SpeechBrain MetricGAN+ 降噪",
        "description": "基于MetricGAN+的深度学习方法，Voicebank-DEMAND数据集上PESQ=3.15/STOI=93.0%",
        "type": "深度学习",
        "advantages": [
            "高质量语音增强",
            "针对感知指标优化",
            "GAN训练策略",
        ],
        "limitations": [
            "计算复杂度较高",
            "依赖SpeechBrain库",
        ],
        "denoiser_name": "speechbrain_metricgan",
        "sample_rate": 16000,
    },
    "speechbrain_sepformer": {
        "name": "SpeechBrain SepFormer 语音分离",
        "description": "基于Transformer的语音分离/增强模型，WHAM!数据集训练",
        "type": "深度学习",
        "advantages": [
            "分离效果好",
            "适合多说话人场景",
            "Transformer架构",
        ],
        "limitations": [
            "模型较大",
            "推理速度较慢",
            "依赖SpeechBrain库",
        ],
        "denoiser_name": "speechbrain_sepformer",
        "sample_rate": 16000,
    },
    # 传统方法
    "spectral_subtraction": {
        "name": "谱减法降噪",
        "description": "经典信号处理降噪方法，基于噪声频谱估计和减法运算",
        "type": "传统方法",
        "advantages": [
            "计算极快，实时性最佳",
            "无需训练/下载模型",
            "内存占用极小",
            "适合嵌入式设备",
        ],
        "limitations": [
            "会产生音乐噪声",
            "对非平稳噪声效果差",
            "需要噪声估计",
        ],
        "denoiser_name": "spectral_subtraction",
        "sample_rate": 16000,
    },
    "wiener_filtering": {
        "name": "维纳滤波降噪",
        "description": "基于最小均方误差准则的最优线性滤波方法",
        "type": "传统方法",
        "advantages": [
            "理论基础扎实",
            "计算效率高",
            "无需训练",
            "平稳噪声效果好",
        ],
        "limitations": [
            "需准确噪声估计",
            "对非平稳噪声敏感",
            "降噪量有限",
        ],
        "denoiser_name": "wiener_filtering",
        "sample_rate": 16000,
    },
}


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

    if SUPERRES_AVAILABLE:
        available["super_resolution"] = {
            "name": "音频超分辨率",
            "class": SuperResolutionRestorer,
            "description": "将低采样率音频重建为高采样率（带宽扩展）",
        }

    # ── 降噪算法（通过适配器）──
    if DENOISE_ADAPTER_AVAILABLE:
        for key, config in _DENOISE_ALGORITHMS.items():
            available[key] = {
                "name": config["name"],
                "class": _make_denoise_restorer_class(
                    config["denoiser_name"],
                    config["sample_rate"],
                ),
                "description": config["description"],
            }

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
    "super_resolution": {
        "name": "音频超分辨率 (Bandwidth Extension)",
        "description": "使用 MossFormer2 语音超分辨率模型，将低采样率 (16kHz) 音频重建为高采样率 (48kHz)，恢复高频细节",
        "type": "深度学习",
        "paper": "ClearerVoice-Studio: Bridging Advanced Speech Processing Research and Practical Deployment (INTERSPEECH 2025)",
        "advantages": [
            "MossFormer2 架构",
            "16k→48k 超分",
            "恢复高频细节",
            "提升听感质量",
        ],
        "limitations": [
            "模型巨大 (2.1GB)",
            "推理时间较长",
            "需 16kHz 以上输入",
        ],
    },
}

# 合并降噪算法描述
for key, config in _DENOISE_ALGORITHMS.items():
    RESTORATION_DESCRIPTIONS[key] = {
        "name": config["name"],
        "description": config["description"],
        "type": config["type"],
        "advantages": config.get("advantages", []),
        "limitations": config.get("limitations", []),
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
    "DenoiseRestorerAdapter",
    "DENOISE_ADAPTER_AVAILABLE",
]
