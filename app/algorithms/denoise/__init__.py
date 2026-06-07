"""
降噪算法模块
提供多种业界先进的音频降噪/增强算法

支持算法:
- 传统方法: 谱减法, 维纳滤波
- SpeechBrain: MetricGAN+, SepFormer
- ClearerVoice: FRCRN, MossFormer, MossFormer2
- DCCRN: 复数卷积循环网络
- FullSubNet: 全带子带融合网络
"""

from .base import BaseDenoiser, DenoiseResult
from .registry import DenoiserRegistry, get_available_denoisers, get_denoiser_description
from .speechbrain_denoiser import SpeechBrainDenoiser
from .traditional_denoiser import TraditionalDenoiser

try:
    from .clearervoice_denoiser import ClearerVoiceDenoiser
    CLEARERVOICE_AVAILABLE = True
except ImportError:
    CLEARERVOICE_AVAILABLE = False

try:
    from .dccrn_denoiser import DCCRNDenoiser
    DCCRN_AVAILABLE = True
except ImportError:
    DCCRN_AVAILABLE = False

try:
    from .fullsubnet_denoiser import FullSubNetDenoiser
    FULLSUBNET_AVAILABLE = True
except ImportError:
    FULLSUBNET_AVAILABLE = False

__all__ = [
    'BaseDenoiser',
    'DenoiseResult',
    'DenoiserRegistry',
    'get_available_denoisers',
    'get_denoiser_description',
    'SpeechBrainDenoiser',
    'TraditionalDenoiser',
]

if CLEARERVOICE_AVAILABLE:
    __all__.append('ClearerVoiceDenoiser')

if DCCRN_AVAILABLE:
    __all__.append('DCCRNDenoiser')

if FULLSUBNET_AVAILABLE:
    __all__.append('FullSubNetDenoiser')
