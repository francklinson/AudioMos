"""
降噪算法注册表
管理所有可用的降噪算法，支持内置算法和自研插件发现。
"""

import os
import sys
import logging
from typing import Dict, Type, List, Optional

from .base import BaseDenoiser

logger = logging.getLogger(__name__)


class DenoiserRegistry:
    """降噪算法注册表"""

    _denoisers: Dict[str, Type[BaseDenoiser]] = {}
    _instances: Dict[str, BaseDenoiser] = {}
    _plugins_loaded: bool = False

    @classmethod
    def register(cls, name: str, denoiser_class: Type[BaseDenoiser]):
        """
        注册降噪算法

        Args:
            name: 算法名称
            denoiser_class: 降噪算法类
        """
        cls._denoisers[name] = denoiser_class
        logger.info(f"降噪算法已注册: {name} ({denoiser_class.__name__})")

    @classmethod
    def get(cls, name: str, **kwargs) -> Optional[BaseDenoiser]:
        """
        获取降噪算法实例

        Args:
            name: 算法名称
            **kwargs: 传递给降噪算法的参数

        Returns:
            降噪算法实例或None
        """
        # 检查是否已有实例
        cache_key = f"{name}_{kwargs.get('device', 'cuda')}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # 创建新实例
        if name in cls._denoisers:
            try:
                instance = cls._denoisers[name](**kwargs)
                cls._instances[cache_key] = instance
                return instance
            except Exception as e:
                logger.error(f"创建降噪算法实例失败 {name}: {e}")
                return None

        # 尝试从插件中发现
        if not cls._plugins_loaded:
            cls.discover_plugins()
            if name in cls._denoisers:
                return cls.get(name, **kwargs)

        return None

    @classmethod
    def list_denoisers(cls) -> List[str]:
        """
        列出所有已注册的算法名称

        Returns:
            算法名称列表
        """
        if not cls._plugins_loaded:
            cls.discover_plugins()
        return list(cls._denoisers.keys())

    @classmethod
    def get_info(cls, name: str) -> Optional[dict]:
        """
        获取算法信息

        Args:
            name: 算法名称

        Returns:
            算法信息字典或None
        """
        if name in cls._denoisers:
            return cls._denoisers[name].get_info()
        return None

    @classmethod
    def clear_instances(cls):
        """清除所有缓存的实例"""
        cls._instances.clear()

    @classmethod
    def discover_plugins(cls, plugin_dirs: Optional[List[str]] = None) -> List[str]:
        """
        自动发现并注册自研降噪算法插件

        扫描指定目录（默认: models/custom_denoisers/），
        加载符合插件规范的降噪算法并注册到系统。

        Args:
            plugin_dirs: 插件搜索目录列表（None=使用默认路径）

        Returns:
            新发现并注册的插件名称列表
        """
        cls._plugins_loaded = True
        discovered = []

        try:
            from .plugin_spec import PluginLoader
        except ImportError:
            logger.debug("插件加载器不可用")
            return discovered

        loader = PluginLoader(additional_paths=plugin_dirs) if plugin_dirs else PluginLoader()

        try:
            plugins = loader.discover_plugins()
        except Exception as e:
            logger.warning(f"插件发现失败: {e}")
            return discovered

        for plugin_name, plugin_class in plugins.items():
            if plugin_name not in cls._denoisers:
                cls.register(plugin_name, plugin_class)
                discovered.append(plugin_name)

                # 同时注册插件元数据到描述系统
                if hasattr(plugin_class, 'get_plugin_metadata'):
                    try:
                        meta = plugin_class.get_plugin_metadata()
                        DENOISER_DESCRIPTIONS[plugin_name] = {
                            "name": meta.name,
                            "description": meta.description,
                            "type": f"自研插件 ({meta.model_format})",
                            "paper": meta.paper_url or "",
                            "pros": meta.tags or [],
                            "cons": [],
                        }
                    except Exception:
                        pass

        if discovered:
            logger.info(f"发现并注册了 {len(discovered)} 个自研插件: {discovered}")

        return discovered

    @classmethod
    def load_plugin(cls, plugin_path: str) -> bool:
        """
        加载单个插件

        Args:
            plugin_path: 插件目录路径或插件名称

        Returns:
            是否加载成功
        """
        try:
            from .plugin_spec import PluginLoader
        except ImportError:
            logger.error("插件加载器不可用")
            return False

        loader = PluginLoader()
        plugin_class = loader.load_plugin(plugin_path)

        if plugin_class is None:
            logger.error(f"无法加载插件: {plugin_path}")
            return False

        if hasattr(plugin_class, 'get_plugin_metadata'):
            meta = plugin_class.get_plugin_metadata()
            cls.register(meta.name, plugin_class)
        else:
            cls.register(plugin_path, plugin_class)

        return True

    @classmethod
    def load_from_config(cls, config: dict) -> List[str]:
        """
        从配置文件加载插件

        Args:
            config: 配置字典，格式: {"plugin_dirs": ["path1", "path2"], "plugins": ["name1", "name2"]}

        Returns:
            加载的插件名称列表
        """
        loaded = []

        # 扫描目录
        plugin_dirs = config.get("plugin_dirs", [])
        if plugin_dirs:
            loaded.extend(cls.discover_plugins(plugin_dirs))

        # 加载指定插件
        for plugin_name in config.get("plugins", []):
            if cls.load_plugin(plugin_name):
                loaded.append(plugin_name)

        return loaded


def get_available_denoisers() -> List[dict]:
    """
    获取所有可用的降噪算法信息
    
    Returns:
        包含算法信息的字典列表
    """
    denoisers = []
    
    for name in DenoiserRegistry.list_denoisers():
        denoiser_class = DenoiserRegistry._denoisers.get(name)
        if denoiser_class:
            denoisers.append({
                "name": name,
                "description": denoiser_class.__doc__ or "",
                "class_name": denoiser_class.__name__
            })
    
    return denoisers


# 降噪算法描述信息
DENOISER_DESCRIPTIONS = {
    # ── ClearVoice-Studio 系列 (5个独立模型) ──
    "clearvoice_frcrn_se_16k": {
        "name": "ClearVoice FRCRN SE (16K)",
        "description": "阿里巴巴FRCRN实时语音增强模型，16kHz采样率，轻量高效，支持流式处理。已在ModelScope超过300万次调用",
        "type": "深度学习",
        "paper": "FRCRN: Boosting Feature Representation via Consecutive Recursive Networks",
        "pros": ["实时性好(支持流式)", "模型轻量", "16kHz实时处理", "300万+生产验证"],
        "cons": ["仅支持16kHz", "极端噪声场景效果有限"]
    },
    "clearvoice_mossformer2_se_48k": {
        "name": "ClearVoice MossFormer2 SE (48K)",
        "description": "MossFormer2架构48kHz高保真语音增强模型，最高降噪质量，适合专业音频处理。性能优于Resemble-enhance和DeepFilterNet",
        "type": "深度学习",
        "paper": "MossFormer2: Pushing the Performance Limit of Monaural Speech Separation and Enhancement",
        "pros": ["48kHz高保真", "最优降噪质量", "SOTA性能", "支持多种音频格式"],
        "cons": ["模型较大(48K)", "推理速度较慢", "需要较多GPU内存"]
    },
    "clearvoice_mossformer_gan_se_16k": {
        "name": "ClearVoice MossFormerGAN SE (16K)",
        "description": "基于GAN的MossFormer语音增强模型，VoiceBank+DEMAND上PESQ=3.47/STOI=0.96，DNS Challenge PESQ=3.57/STOI=0.98",
        "type": "深度学习",
        "paper": "MossFormerGAN: GAN-Enhanced MossFormer for Speech Enhancement",
        "pros": ["SOTA客观指标", "GAN增强语音自然度", "16kHz", "VoiceBank+DEMAND最优"],
        "cons": ["GAN训练不稳定风险", "推理速度中等"]
    },
    "clearvoice_mossformer2_ss_16k": {
        "name": "ClearVoice MossFormer2 SS (16K)",
        "description": "MossFormer2语音分离模型，WSJ0-2Mix上SI-SNRi=22.0dB/WHAM!上SI-SNRi=17.4dB/LRS2_2Mix上SI-SNRi=15.5dB",
        "type": "深度学习",
        "paper": "MossFormer: Pushing the Performance Limit of Monaural Speech Separation (INTERSPEECH 2022)",
        "pros": ["分离性能SOTA", "WSJ0-2Mix最优", "支持多说话人场景", "2人分离效果好"],
        "cons": ["仅支持2人分离", "计算资源需求高", "长音频需分段处理"]
    },
    "clearvoice_mossformer2_sr_48k": {
        "name": "ClearVoice MossFormer2 SR (48K)",
        "description": "MossFormer2语音超分辨率模型，将低采样率(16kHz)音频提升至48kHz高保真质量，恢复高频细节",
        "type": "深度学习",
        "paper": "ClearerVoice-Studio: Bridging Advanced Speech Processing Research and Practical Deployment (INTERSPEECH 2025)",
        "pros": ["16k→48k超分", "恢复高频细节", "提升听感质量", "联合降噪+超分"],
        "cons": ["需16kHz以上输入", "推理时间较长", "模型较大"]
    },
    # ── 向后兼容别名 ──
    "clearervoice_frcrn": {
        "name": "ClearVoice FRCRN SE (16K)",
        "description": "阿里巴巴FRCRN实时语音增强模型，16kHz采样率，轻量高效，支持流式处理",
        "type": "深度学习",
        "paper": "FRCRN: Boosting Feature Representation via Consecutive Recursive Networks",
        "pros": ["实时性好(支持流式)", "模型轻量", "16kHz实时处理"],
        "cons": ["仅支持16kHz", "极端噪声场景效果有限"]
    },
    "clearervoice_mossformer": {
        "name": "ClearVoice MossFormer2 SE (48K)",
        "description": "MossFormer2架构48kHz高保真语音增强模型，最高降噪质量",
        "type": "深度学习",
        "paper": "MossFormer2: Pushing the Performance Limit of Monaural Speech Separation and Enhancement",
        "pros": ["48kHz高保真", "最优降噪质量", "SOTA性能"],
        "cons": ["模型较大", "推理速度较慢"]
    },
    "clearervoice_mossformer2": {
        "name": "ClearVoice MossFormer2 SE (48K)",
        "description": "MossFormer2架构48kHz高保真语音增强模型，最高降噪质量",
        "type": "深度学习",
        "paper": "MossFormer2: Pushing the Performance Limit of Monaural Speech Separation and Enhancement",
        "pros": ["48kHz高保真", "最优降噪质量", "SOTA性能"],
        "cons": ["模型较大", "推理速度较慢"]
    },
    # ── SpeechBrain 系列 ──
    "speechbrain_metricgan": {
        "name": "SpeechBrain MetricGAN+",
        "description": "基于MetricGAN+的深度学习方法，在Voicebank-DEMAND数据集上达到PESQ 3.15分和STOI 93.0分",
        "type": "深度学习",
        "paper": "MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement",
        "pros": ["高质量语音增强", "针对感知指标优化"],
        "cons": ["计算复杂度较高"]
    },
    "speechbrain_sepformer": {
        "name": "SpeechBrain SepFormer",
        "description": "基于Transformer的语音分离/增强模型，在WHAM!数据集上训练",
        "type": "深度学习",
        "paper": "Attention is All You Need in Speech Separation",
        "pros": ["分离效果好", "适合多说话人场景"],
        "cons": ["模型较大", "推理速度较慢"]
    },
    # ── 传统方法 ──
    "spectral_subtraction": {
        "name": "谱减法 (Spectral Subtraction)",
        "description": "经典的信号处理降噪方法，基于噪声频谱估计",
        "type": "传统方法",
        "paper": "Suppression of Acoustic Noise in Speech Using Spectral Subtraction",
        "pros": ["计算简单", "实时性好", "无需训练"],
        "cons": ["会产生音乐噪声", "对非平稳噪声效果差"]
    },
    "wiener_filtering": {
        "name": "维纳滤波 (Wiener Filtering)",
        "description": "基于最小均方误差准则的最优线性滤波方法",
        "type": "传统方法",
        "paper": "Wiener Filtering of Speech",
        "pros": ["理论基础扎实", "计算效率高"],
        "cons": ["需要准确的噪声估计", "对非平稳噪声敏感"]
    },
    # ── 其他深度学习模型 ──
    "dccrn": {
        "name": "DCCRN (Deep Complex CRN)",
        "description": "基于复数卷积循环网络的深度学习语音增强算法，同时处理幅度和相位信息",
        "type": "深度学习",
        "paper": "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement (INTERSPEECH 2020)",
        "pros": ["复数域处理", "同时估计幅度和相位", "DNS Challenge优异表现"],
        "cons": ["模型较大", "训练数据需求高"]
    },
    "fullsubnet": {
        "name": "FullSubNet (Full-band Sub-band Fusion)",
        "description": "全带和子带融合网络的实时语音增强算法，结合全局频谱特征和局部频段精细处理",
        "type": "深度学习",
        "paper": "FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Speech Enhancement (ICASSP 2021)",
        "pros": ["实时处理", "全带+子带融合", "轻量高效"],
        "cons": ["极端噪声场景待优化", "需要16kHz输入"]
    }
}


def get_denoiser_description(name: str) -> dict:
    """
    获取降噪算法的详细描述
    
    Args:
        name: 算法名称
        
    Returns:
        算法描述字典
    """
    return DENOISER_DESCRIPTIONS.get(name, {
        "name": name,
        "description": "暂无描述",
        "type": "未知"
    })
