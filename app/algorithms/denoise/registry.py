"""
降噪算法注册表
管理所有可用的降噪算法
"""

from typing import Dict, Type, List, Optional
from .base import BaseDenoiser


class DenoiserRegistry:
    """降噪算法注册表"""
    
    _denoisers: Dict[str, Type[BaseDenoiser]] = {}
    _instances: Dict[str, BaseDenoiser] = {}
    
    @classmethod
    def register(cls, name: str, denoiser_class: Type[BaseDenoiser]):
        """
        注册降噪算法
        
        Args:
            name: 算法名称
            denoiser_class: 降噪算法类
        """
        cls._denoisers[name] = denoiser_class
        
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
                print(f"创建降噪算法实例失败 {name}: {e}")
                return None
        return None
    
    @classmethod
    def list_denoisers(cls) -> List[str]:
        """
        列出所有已注册的算法名称
        
        Returns:
            算法名称列表
        """
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
    "clearervoice_frcrn": {
        "name": "ClearerVoice FRCRN",
        "description": "阿里巴巴开源的FRCRN模型，专注于实时语音增强",
        "type": "深度学习",
        "paper": "FRCRN: Boosting Feature Representation via Consecutive Recursive Networks",
        "pros": ["实时性好", "模型轻量"],
        "cons": ["极端噪声场景效果一般"]
    },
    "clearervoice_mossformer": {
        "name": "ClearerVoice MossFormer",
        "description": "基于MossFormer架构的语音增强模型，使用2.5百万次训练数据",
        "type": "深度学习",
        "paper": "MossFormer: Pushing the Performance Limit of Monaural Speech Separation",
        "pros": ["分离性能优异", "适合复杂场景"],
        "cons": ["计算资源需求高"]
    },
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
