"""
SpeechBrain语音增强算法
集成MetricGAN+和SepFormer等先进模型
"""

import numpy as np
import torch
import librosa
from typing import Optional
import time
import os
import logging

logger = logging.getLogger('audiomos')

from .base import BaseDenoiser, DenoiseResult
from .registry import DenoiserRegistry

# 项目根目录（绝对路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_SAVEDIR = os.path.join(_PROJECT_ROOT, "models", "speechbrain")


class SpeechBrainDenoiser(BaseDenoiser):
    """
    SpeechBrain语音增强基类
    使用SpeechBrain预训练模型进行语音增强
    """

    def __init__(self, name: str, model_name: str, sample_rate: int = 16000,
                 device: str = "cuda", savedir: str = _DEFAULT_SAVEDIR):
        """
        初始化SpeechBrain降噪器
        
        Args:
            name: 算法标识名
            model_name: SpeechBrain模型名称
            sample_rate: 采样率
            device: 计算设备
            savedir: 模型保存目录
        """
        super().__init__(name, sample_rate, device)
        self.model_name = model_name
        self.savedir = savedir
        self._separator = None
    
    def initialize(self) -> bool:
        """
        初始化SpeechBrain模型
        
        Returns:
            是否初始化成功
        """
        try:
            # SpeechBrain 1.0+ 使用 inference 模块
            try:
                from speechbrain.inference.separation import SepformerSeparation as Separator
            except ImportError:
                from speechbrain.pretrained import SepformerSeparation as Separator
            
            # 确保保存目录存在
            os.makedirs(self.savedir, exist_ok=True)
            
            # 加载模型
            self._separator = Separator.from_hparams(
                source=self.model_name,
                savedir=self.savedir,
                run_opts={"device": self.device}
            )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"SpeechBrain模型初始化失败: {e}")
            self._is_initialized = False
            return False
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """
        执行语音增强
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            
        Returns:
            DenoiseResult对象
        """
        start_time = time.time()
        
        if not self._is_initialized:
            self.initialize()
        
        # 重采样
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        try:
            # 转换为tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            
            # 执行分离/增强
            est_sources = self._separator.separate_batch(audio_tensor)
            
            # 取第一个源(增强后的语音)
            enhanced = est_sources[:, :, 0].squeeze().cpu().numpy()
            
            processing_time = time.time() - start_time
            
            return DenoiseResult(
                audio=enhanced,
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name
            )
            
        except Exception as e:
            logger.error(f"SpeechBrain增强失败: {e}")
            # 返回原始音频
            return DenoiseResult(
                audio=audio,
                sample_rate=self.sample_rate,
                processing_time=time.time() - start_time,
                algorithm_name=self.name
            )


class MetricGANDenoiser(BaseDenoiser):
    """
    MetricGAN+语音增强器
    针对感知指标优化的语音增强模型
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cuda",
                 savedir: str = _DEFAULT_SAVEDIR):
        """
        初始化MetricGAN+降噪器
        
        Args:
            sample_rate: 采样率
            device: 计算设备
            savedir: 模型保存目录
        """
        super().__init__("speechbrain_metricgan", sample_rate, device)
        self.savedir = savedir
        self.model_source = "speechbrain/metricgan-plus-voicebank"
        self._enhancer = None
    
    def initialize(self) -> bool:
        """初始化MetricGAN+模型"""
        try:
            # SpeechBrain 1.0+ 使用 inference 模块
            try:
                from speechbrain.inference.enhancement import SpectralMaskEnhancement
            except ImportError:
                from speechbrain.pretrained import SpectralMaskEnhancement
            
            os.makedirs(self.savedir, exist_ok=True)
            
            self._enhancer = SpectralMaskEnhancement.from_hparams(
                source=self.model_source,
                savedir=os.path.join(self.savedir, "metricgan-plus-voicebank"),
                run_opts={"device": self.device}
            )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"MetricGAN+模型初始化失败: {e}")
            self._is_initialized = False
            return False
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """执行增强"""
        start_time = time.time()
        
        if not self._is_initialized:
            self.initialize()
        
        # 重采样到16kHz (MetricGAN+要求)
        if sample_rate is not None and sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        try:
            # 转换为tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            
            # 增强
            enhanced = self._enhancer.enhance_batch(audio_tensor, lengths=torch.tensor([1.0]))
            enhanced = enhanced.squeeze().cpu().numpy()
            
            # 重采样回目标采样率
            if self.sample_rate != 16000:
                enhanced = librosa.resample(enhanced, orig_sr=16000, target_sr=self.sample_rate)
            
            processing_time = time.time() - start_time
            
            return DenoiseResult(
                audio=enhanced,
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name
            )
            
        except Exception as e:
            logger.error(f"MetricGAN+增强失败: {e}")
            return DenoiseResult(
                audio=audio,
                sample_rate=self.sample_rate,
                processing_time=time.time() - start_time,
                algorithm_name=self.name
            )


class SepFormerDenoiser(BaseDenoiser):
    """
    SepFormer语音分离/增强器
    基于Transformer的语音分离模型
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cuda",
                 savedir: str = _DEFAULT_SAVEDIR):
        """
        初始化SepFormer降噪器
        
        Args:
            sample_rate: 采样率
            device: 计算设备
            savedir: 模型保存目录
        """
        super().__init__("speechbrain_sepformer", sample_rate, device)
        self.savedir = savedir
        self.model_source = "speechbrain/sepformer-wham-enhancement"
        self._separator = None
    
    def initialize(self) -> bool:
        """初始化SepFormer模型"""
        try:
            # SpeechBrain 1.0+ 使用 inference 模块
            try:
                from speechbrain.inference.separation import SepformerSeparation
            except ImportError:
                from speechbrain.pretrained import SepformerSeparation
            
            os.makedirs(self.savedir, exist_ok=True)
            
            self._separator = SepformerSeparation.from_hparams(
                source=self.model_source,
                savedir=os.path.join(self.savedir, "sepformer-wham"),
                run_opts={"device": self.device}
            )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"SepFormer模型初始化失败: {e}")
            self._is_initialized = False
            return False
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """执行分离/增强"""
        start_time = time.time()
        
        if not self._is_initialized:
            self.initialize()
        
        # 重采样到8kHz (SepFormer要求)
        if sample_rate is not None and sample_rate != 8000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=8000)
        
        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        try:
            # 转换为tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

            # 分离
            est_sources = self._separator.separate_batch(audio_tensor)

            # 取第一个源(语音)
            enhanced = est_sources[:, :, 0].squeeze().cpu().numpy()

            # 重采样回目标采样率
            if self.sample_rate != 8000:
                enhanced = librosa.resample(enhanced, orig_sr=8000, target_sr=self.sample_rate)

            # 检查并防止削波失真
            peak = np.max(np.abs(enhanced))
            if peak > 1.0:
                # 如果峰值超过1.0，进行归一化
                enhanced = enhanced / peak * 0.95

            processing_time = time.time() - start_time

            return DenoiseResult(
                audio=enhanced,
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name
            )

        except Exception as e:
            logger.error(f"SepFormer增强失败: {e}")
            return DenoiseResult(
                audio=audio,
                sample_rate=self.sample_rate,
                processing_time=time.time() - start_time,
                algorithm_name=self.name
            )


# 注册SpeechBrain降噪算法
DenoiserRegistry.register("speechbrain_metricgan", MetricGANDenoiser)
DenoiserRegistry.register("speechbrain_sepformer", SepFormerDenoiser)
