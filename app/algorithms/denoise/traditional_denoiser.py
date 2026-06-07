"""
传统信号处理降噪算法
包括谱减法、维纳滤波等经典方法
"""

import numpy as np
import librosa
from scipy import signal
from typing import Optional
import time

from .base import BaseDenoiser, DenoiseResult
from .registry import DenoiserRegistry


class SpectralSubtractionDenoiser(BaseDenoiser):
    """
    谱减法降噪器
    基于噪声频谱估计的经典降噪方法
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cuda",
                 n_fft: int = 2048, hop_length: int = 512,
                 noise_frames: int = 6, alpha: float = 1.0):
        """
        初始化谱减法降噪器
        
        Args:
            sample_rate: 采样率
            device: 设备(传统方法不使用GPU)
            n_fft: FFT窗口大小
            hop_length: 帧移
            noise_frames: 用于估计噪声的初始帧数
            alpha: 过减因子
        """
        super().__init__("spectral_subtraction", sample_rate, device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_frames = noise_frames
        self.alpha = alpha
    
    def initialize(self) -> bool:
        """初始化(传统方法无需加载模型)"""
        self._is_initialized = True
        return True
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """
        执行谱减法降噪
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            
        Returns:
            DenoiseResult对象
        """
        start_time = time.time()
        
        # 重采样到目标采样率
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # 确保音频为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 计算STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 估计噪声频谱(使用前noise_frames帧)
        noise_spectrum = np.mean(magnitude[:, :self.noise_frames], axis=1, keepdims=True)
        
        # 谱减法
        magnitude_denoised = np.maximum(
            magnitude - self.alpha * noise_spectrum,
            0.01 * magnitude  #  flooring to avoid negative values
        )
        
        # 重建信号
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, hop_length=self.hop_length, length=len(audio))
        
        processing_time = time.time() - start_time
        
        return DenoiseResult(
            audio=audio_denoised,
            sample_rate=self.sample_rate,
            processing_time=processing_time,
            algorithm_name=self.name
        )


class WienerFilterDenoiser(BaseDenoiser):
    """
    维纳滤波降噪器
    基于最小均方误差准则的最优线性滤波
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cuda",
                 n_fft: int = 2048, hop_length: int = 512,
                 noise_frames: int = 6):
        """
        初始化维纳滤波降噪器
        
        Args:
            sample_rate: 采样率
            device: 设备
            n_fft: FFT窗口大小
            hop_length: 帧移
            noise_frames: 用于估计噪声的初始帧数
        """
        super().__init__("wiener_filtering", sample_rate, device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_frames = noise_frames
    
    def initialize(self) -> bool:
        """初始化"""
        self._is_initialized = True
        return True
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """
        执行维纳滤波降噪
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            
        Returns:
            DenoiseResult对象
        """
        start_time = time.time()
        
        # 重采样
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 计算STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        power = magnitude ** 2
        
        # 估计噪声功率谱
        noise_power = np.mean(power[:, :self.noise_frames], axis=1, keepdims=True)
        
        # 估计信号功率谱(使用平滑)
        signal_power = np.maximum(power - noise_power, 0)
        
        # 计算维纳滤波增益
        # H = signal_power / (signal_power + noise_power)
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
        
        # 应用滤波
        magnitude_denoised = magnitude * wiener_gain
        
        # 重建信号
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, hop_length=self.hop_length, length=len(audio))
        
        processing_time = time.time() - start_time
        
        return DenoiseResult(
            audio=audio_denoised,
            sample_rate=self.sample_rate,
            processing_time=processing_time,
            algorithm_name=self.name
        )


class TraditionalDenoiser(BaseDenoiser):
    """
    传统降噪算法统一接口
    自动选择最佳传统方法
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cuda",
                 method: str = "wiener"):
        """
        初始化传统降噪器
        
        Args:
            sample_rate: 采样率
            device: 设备
            method: 方法选择 (wiener/spectral)
        """
        super().__init__(f"traditional_{method}", sample_rate, device)
        self.method = method
        self._denoiser = None
    
    def initialize(self) -> bool:
        """初始化"""
        if self.method == "wiener":
            self._denoiser = WienerFilterDenoiser(self.sample_rate, self.device)
        elif self.method == "spectral":
            self._denoiser = SpectralSubtractionDenoiser(self.sample_rate, self.device)
        else:
            raise ValueError(f"未知的方法: {self.method}")
        
        self._is_initialized = self._denoiser.initialize()
        return self._is_initialized
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """执行降噪"""
        if not self._is_initialized:
            self.initialize()
        
        return self._denoiser.denoise(audio, sample_rate)


# 注册传统降噪算法
DenoiserRegistry.register("spectral_subtraction", SpectralSubtractionDenoiser)
DenoiserRegistry.register("wiener_filtering", WienerFilterDenoiser)
DenoiserRegistry.register("traditional", TraditionalDenoiser)
