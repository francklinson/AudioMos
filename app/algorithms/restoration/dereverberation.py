"""
音频去混响算法

使用深度学习模型去除音频中的混响效果。

支持的模型：
- SpeechBrain SepFormer (WHAMR!): speechbrain/sepformer-whamr-enhancement
  联合降噪+去混响，基于SepFormer架构

参考文献：
- Subakan et al., "Attention is All You Need in Speech Separation", ICASSP 2021
- Maciejewski et al., "WHAMR!: Noisy and Reverberant Single-Channel Speech Separation", ICASSP 2020
"""

import numpy as np
import torch
import librosa
from typing import Optional
import time
import os

from .base import BaseRestorer, RestorationResult, RestorationRegistry


class DereverbRestorer(BaseRestorer):
    """
    深度学习去混响器

    使用SpeechBrain预训练模型进行去混响处理。
    支持 WHAMR! 增强模型，可同时去除噪声和混响。
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        device: str = "cuda",
        model_dir: str = "./models/speechbrain",
        model_source: str = "speechbrain/sepformer-whamr-enhancement",
    ):
        """
        初始化去混响器

        Args:
            sample_rate: 目标采样率（WHAMR!模型使用8kHz）
            device: 计算设备
            model_dir: 模型存储目录
            model_source: 预训练模型名称
        """
        super().__init__("dereverberation", sample_rate, device)
        self.model_dir = model_dir
        self.model_source = model_source
        self._model = None

    def initialize(self) -> bool:
        """初始化去混响模型"""
        try:
            from speechbrain.inference.separation import SepformerSeparation

            os.makedirs(self.model_dir, exist_ok=True)

            # 使用Sepformer模型进行增强（包含去混响功能）
            self._model = SepformerSeparation.from_hparams(
                source=self.model_source,
                savedir=os.path.join(self.model_dir, "sepformer-whamr"),
                run_opts={"device": self.device},
            )

            self._is_initialized = True
            return True

        except Exception as e:
            print(f"去混响模型初始化失败: {e}")
            self._is_initialized = False
            return False

    def restore(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> RestorationResult:
        """
        执行去混响处理

        Args:
            audio: 输入音频（带混响）
            sample_rate: 输入采样率

        Returns:
            RestorationResult 对象
        """
        start_time = time.time()

        if not self._is_initialized:
            success = self.initialize()
            if not success:
                return RestorationResult(
                    audio=audio,
                    sample_rate=sample_rate or self.sample_rate,
                    processing_time=time.time() - start_time,
                    algorithm_name=self.name,
                    metadata={"error": "模型初始化失败"},
                )

        original_sr = sample_rate or self.sample_rate

        # 重采样到模型要求的采样率
        if original_sr != self.sample_rate:
            audio_resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
        else:
            audio_resampled = audio

        # 确保单声道
        if len(audio_resampled.shape) > 1:
            audio_resampled = np.mean(audio_resampled, axis=1)

        try:
            # 转换为tensor
            audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0)

            # 执行增强（分离语音和混响/噪声）
            with torch.no_grad():
                enhanced = self._model.separate_batch(audio_tensor)

            # 取第一个输出（增强后的语音）
            if isinstance(enhanced, torch.Tensor):
                enhanced_np = enhanced.squeeze(0).squeeze(0).cpu().numpy()
            else:
                enhanced_np = enhanced[0].squeeze(0).squeeze(0).cpu().numpy()

            # 重采样回原始采样率
            if original_sr != self.sample_rate:
                enhanced_np = librosa.resample(enhanced_np, orig_sr=self.sample_rate, target_sr=original_sr)

            processing_time = time.time() - start_time

            # 计算混响抑制量（简化估计）
            original_energy = np.mean(audio**2)
            enhanced_energy = np.mean(enhanced_np**2)
            reduction_db = 10 * np.log10(original_energy / (enhanced_energy + 1e-10))
            reduction_db = max(0, min(reduction_db, 30))

            return RestorationResult(
                audio=enhanced_np,
                sample_rate=original_sr,
                processing_time=processing_time,
                algorithm_name=self.name,
                metadata={
                    "model": self.model_source,
                    "model_sr": self.sample_rate,
                    "energy_reduction_db": round(reduction_db, 2),
                },
            )

        except Exception as e:
            print(f"去混响处理失败: {e}")
            return RestorationResult(
                audio=audio,
                sample_rate=original_sr,
                processing_time=time.time() - start_time,
                algorithm_name=self.name,
                metadata={"error": str(e)},
            )


class DereverbWienerRestorer(BaseRestorer):
    """
    传统信号处理去混响器

    使用谱减法变体进行简单的混响抑制。
    适用于轻度混响场景，计算速度快。
    """

    def __init__(self, sample_rate: int = 16000, device: str = "cuda"):
        super().__init__("dereverberation_wiener", sample_rate, device)
        self._reverb_tail_ms = 50  # 混响尾部长度的估计（ms）

    def initialize(self) -> bool:
        """无需模型加载"""
        self._is_initialized = True
        return True

    def restore(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> RestorationResult:
        """
        使用晚期混响抑制进行去混响

        基于倒谱域处理的简化方法，去除晚期混响分量。
        """
        start_time = time.time()

        original_sr = sample_rate or self.sample_rate

        # 重采样到目标采样率
        if original_sr != self.sample_rate:
            audio_resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
        else:
            audio_resampled = audio

        if len(audio_resampled.shape) > 1:
            audio_resampled = np.mean(audio_resampled, axis=1)

        try:
            # 计算STFT
            n_fft = 2048
            hop_length = 512
            D = librosa.stft(audio_resampled, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            phase = np.angle(D)

            # 晚期混响抑制：对每个频带应用平滑
            # 使用时间平滑来抑制晚期混响
            n_frames = magnitude.shape[1]
            reverb_tail_frames = int(self._reverb_tail_ms * self.sample_rate / 1000 / hop_length)
            reverb_tail_frames = max(1, min(reverb_tail_frames, n_frames // 3))

            # 计算每个频带的时间平滑版本（估计晚期混响）
            from scipy.ndimage import uniform_filter1d

            smoothed_mag = np.zeros_like(magnitude)
            for freq_bin in range(magnitude.shape[0]):
                smoothed_mag[freq_bin, :] = uniform_filter1d(magnitude[freq_bin, :], size=reverb_tail_frames)

            # 晚期混响 = 平滑版本
            # 早期信号 = 原始幅度 - 部分晚期混响
            suppression_factor = 0.7  # 混响抑制强度
            enhanced_mag = magnitude - suppression_factor * smoothed_mag
            enhanced_mag = np.maximum(enhanced_mag, magnitude * 0.01)  # 保留至少1%

            # 重建信号
            enhanced_D = enhanced_mag * np.exp(1j * phase)
            enhanced_np = librosa.istft(enhanced_D, hop_length=hop_length, length=len(audio_resampled))

            # 重采样回原始采样率
            if original_sr != self.sample_rate:
                enhanced_np = librosa.resample(enhanced_np, orig_sr=self.sample_rate, target_sr=original_sr)

            processing_time = time.time() - start_time

            return RestorationResult(
                audio=enhanced_np,
                sample_rate=original_sr,
                processing_time=processing_time,
                algorithm_name=self.name,
                metadata={"method": "late_reverb_suppression", "reverb_tail_ms": self._reverb_tail_ms},
            )

        except Exception as e:
            print(f"传统去混响处理失败: {e}")
            return RestorationResult(
                audio=audio,
                sample_rate=original_sr,
                processing_time=time.time() - start_time,
                algorithm_name=self.name,
                metadata={"error": str(e)},
            )


# 注册算法
RestorationRegistry.register("dereverberation", DereverbRestorer)
RestorationRegistry.register("dereverberation_wiener", DereverbWienerRestorer)
