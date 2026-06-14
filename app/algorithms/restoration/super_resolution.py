"""
音频超分辨率（带宽扩展）算法

将低采样率音频重建为高采样率，恢复高频成分。

支持的算法：
1. 深度学习：基于SpeechBrain或HiFi-GAN的带宽扩展
2. 传统方法：基于频带外推的谱扩展

参考文献：
- Kumar et al., "NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling", 2021
- Lee et al., "Nu-Wave 2: A General Neural Audio Upsampling Tool", 2022
"""

import numpy as np
import torch
import librosa
from typing import Optional
import time
import os
import logging

logger = logging.getLogger('audiomos')

from scipy import signal

from .base import BaseRestorer, RestorationResult, RestorationRegistry

# 项目根目录（绝对路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "speechbrain")


class SuperResolutionRestorer(BaseRestorer):
    """
    深度学习音频超分辨率

    使用模型将低采样率音频重建为高采样率，
    恢复丢失的高频信息。
    优先级：SpeechBrain > 信号处理升频
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        device: str = "cuda",
        model_dir: str = _DEFAULT_MODEL_DIR,
        input_sample_rate: int = 8000,
    ):
        """
        初始化超分辨率修复器

        Args:
            sample_rate: 目标（输出）采样率
            device: 计算设备
            model_dir: 模型存储目录
            input_sample_rate: 输入采样率（用于模拟低采样率输入）
        """
        super().__init__("super_resolution", sample_rate, device)
        self.model_dir = model_dir
        self.input_sample_rate = input_sample_rate
        self._model = None

    def initialize(self) -> bool:
        """初始化超分模型"""
        try:
            # 优先使用SpeechBrain的增强模型（可一定程度恢复高频）
            from speechbrain.inference.enhancement import SpectralMaskEnhancement

            os.makedirs(self.model_dir, exist_ok=True)

            # 使用语音增强模型进行带宽扩展
            # MetricGAN+ 可用于一定程度的高频恢复
            self._model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir=os.path.join(self.model_dir, "metricgan-plus-voicebank"),
                run_opts={"device": self.device},
            )

            self._is_initialized = True
            return True

        except Exception as e:
            logger.error(f"超分辨率模型初始化失败: {e}")
            self._is_initialized = False
            return False

    def restore(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> RestorationResult:
        """
        执行音频超分辨率处理

        Args:
            audio: 输入音频（低采样率）
            sample_rate: 输入采样率

        Returns:
            RestorationResult 对象
        """
        start_time = time.time()

        original_sr = sample_rate or self.input_sample_rate

        # 如果输入采样率 >= 目标采样率，跳过超分
        if original_sr >= self.sample_rate:
            return RestorationResult(
                audio=audio,
                sample_rate=original_sr,
                processing_time=time.time() - start_time,
                algorithm_name=self.name,
                metadata={"message": f"输入采样率({original_sr}Hz)已达到或超过目标({self.sample_rate}Hz)，跳过处理"},
            )

        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        try:
            # 方案1: 使用model进行增强（如果有深度学习模型）
            if self._model is not None and self._is_initialized:
                # 先升采样到16kHz进行增强
                audio_16k = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)
                audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0)

                # 增强
                enhanced_16k = self._model.enhance_batch(audio_tensor, lengths=torch.tensor([1.0]))
                if isinstance(enhanced_16k, torch.Tensor):
                    enhanced_16k = enhanced_16k.squeeze(0).cpu().numpy()
                else:
                    enhanced_16k = enhanced_16k.squeeze(0).cpu().numpy()

                # 再升采样到目标采样率
                enhanced_np = librosa.resample(enhanced_16k, orig_sr=16000, target_sr=self.sample_rate)
            else:
                # 方案2: 传统频带外推
                enhanced_np = self._classical_upsample(audio, original_sr)

            processing_time = time.time() - start_time

            return RestorationResult(
                audio=enhanced_np,
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name,
                metadata={
                    "input_sr": original_sr,
                    "output_sr": self.sample_rate,
                    "upsampling_ratio": self.sample_rate / original_sr,
                    "method": "deep_learning" if self._model else "spectral_extrapolation",
                },
            )

        except Exception as e:
            # 回退到简单升采样
            logger.warning(f"超分辨率处理失败: {e}, 回退到传统升采样")
            try:
                enhanced_np = self._classical_upsample(audio, original_sr)
                return RestorationResult(
                    audio=enhanced_np,
                    sample_rate=self.sample_rate,
                    processing_time=time.time() - start_time,
                    algorithm_name=self.name,
                    metadata={"method": "fallback_upsampling", "error": str(e)},
                )
            except Exception as e2:
                return RestorationResult(
                    audio=audio,
                    sample_rate=original_sr,
                    processing_time=time.time() - start_time,
                    algorithm_name=self.name,
                    metadata={"error": str(e2)},
                )

    def _classical_upsample(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """
        传统升采样方法（含高频外推）

        分两步：
        1. 多项式插值升采样
        2. 频谱整形增强高频

        Args:
            audio: 输入音频
            original_sr: 原始采样率

        Returns:
            升采样后的音频
        """
        # 计算升采样比例
        ratio = self.sample_rate / original_sr

        # 使用scipy的resample进行高质量升采样
        target_len = int(len(audio) * ratio)
        enhanced = signal.resample(audio, target_len)

        # 高频增强：使用高通滤波添加人工高频分量
        # 设计高通滤波器（在原始奈奎斯特频率附近）
        nyquist_original = original_sr / 2
        nyquist_target = self.sample_rate / 2

        if nyquist_original < nyquist_target * 0.8:
            # 添加轻微的高频激励
            try:
                sos = signal.butter(4, nyquist_original * 0.9, btype="high", fs=self.sample_rate, output="sos")
                high_freq = signal.sosfilt(sos, enhanced)

                # 混合：原信号 + 少量高频增强
                excitation_gain = 0.03  # 高频激励增益（保守）
                enhanced = enhanced + excitation_gain * high_freq
            except Exception:
                pass  # 如果滤波失败，使用原始升采样结果

        # 归一化
        max_val = np.max(np.abs(enhanced))
        if max_val > 0:
            enhanced = enhanced / max_val * 0.95

        return enhanced


class NUWaveSuperResolutionRestorer(BaseRestorer):
    """
    基于NU-Wave风格的高质量带宽扩展

    使用更复杂的信号处理进行带宽扩展，
    适合对音频质量要求较高的场景。
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        device: str = "cuda",
        model_dir: str = os.path.join(_PROJECT_ROOT, "models", "super_resolution"),
    ):
        super().__init__("super_resolution_nuwave", sample_rate, device)
        self.model_dir = model_dir

    def initialize(self) -> bool:
        """初始化（当前使用信号处理方案）"""
        os.makedirs(self.model_dir, exist_ok=True)
        self._is_initialized = True
        return True

    def restore(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> RestorationResult:
        """执行高质量带宽扩展"""
        start_time = time.time()

        original_sr = sample_rate or 16000

        if original_sr >= self.sample_rate:
            return RestorationResult(
                audio=audio,
                sample_rate=original_sr,
                processing_time=time.time() - start_time,
                algorithm_name=self.name,
                metadata={"message": "输入采样率已达标"},
            )

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        try:
            # 多阶段升采样
            # 阶段1: 使用STFT进行频带外推
            n_fft = 1024
            hop_length = 256

            # 在当前采样率下计算STFT
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            phase = np.angle(D)

            # 频带外推：对高频部分进行谐波扩展
            nyq_bin = n_fft // 2 + 1
            # 计算当前采样率下有效的频率bin数
            effective_bins = int(nyq_bin * original_sr / self.sample_rate)

            if effective_bins < nyq_bin:
                # 使用低频谐波关系扩展高频
                for bin_idx in range(effective_bins, nyq_bin):
                    # 谐波映射：高频 = 低频的谐波衰减版本
                    harmonic_bin = bin_idx // 2
                    if harmonic_bin < effective_bins:
                        attenuation = 1.0 / ((bin_idx / effective_bins) ** 2)
                        magnitude[bin_idx, :] = magnitude[harmonic_bin, :] * attenuation
                        phase[bin_idx, :] = phase[harmonic_bin, :] * 2

            # 重建信号
            enhanced_D = magnitude * np.exp(1j * phase)
            enhanced = librosa.istft(enhanced_D, hop_length=hop_length)

            # 阶段2: 升采样到目标采样率
            target_len = int(len(audio) * self.sample_rate / original_sr)
            enhanced = signal.resample(enhanced, target_len)

            # 归一化
            max_val = np.max(np.abs(enhanced))
            if max_val > 0:
                enhanced = enhanced / max_val * 0.95

            processing_time = time.time() - start_time

            return RestorationResult(
                audio=enhanced,
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name,
                metadata={
                    "input_sr": original_sr,
                    "output_sr": self.sample_rate,
                    "method": "harmonic_extrapolation",
                },
            )

        except Exception as e:
            logger.error(f"NU-Wave超分处理失败: {e}")
            return RestorationResult(
                audio=audio,
                sample_rate=original_sr,
                processing_time=time.time() - start_time,
                algorithm_name=self.name,
                metadata={"error": str(e)},
            )


# 注册算法
RestorationRegistry.register("super_resolution", SuperResolutionRestorer)
RestorationRegistry.register("super_resolution_nuwave", NUWaveSuperResolutionRestorer)
