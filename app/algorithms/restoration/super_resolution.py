"""
音频超分辨率（带宽扩展）算法

将低采样率音频重建为高采样率，恢复高频成分。

支持的算法：
1. 深度学习：ClearVoice MossFormer2_SR_48K（MossFormer2 + HiFiGAN）
   支持 8k/16k/24k/32k → 48k 带宽扩展
2. 传统方法：基于谐波外推的频带扩展（NUWaveSuperResolutionRestorer，内部 fallback）

参考文献：
- ClearerVoice-Studio: Bridging Advanced Speech Processing Research and Practical Deployment (INTERSPEECH 2025)
- Kumar et al., "NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling", 2021
"""

import numpy as np
import torch
import librosa
from typing import Optional
import time
import os
import sys
import logging

logger = logging.getLogger('audiomos')

from scipy import signal

from .base import BaseRestorer, RestorationResult, RestorationRegistry

# 项目根目录（绝对路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SuperResolutionRestorer(BaseRestorer):
    """
    深度学习音频超分辨率

    使用 ClearVoice MossFormer2_SR_48K 模型将低采样率音频重建为高采样率，
    恢复丢失的高频信息。支持 8k/16k/24k/32k → 48k 带宽扩展。

    内部委托给 ClearVoiceWrapperDenoiser（已实现 monkey patch 本地加载、
    临时文件推理、输出解析、音量归一化等完整逻辑）。
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        device: str = "cuda",
        model_dir: str = None,
    ):
        """
        初始化超分辨率修复器

        Args:
            sample_rate: 目标（输出）采样率，固定 48000
            device: 计算设备
            model_dir: 模型存储目录（默认 models/clearvoice）
        """
        super().__init__("super_resolution", sample_rate, device)
        if model_dir is None:
            model_dir = os.path.join(_PROJECT_ROOT, "models", "clearvoice")
        self.model_dir = model_dir
        self._cv_denoiser = None  # ClearVoiceWrapperDenoiser 实例

    def initialize(self) -> bool:
        """初始化 ClearVoice 超分辨率模型"""
        if self._is_initialized:
            return True

        try:
            # 延迟导入，确保 sys.path 已配置
            self._ensure_path()
            from denoise.clearervoice_denoiser import ClearVoiceWrapperDenoiser

            logger.info("[超分辨率] 创建 ClearVoiceWrapperDenoiser (model_key=clearvoice_mossformer2_sr_48k)")
            self._cv_denoiser = ClearVoiceWrapperDenoiser(
                model_key="clearvoice_mossformer2_sr_48k",
                sample_rate=48000,
                device=self.device,
                model_dir=self.model_dir,
            )

            success = self._cv_denoiser.initialize()
            if success:
                self._is_initialized = True
                logger.info("[超分辨率] ✓ MossFormer2_SR_48K 模型加载成功")
            else:
                logger.error("[超分辨率] ✗ MossFormer2_SR_48K 模型初始化返回 False")
            return success

        except ImportError as e:
            logger.error(f"[超分辨率] 无法导入 ClearVoiceWrapperDenoiser: {e}")
            return False
        except Exception as e:
            logger.error(f"[超分辨率] 模型初始化失败: {e}")
            import traceback
            logger.error(f"[超分辨率] 错误详情: {traceback.format_exc()}")
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
        original_sr = sample_rate or 16000

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

        if not self._is_initialized:
            if not self.initialize():
                # 初始化失败，回退到传统升采样
                logger.warning("[超分辨率] 模型初始化失败，回退到传统升采样")
                enhanced_np = self._classical_upsample(audio, original_sr)
                return RestorationResult(
                    audio=enhanced_np,
                    sample_rate=self.sample_rate,
                    processing_time=time.time() - start_time,
                    algorithm_name=self.name,
                    metadata={
                        "input_sr": original_sr,
                        "output_sr": self.sample_rate,
                        "method": "fallback_classical",
                        "error": "模型初始化失败",
                    },
                )

        try:
            # 委托给 ClearVoice 推理
            denoise_result = self._cv_denoiser.denoise(audio, original_sr)

            processing_time = time.time() - start_time

            return RestorationResult(
                audio=denoise_result.audio,
                sample_rate=denoise_result.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name,
                metadata={
                    "input_sr": original_sr,
                    "output_sr": denoise_result.sample_rate,
                    "upsampling_ratio": denoise_result.sample_rate / original_sr,
                    "method": "deep_learning_mossformer2_sr",
                    "model": "MossFormer2_SR_48K",
                },
            )

        except Exception as e:
            logger.warning(f"[超分辨率] ClearVoice 推理失败: {e}，回退到传统升采样")
            try:
                enhanced_np = self._classical_upsample(audio, original_sr)
                return RestorationResult(
                    audio=enhanced_np,
                    sample_rate=self.sample_rate,
                    processing_time=time.time() - start_time,
                    algorithm_name=self.name,
                    metadata={
                        "input_sr": original_sr,
                        "output_sr": self.sample_rate,
                        "method": "fallback_classical",
                        "error": str(e),
                    },
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
        传统升采样方法（含高频外推），作为深度学习失败时的 fallback

        分两步：
        1. 多项式插值升采样
        2. 频谱整形增强高频
        """
        ratio = self.sample_rate / original_sr
        target_len = int(len(audio) * ratio)
        enhanced = signal.resample(audio, target_len)

        nyquist_original = original_sr / 2
        nyquist_target = self.sample_rate / 2

        if nyquist_original < nyquist_target * 0.8:
            try:
                sos = signal.butter(4, nyquist_original * 0.9, btype="high", fs=self.sample_rate, output="sos")
                high_freq = signal.sosfilt(sos, enhanced)
                excitation_gain = 0.03
                enhanced = enhanced + excitation_gain * high_freq
            except Exception:
                pass

        max_val = np.max(np.abs(enhanced))
        if max_val > 0:
            enhanced = enhanced / max_val * 0.95

        return enhanced

    def _ensure_path(self):
        """确保 app/algorithms 目录在 sys.path 中"""
        algorithms_dir = os.path.join(_PROJECT_ROOT, "app", "algorithms")
        if algorithms_dir not in sys.path:
            sys.path.insert(0, algorithms_dir)


class NUWaveSuperResolutionRestorer(BaseRestorer):
    """
    基于谐波外推的传统带宽扩展（内部 fallback，不暴露给前端）

    使用 STFT 频带外推 + 谐波映射进行带宽扩展，
    适合作为深度学习模型不可用时的回退方案。
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        device: str = "cuda",
        model_dir: str = os.path.join(_PROJECT_ROOT, "models", "super_resolution"),
    ):
        super().__init__("super_resolution_classical", sample_rate, device)
        self.model_dir = model_dir

    def initialize(self) -> bool:
        """初始化（当前使用信号处理方案，无模型加载）"""
        os.makedirs(self.model_dir, exist_ok=True)
        self._is_initialized = True
        return True

    def restore(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> RestorationResult:
        """执行谐波外推带宽扩展"""
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
            n_fft = 1024
            hop_length = 256

            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            phase = np.angle(D)

            nyq_bin = n_fft // 2 + 1
            effective_bins = int(nyq_bin * original_sr / self.sample_rate)

            if effective_bins < nyq_bin:
                for bin_idx in range(effective_bins, nyq_bin):
                    harmonic_bin = bin_idx // 2
                    if harmonic_bin < effective_bins:
                        attenuation = 1.0 / ((bin_idx / effective_bins) ** 2)
                        magnitude[bin_idx, :] = magnitude[harmonic_bin, :] * attenuation
                        phase[bin_idx, :] = phase[harmonic_bin, :] * 2

            enhanced_D = magnitude * np.exp(1j * phase)
            enhanced = librosa.istft(enhanced_D, hop_length=hop_length)

            target_len = int(len(audio) * self.sample_rate / original_sr)
            enhanced = signal.resample(enhanced, target_len)

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
            logger.error(f"[超分辨率-传统] 谐波外推失败: {e}")
            return RestorationResult(
                audio=audio,
                sample_rate=original_sr,
                processing_time=time.time() - start_time,
                algorithm_name=self.name,
                metadata={"error": str(e)},
            )


# 注册算法
RestorationRegistry.register("super_resolution", SuperResolutionRestorer)
RestorationRegistry.register("super_resolution_classical", NUWaveSuperResolutionRestorer)
