"""
降噪算法 → 音频修复适配器

将所有已注册的降噪算法（BaseDenoiser）包装为音频修复接口（BaseRestorer），
使用户可以在"音频修复"Tab中直接选择任意降噪算法处理带噪音频。
"""

import numpy as np
from typing import Optional, Dict, Any

from .base import BaseRestorer, RestorationResult


class DenoiseRestorerAdapter(BaseRestorer):
    """
    降噪算法适配器

    将 BaseDenoiser 接口适配为 BaseRestorer 接口。
    denoise() → restore() 语义映射。

    用法:
        from denoise import DenoiserRegistry
        adapter = DenoiseRestorerAdapter(
            denoiser_name="clearvoice_frcrn_se_16k",
            sample_rate=16000,
            device="cuda",
        )
        adapter.initialize()
        result = adapter.restore(noisy_audio, sr=16000)
    """

    def __init__(
        self,
        denoiser_name: str = "spectral_subtraction",
        sample_rate: int = 16000,
        device: str = "cuda",
    ):
        """
        初始化适配器

        Args:
            denoiser_name: 降噪算法在 DenoiserRegistry 中的注册名
            sample_rate: 目标采样率
            device: 计算设备
        """
        super().__init__(denoiser_name, sample_rate, device)
        self._denoiser_name = denoiser_name
        self._denoiser = None

    def initialize(self) -> bool:
        """初始化底层降噪算法模型"""
        if self._is_initialized:
            return True

        try:
            from denoise import DenoiserRegistry

            self._denoiser = DenoiserRegistry.get(
                self._denoiser_name,
                sample_rate=self.sample_rate,
                device=self.device,
            )

            if self._denoiser is None:
                print(f"降噪算法 {self._denoiser_name} 不可用")
                return False

            if not self._denoiser.is_initialized():
                success = self._denoiser.initialize()
                if not success:
                    print(f"降噪算法 {self._denoiser_name} 初始化失败")
                    return False

            self._is_initialized = True
            return True

        except ImportError:
            print("降噪模块不可用")
            return False
        except Exception as e:
            print(f"适配器初始化失败 ({self._denoiser_name}): {e}")
            return False

    def restore(
        self, audio: np.ndarray, sample_rate: Optional[int] = None
    ) -> RestorationResult:
        """
        执行音频降噪修复

        将 restore() 调用委派给底层降噪算法的 denoise() 方法。

        Args:
            audio: 输入带噪音频
            sample_rate: 采样率

        Returns:
            RestorationResult 对象
        """
        if not self._is_initialized:
            if not self.initialize():
                return RestorationResult(
                    audio=audio,
                    sample_rate=sample_rate or self.sample_rate,
                    processing_time=0.0,
                    algorithm_name=self._denoiser_name,
                    metadata={"error": "算法初始化失败"},
                )

        # 委派给降噪算法
        denoise_result = self._denoiser.denoise(audio, sample_rate)

        # 提取元数据
        metadata: Dict[str, Any] = {"denoiser": self._denoiser_name}
        if denoise_result.snr_before is not None:
            metadata["snr_before"] = round(denoise_result.snr_before, 2)
        if denoise_result.snr_after is not None:
            metadata["snr_after"] = round(denoise_result.snr_after, 2)
        if denoise_result.noise_reduction_db is not None:
            metadata["noise_reduction_db"] = round(denoise_result.noise_reduction_db, 2)

        # 映射到 RestorationResult
        return RestorationResult(
            audio=denoise_result.audio,
            sample_rate=denoise_result.sample_rate,
            processing_time=denoise_result.processing_time,
            algorithm_name=denoise_result.algorithm_name,
            metadata=metadata,
        )

    def is_initialized(self) -> bool:
        return self._is_initialized

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["denoiser_name"] = self._denoiser_name
        info["denoiser_initialized"] = (
            self._denoiser.is_initialized()
            if self._denoiser
            else False
        )
        return info
