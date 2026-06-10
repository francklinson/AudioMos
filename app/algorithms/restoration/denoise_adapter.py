"""
降噪算法 → 音频修复适配器

将所有已注册的降噪算法（BaseDenoiser）包装为音频修复接口（BaseRestorer），
使用户可以在"音频修复"Tab中直接选择任意降噪算法处理带噪音频。
"""

import numpy as np
import os
import sys
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

    # 类级别的缓存，避免重复导入和初始化
    _DenoiserRegistry = None
    _import_attempted = False

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
            # 使用类级别的缓存，避免重复导入
            if DenoiseRestorerAdapter._DenoiserRegistry is None and not DenoiseRestorerAdapter._import_attempted:
                # 确保项目根目录在路径中
                project_root = self._find_project_root()
                print(f"[DenoiseAdapter] 项目根目录: {project_root}")
                print(f"[DenoiseAdapter] 当前 sys.path 前5项: {sys.path[:5]}")

                # 确保项目根目录在路径最前面
                if project_root in sys.path:
                    sys.path.remove(project_root)
                sys.path.insert(0, project_root)
                print(f"[DenoiseAdapter] 已添加项目根目录到 sys.path[0]")

                # 使用绝对导入
                print(f"[DenoiseAdapter] 尝试导入 DenoiserRegistry...")
                import importlib.util
                import importlib.machinery

                # 先尝试直接导入
                DenoiserRegistry = None
                try:
                    from app.algorithms.denoise import DenoiserRegistry
                    print(f"[DenoiseAdapter] 成功导入 DenoiserRegistry (方法1)")
                except ImportError as e1:
                    print(f"[DenoiseAdapter] 方法1导入失败: {e1}")
                    # 方法2：使用 importlib 动态导入
                    denoise_path = os.path.join(project_root, "app", "algorithms", "denoise", "__init__.py")
                    print(f"[DenoiseAdapter] 尝试从 {denoise_path} 导入")

                    if os.path.exists(denoise_path):
                        try:
                            spec = importlib.util.spec_from_file_location("app.algorithms.denoise", denoise_path)
                            denoise_module = importlib.util.module_from_spec(spec)
                            sys.modules["app.algorithms.denoise"] = denoise_module
                            spec.loader.exec_module(denoise_module)
                            DenoiserRegistry = denoise_module.DenoiserRegistry
                            print(f"[DenoiseAdapter] 成功导入 DenoiserRegistry (方法2)")
                        except Exception as e2:
                            print(f"[DenoiseAdapter] 方法2导入也失败: {e2}")
                            import traceback
                            traceback.print_exc()
                            DenoiseRestorerAdapter._import_attempted = True
                            return False
                    else:
                        print(f"[DenoiseAdapter] 找不到 denoise 模块: {denoise_path}")
                        DenoiseRestorerAdapter._import_attempted = True
                        return False

                # 缓存导入结果
                DenoiseRestorerAdapter._DenoiserRegistry = DenoiserRegistry
                DenoiseRestorerAdapter._import_attempted = True

            if DenoiseRestorerAdapter._DenoiserRegistry is None:
                print("[DenoiseAdapter] DenoiserRegistry 导入失败")
                return False

            # 使用缓存的 Registry
            self._denoiser = DenoiseRestorerAdapter._DenoiserRegistry.get(
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

        except ImportError as e:
            print(f"降噪模块不可用: {e}")
            return False
        except Exception as e:
            print(f"适配器初始化失败 ({self._denoiser_name}): {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_project_root(self) -> str:
        """查找项目根目录"""
        # 从当前文件开始向上查找
        current = os.path.dirname(os.path.abspath(__file__))
        print(f"[DenoiseAdapter] 当前文件目录: {current}")

        # 向上遍历目录树
        for _ in range(10):
            parent = os.path.dirname(current)
            if parent == current:
                break

            # 检查是否包含 app 目录和 backend 目录（项目根目录的标志）
            if os.path.isdir(os.path.join(parent, "app")) and os.path.isdir(os.path.join(parent, "backend")):
                print(f"[DenoiseAdapter] 找到项目根目录: {parent}")
                return parent

            # 备选：检查标志文件
            for marker in ["models/clearvoice", ".git"]:
                marker_path = os.path.join(parent, marker)
                if os.path.exists(marker_path):
                    print(f"[DenoiseAdapter] 通过标志 {marker} 找到项目根目录: {parent}")
                    return parent

            current = parent

        # 如果找不到，使用当前工作目录
        cwd = os.getcwd()
        print(f"[DenoiseAdapter] 未找到项目根目录，使用当前工作目录: {cwd}")
        return cwd

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
