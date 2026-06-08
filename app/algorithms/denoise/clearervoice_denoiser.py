"""
ClearVoice-Studio 语音增强/分离/超分辨率算法适配器
基于阿里巴巴开源的 ClearerVoice-Studio 工具包 (clearvoice >= 0.1.0)

支持全部5个预训练模型:
  - FRCRN_SE_16K:         16kHz 实时语音增强 (轻量高效)
  - MossFormer2_SE_48K:   48kHz 语音增强 (最高质量)
  - MossFormerGAN_SE_16K: 16kHz GAN语音增强 (SOTA性能)
  - MossFormer2_SS_16K:   16kHz 语音分离 (多说话人)
  - MossFormer2_SR_48K:   48kHz 语音超分辨率 (16k→48k)

模型来源: HuggingFace alibabasglab/{model_name}
本地缓存: {project_root}/models/clearvoice/{model_name}/
"""

import numpy as np
import librosa
import os
import time
import tempfile
import soundfile as sf
from typing import Optional, Dict, Any
import logging

from .base import BaseDenoiser, DenoiseResult
from .registry import DenoiserRegistry

logger = logging.getLogger(__name__)

# ── 模型配置表 ──────────────────────────────────────────────────
# 每个模型的任务类型、采样率、显示信息

CLEARVOICE_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "clearvoice_frcrn_se_16k": {
        "task": "speech_enhancement",
        "model_name": "FRCRN_SE_16K",
        "sample_rate": 16000,
        "display_name": "ClearVoice FRCRN (16K)",
        "description": "FRCRN实时语音增强模型，16kHz采样率，轻量高效，支持流式处理",
        "checkpoint_subdir": "models/clearvoice/FRCRN_SE_16K",
    },
    "clearvoice_mossformer2_se_48k": {
        "task": "speech_enhancement",
        "model_name": "MossFormer2_SE_48K",
        "sample_rate": 48000,
        "display_name": "ClearVoice MossFormer2 SE (48K)",
        "description": "MossFormer2架构48kHz语音增强模型，最高降噪质量，适合专业音频处理",
        "checkpoint_subdir": "models/clearvoice/MossFormer2_SE_48K",
    },
    "clearvoice_mossformer_gan_se_16k": {
        "task": "speech_enhancement",
        "model_name": "MossFormerGAN_SE_16K",
        "sample_rate": 16000,
        "display_name": "ClearVoice MossFormerGAN (16K)",
        "description": "基于GAN的MossFormer语音增强模型，16kHz，VoiceBank+DEMAND上PESQ=3.47 SOTA性能",
        "checkpoint_subdir": "models/clearvoice/MossFormerGAN_SE_16K",
    },
    "clearvoice_mossformer2_ss_16k": {
        "task": "speech_separation",
        "model_name": "MossFormer2_SS_16K",
        "sample_rate": 16000,
        "display_name": "ClearVoice MossFormer2 SS (16K)",
        "description": "MossFormer2语音分离模型，16kHz，WSJ0-2Mix上SI-SNRi=22.0dB，支持2人分离",
        "checkpoint_subdir": "models/clearvoice/MossFormer2_SS_16K",
    },
    "clearvoice_mossformer2_sr_48k": {
        "task": "speech_super_resolution",
        "model_name": "MossFormer2_SR_48K",
        "sample_rate": 48000,
        "display_name": "ClearVoice MossFormer2 SR (48K)",
        "description": "MossFormer2语音超分辨率模型，将16kHz音频提升至48kHz高保真质量",
        "checkpoint_subdir": "models/clearvoice/MossFormer2_SR_48K",
    },
}


class ClearVoiceWrapperDenoiser(BaseDenoiser):
    """
    ClearVoice-Studio 统一降噪/增强/分离/超分适配器

    使用 clearvoice 包的原生 API，自动从 HuggingFace 下载预训练模型。
    支持 file-based 和 tensor-based 两种推理模式。

    用法:
        denoiser = ClearVoiceWrapperDenoiser(model_key="clearvoice_frcrn_se_16k")
        denoiser.initialize()
        result = denoiser.denoise(noisy_audio, sr=16000)
    """

    def __init__(
        self,
        model_key: str = "clearvoice_frcrn_se_16k",
        sample_rate: int = 16000,
        device: str = "cuda",
        model_dir: str = "./models/clearvoice",
    ):
        """
        初始化 ClearVoice 适配器

        Args:
            model_key: 模型标识符 (见 CLEARVOICE_MODEL_SPECS 的 key)
            sample_rate: 目标采样率（会被模型原生采样率覆盖）
            device: 计算设备 (cuda/cpu)
            model_dir: 模型下载根目录
        """
        spec = CLEARVOICE_MODEL_SPECS.get(model_key)
        if spec is None:
            raise ValueError(
                f"不支持的模型: {model_key}，"
                f"可用: {list(CLEARVOICE_MODEL_SPECS.keys())}"
            )

        self.model_key = model_key
        self._spec = spec
        self.model_dir = model_dir

        # 使用模型原生采样率
        native_sr = spec["sample_rate"]
        super().__init__(spec["model_name"], native_sr, device)

        self._cv_instance = None  # ClearVoice 实例（延迟初始化）
        self._model_loaded = False

    # ── BaseDenoiser 接口 ──────────────────────────────────────

    def initialize(self) -> bool:
        """
        初始化并下载/加载 ClearVoice 模型

        Returns:
            是否初始化成功
        """
        if self._is_initialized:
            return True

        try:
            from clearvoice import ClearVoice

            # 确保模型目录存在
            os.makedirs(self.model_dir, exist_ok=True)

            # 设置 checkpoint_dir 环境变量，让 clearvoice 找到正确的路径
            checkpoint_dir = os.path.join(
                self.model_dir, self._spec["model_name"]
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存当前工作目录并切换到项目根目录
            # (clearvoice 使用相对路径 checkpoint_dir，需要从项目根目录运行)
            original_cwd = os.getcwd()

            # 查找项目根目录（包含 models/clearvoice 目录的父目录）
            project_root = self._find_project_root()
            os.chdir(project_root)

            try:
                logger.info(
                    f"正在加载 ClearVoice 模型: {self._spec['model_name']} "
                    f"(task={self._spec['task']})"
                )
                self._cv_instance = ClearVoice(
                    task=self._spec["task"],
                    model_names=[self._spec["model_name"]],
                )
            finally:
                os.chdir(original_cwd)

            self._is_initialized = True
            self._model_loaded = True
            logger.info(
                f"✓ ClearVoice 模型加载成功: {self._spec['model_name']}"
            )
            return True

        except ImportError:
            logger.error(
                "clearvoice 包未安装。请运行: pip install clearvoice"
            )
            self._is_initialized = False
            return False
        except Exception as e:
            logger.error(
                f"ClearVoice 模型初始化失败 ({self._spec['model_name']}): {e}"
            )
            self._is_initialized = False
            return False

    def denoise(
        self, audio: np.ndarray, sample_rate: Optional[int] = None
    ) -> DenoiseResult:
        """
        执行语音处理（增强/分离/超分辨率）

        使用临时文件方式调用 ClearVoice 的 file-based API，
        这是最稳定可靠的方式。

        Args:
            audio: 输入音频 (numpy array, float32 in [-1, 1])
            sample_rate: 输入采样率

        Returns:
            DenoiseResult 对象
        """
        start_time = time.time()

        if not self._is_initialized:
            if not self.initialize():
                return DenoiseResult(
                    audio=audio,
                    sample_rate=sample_rate or self.sample_rate,
                    processing_time=time.time() - start_time,
                    algorithm_name=self.name,
                )

        target_sr = self._spec["sample_rate"]

        # ── 预处理: 重采样 + 单声道 ──
        if sample_rate is not None and sample_rate != target_sr:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=target_sr
            )

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 确保 float32 范围
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # ── 使用临时文件进行推理 ──
        temp_input = None
        temp_input_path = None

        try:
            temp_input = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            temp_input_path = temp_input.name
            temp_input.close()

            # 写入输入音频
            sf.write(temp_input_path, audio, target_sr)

            # 调用 ClearVoice 推理
            result = self._cv_instance(
                input_path=temp_input_path,
                online_write=False,
            )

            # 解析输出
            enhanced = self._parse_output(result, audio)

            # ── 后处理: 重采样回目标采样率 ──
            if self.sample_rate != target_sr:
                enhanced = librosa.resample(
                    enhanced, orig_sr=target_sr, target_sr=self.sample_rate
                )

            processing_time = time.time() - start_time

            return DenoiseResult(
                audio=enhanced.astype(np.float32),
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name,
            )

        except Exception as e:
            logger.error(
                f"ClearVoice 推理失败 ({self._spec['model_name']}): {e}"
            )
            return DenoiseResult(
                audio=audio,
                sample_rate=self.sample_rate,
                processing_time=time.time() - start_time,
                algorithm_name=self.name,
            )
        finally:
            # 清理临时文件
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.unlink(temp_input_path)
                except OSError:
                    pass

    # ── 辅助方法 ──────────────────────────────────────────────

    def _parse_output(
        self, result, fallback: np.ndarray
    ) -> np.ndarray:
        """
        解析 ClearVoice 输出结果

        ClearVoice 不同模型返回格式略有不同:
        - SE/SR 模型: 直接返回 numpy array (samples,)
        - SS 模型: 返回 list of numpy arrays (每个说话人)

        Args:
            result: ClearVoice 推理结果
            fallback: 解析失败时的回退值

        Returns:
            处理后的音频 numpy array
        """
        if result is None:
            return fallback

        # 语音分离模型返回多个说话人，取第一个（目标说话人）
        if isinstance(result, list):
            if len(result) > 0:
                audio = np.array(result[0], dtype=np.float32)
                return audio if audio.ndim <= 1 else audio[:, 0]
            return fallback

        # numpy 数组
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                return result.astype(np.float32)
            elif result.ndim == 2:
                # 多维数组取第一个通道
                return result[0].astype(np.float32) if result.shape[0] <= 2 else result[:, 0].astype(np.float32)

        # dict 格式
        if isinstance(result, dict):
            for key in ["output", "enhanced", "separated", "audio"]:
                if key in result:
                    val = result[key]
                    if isinstance(val, np.ndarray):
                        return val.astype(np.float32) if val.ndim == 1 else val[:, 0].astype(np.float32)
            # 取第一个 numpy 值
            for val in result.values():
                if isinstance(val, np.ndarray):
                    return val.astype(np.float32) if val.ndim == 1 else val[:, 0].astype(np.float32)

        return fallback

    def _find_project_root(self) -> str:
        """
        查找项目根目录

        从当前文件向上搜索包含 models/clearvoice 或 pyproject.toml 的目录
        """
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(10):
            parent = os.path.dirname(current)
            if parent == current:
                break
            # 检查标志文件
            for marker in ["models/clearvoice", "pyproject.toml", "setup.py", ".git"]:
                if os.path.exists(os.path.join(parent, marker)):
                    return parent
            current = parent
        return os.getcwd()

    def is_model_downloaded(self) -> bool:
        """检查模型是否已下载到本地"""
        checkpoint_dir = os.path.join(
            self.model_dir, self._spec["model_name"]
        )
        best_checkpoint = os.path.join(
            checkpoint_dir, "last_best_checkpoint"
        )
        return os.path.isfile(best_checkpoint)

    def get_info(self) -> dict:
        """获取算法信息"""
        info = super().get_info()
        info.update(
            {
                "model_key": self.model_key,
                "task": self._spec["task"],
                "display_name": self._spec["display_name"],
                "description": self._spec["description"],
                "native_sample_rate": self._spec["sample_rate"],
                "downloaded": self.is_model_downloaded(),
            }
        )
        return info


# ── 便捷子类：保持向后兼容的独立类名 ────────────────────────────


class FRCRNSE16KDenoiser(ClearVoiceWrapperDenoiser):
    """FRCRN 16kHz 语音增强"""

    def __init__(self, sample_rate=16000, device="cuda", model_dir="./models/clearvoice"):
        super().__init__(
            model_key="clearvoice_frcrn_se_16k",
            sample_rate=sample_rate,
            device=device,
            model_dir=model_dir,
        )


class MossFormer2SE48KDenoiser(ClearVoiceWrapperDenoiser):
    """MossFormer2 48kHz 语音增强"""

    def __init__(self, sample_rate=48000, device="cuda", model_dir="./models/clearvoice"):
        super().__init__(
            model_key="clearvoice_mossformer2_se_48k",
            sample_rate=sample_rate,
            device=device,
            model_dir=model_dir,
        )


class MossFormerGANSE16KDenoiser(ClearVoiceWrapperDenoiser):
    """MossFormerGAN 16kHz 语音增强"""

    def __init__(self, sample_rate=16000, device="cuda", model_dir="./models/clearvoice"):
        super().__init__(
            model_key="clearvoice_mossformer_gan_se_16k",
            sample_rate=sample_rate,
            device=device,
            model_dir=model_dir,
        )


class MossFormer2SS16KDenoiser(ClearVoiceWrapperDenoiser):
    """MossFormer2 16kHz 语音分离"""

    def __init__(self, sample_rate=16000, device="cuda", model_dir="./models/clearvoice"):
        super().__init__(
            model_key="clearvoice_mossformer2_ss_16k",
            sample_rate=sample_rate,
            device=device,
            model_dir=model_dir,
        )


class MossFormer2SR48KDenoiser(ClearVoiceWrapperDenoiser):
    """MossFormer2 48kHz 语音超分辨率"""

    def __init__(self, sample_rate=48000, device="cuda", model_dir="./models/clearvoice"):
        super().__init__(
            model_key="clearvoice_mossformer2_sr_48k",
            sample_rate=sample_rate,
            device=device,
            model_dir=model_dir,
        )


# ── 向后兼容：保留旧类名（重定向到新实现）───────────────────────


class FRCRNDenoiser(FRCRNSE16KDenoiser):
    """
    [兼容] FRCRN 降噪器 — 重定向到 FRCRNSE16KDenoiser
    保留此类名以确保现有代码不受影响
    """
    pass


class MossFormerDenoiser(MossFormer2SE48KDenoiser):
    """
    [兼容] MossFormer 降噪器 — 重定向到 MossFormer2SE48KDenoiser
    保留此类名以确保现有代码不受影响
    """
    pass


class MossFormer2Denoiser(MossFormer2SE48KDenoiser):
    """
    [兼容] MossFormer2 降噪器 — 重定向到 MossFormer2SE48KDenoiser
    保留此类名以确保现有代码不受影响
    """
    pass


# ── 注册所有算法 ──────────────────────────────────────────────

# 新注册（5个独立模型）
DenoiserRegistry.register("clearvoice_frcrn_se_16k", FRCRNSE16KDenoiser)
DenoiserRegistry.register("clearvoice_mossformer2_se_48k", MossFormer2SE48KDenoiser)
DenoiserRegistry.register("clearvoice_mossformer_gan_se_16k", MossFormerGANSE16KDenoiser)
DenoiserRegistry.register("clearvoice_mossformer2_ss_16k", MossFormer2SS16KDenoiser)
DenoiserRegistry.register("clearvoice_mossformer2_sr_48k", MossFormer2SR48KDenoiser)

# 向后兼容注册（指向新实现）
DenoiserRegistry.register("clearervoice_frcrn", FRCRNSE16KDenoiser)
DenoiserRegistry.register("clearervoice_mossformer", MossFormer2SE48KDenoiser)
DenoiserRegistry.register("clearervoice_mossformer2", MossFormer2SE48KDenoiser)
