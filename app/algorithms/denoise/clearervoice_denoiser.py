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

from .base import BaseDenoiser, DenoiseResult
from .registry import DenoiserRegistry

# 使用统一的 logger
import logging
logger = logging.getLogger("audiomos")

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

            # 保存当前工作目录
            original_cwd = os.getcwd()
            logger.info(f"[模型初始化] 当前工作目录: {original_cwd}")

            # 查找项目根目录（包含 models/clearvoice 目录的父目录）
            project_root = self._find_project_root()
            logger.info(f"[模型初始化] 项目根目录: {project_root}")
            
            # 设置 checkpoint_dir 为本地模型路径（使用绝对路径）
            # ClearVoice 默认使用 checkpoints/模型名，我们需要指向 models/clearvoice/模型名
            checkpoint_dir = os.path.join(
                project_root, self.model_dir, self._spec["model_name"]
            )
            logger.info(f"[模型初始化] 模型目录: {checkpoint_dir}")
            
            # 检查本地模型是否存在
            last_best_file = os.path.join(checkpoint_dir, 'last_best_checkpoint')
            logger.info(f"[模型初始化] 检查模型文件: {last_best_file}")
            
            if not os.path.isfile(last_best_file):
                logger.error(
                    f"[模型初始化] 本地模型文件不存在: {last_best_file}\n"
                    f"[模型初始化] 请确保模型已下载到 {checkpoint_dir}"
                )
                # 列出模型目录内容以便调试
                if os.path.isdir(checkpoint_dir):
                    logger.info(f"[模型初始化] 目录 {checkpoint_dir} 存在，内容:")
                    for item in os.listdir(checkpoint_dir):
                        logger.info(f"  - {item}")
                else:
                    logger.error(f"[模型初始化] 目录 {checkpoint_dir} 不存在")
                    # 检查父目录
                    parent_dir = os.path.dirname(checkpoint_dir)
                    if os.path.isdir(parent_dir):
                        logger.info(f"[模型初始化] 父目录 {parent_dir} 存在，内容:")
                        for item in os.listdir(parent_dir):
                            logger.info(f"  - {item}")
                return False
            
            logger.info(f"[模型初始化] 模型文件存在: {last_best_file}")
            
            # 读取 checkpoint 文件名
            with open(last_best_file, 'r') as f:
                checkpoint_name = f.readline().strip()
            logger.info(f"[模型初始化] Checkpoint 文件名: {checkpoint_name}")
            
            # 切换到项目根目录
            # (clearvoice 使用相对路径 checkpoint_dir，需要从项目根目录运行)
            os.chdir(project_root)
            logger.info(f"[模型初始化] 切换到项目根目录: {project_root}")

            # 设置环境变量禁用 HuggingFace Hub 网络访问
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            logger.info("[模型初始化] 已设置离线模式环境变量")

            try:
                logger.info(
                    f"[模型初始化] 开始加载 ClearVoice 模型: {self._spec['model_name']} "
                    f"(task={self._spec['task']})"
                )
                
                # 使用 monkey patch 修改 load_model 方法，强制使用本地模型
                import clearvoice.networks as cv_networks
                original_load_model = cv_networks.SpeechModel.load_model
                logger.info("[模型初始化] 已保存原始 load_model 方法")
                
                def patched_load_model(self):
                    """修改后的 load_model，使用本地模型路径"""
                    import os
                    import torch.nn as nn
                    
                    logger.info(f"[模型初始化] 进入 patched_load_model，模型名称: {self.name}")
                    
                    # 使用本地模型路径
                    self.args.checkpoint_dir = checkpoint_dir
                    logger.info(f"[模型初始化] 设置 checkpoint_dir: {checkpoint_dir}")
                    
                    # 检查本地模型是否存在
                    best_name = os.path.join(self.args.checkpoint_dir, 'last_best_checkpoint')
                    logger.info(f"[模型初始化] 检查 checkpoint 文件: {best_name}")
                    
                    if not os.path.isfile(best_name):
                        logger.error(f"[模型初始化] 本地模型文件不存在: {best_name}")
                        return
                    
                    logger.info(f"[模型初始化] 找到 checkpoint 文件，开始加载模型权重...")
                    
                    # 调用原始的 _load_model 逻辑（跳过 download_model）
                    if isinstance(self.model, nn.ModuleList):
                        logger.info("[模型初始化] 检测到 ModuleList 模型结构")
                        with open(best_name, 'r') as f:
                            model_name = f.readline().strip()
                            checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                            logger.info(f"[模型初始化] 加载第一个模型: {checkpoint_path}")
                            self._load_model(self.model[0], checkpoint_path, model_key='mossformer')
                            
                            model_name = f.readline().strip()
                            checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                            logger.info(f"[模型初始化] 加载第二个模型: {checkpoint_path}")
                            self._load_model(self.model[1], checkpoint_path, model_key='generator')
                    else:
                        with open(best_name, 'r') as f:
                            model_name = f.readline().strip()
                        checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                        logger.info(f"[模型初始化] 加载模型权重: {checkpoint_path}")
                        self._load_model(self.model, checkpoint_path, model_key='model')
                    
                    logger.info("[模型初始化] 模型权重加载完成")
                
                # 应用 monkey patch
                cv_networks.SpeechModel.load_model = patched_load_model
                logger.info("[模型初始化] 已应用 patched_load_model")
                
                try:
                    # 创建 ClearVoice 实例（会使用 patched_load_model）
                    logger.info("[模型初始化] 创建 ClearVoice 实例...")
                    self._cv_instance = ClearVoice(
                        task=self._spec["task"],
                        model_names=[self._spec["model_name"]],
                    )
                    logger.info("[模型初始化] ClearVoice 实例创建完成")
                finally:
                    # 恢复原始方法
                    cv_networks.SpeechModel.load_model = original_load_model
                    logger.info("[模型初始化] 已恢复原始 load_model 方法")
                    
            finally:
                os.chdir(original_cwd)
                logger.info(f"[模型初始化] 恢复工作目录: {original_cwd}")

            self._is_initialized = True
            self._model_loaded = True
            logger.info(
                f"[模型初始化] ✓ ClearVoice 模型加载成功: {self._spec['model_name']}"
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
        logger.info(f"[降噪处理] 开始处理音频，输入采样率: {sample_rate}, 目标采样率: {self._spec['sample_rate']}")
        logger.info(f"[降噪处理] 输入音频形状: {audio.shape}, 数据类型: {audio.dtype}")

        if not self._is_initialized:
            logger.info("[降噪处理] 模型未初始化，开始初始化...")
            if not self.initialize():
                logger.error("[降噪处理] 模型初始化失败")
                return DenoiseResult(
                    audio=audio,
                    sample_rate=sample_rate or self.sample_rate,
                    processing_time=time.time() - start_time,
                    algorithm_name=self.name,
                )

        target_sr = self._spec["sample_rate"]

        # ── 预处理: 重采样 + 单声道 ──
        if sample_rate is not None and sample_rate != target_sr:
            logger.info(f"[降噪处理] 重采样: {sample_rate} Hz -> {target_sr} Hz")
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=target_sr
            )
            logger.info(f"[降噪处理] 重采样后音频形状: {audio.shape}")

        if len(audio.shape) > 1:
            logger.info(f"[降噪处理] 转换为单声道，原形状: {audio.shape}")
            audio = np.mean(audio, axis=1)
            logger.info(f"[降噪处理] 转换后形状: {audio.shape}")

        # 确保 float32 范围
        if audio.dtype != np.float32:
            logger.info(f"[降噪处理] 转换数据类型: {audio.dtype} -> float32")
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
            logger.info(f"[降噪处理] 创建临时输入文件: {temp_input_path}")

            # 写入输入音频
            sf.write(temp_input_path, audio, target_sr)
            logger.info(f"[降噪处理] 已写入临时文件，采样率: {target_sr} Hz")

            # 调用 ClearVoice 推理
            logger.info("[降噪处理] 调用 ClearVoice 推理...")
            result = self._cv_instance(
                input_path=temp_input_path,
                online_write=False,
            )
            logger.info("[降噪处理] ClearVoice 推理完成")

            # 解析输出
            logger.info("[降噪处理] 解析输出结果...")
            enhanced = self._parse_output(result, audio)
            logger.info(f"[降噪处理] 解析后音频形状: {enhanced.shape}")

            # 后处理：重采样回目标采样率 + 音量归一化
            if self.sample_rate != target_sr:
                logger.info(f"[降噪处理] 重采样回目标采样率: {target_sr} Hz -> {self.sample_rate} Hz")
                enhanced = librosa.resample(
                    enhanced, orig_sr=target_sr, target_sr=self.sample_rate
                )
                logger.info(f"[降噪处理] 重采样后形状: {enhanced.shape}")

            # 音量归一化：防止音量过小或削波失真
            original_peak = np.max(np.abs(enhanced))
            logger.info(f"[降噪处理] 音量归一化前，Peak: {original_peak:.4f}")
            
            peak = original_peak
            if peak > 0:
                # 归一化到 -3dB (0.707) 峰值，留出余量防止 clipping
                target_peak = 0.707
                enhanced = enhanced * (target_peak / peak)
                logger.info(f"[降噪处理] 音量归一化: Peak {peak:.4f} -> {target_peak:.4f}")
            
            # 如果音量仍然太小，进行增益补偿
            rms = np.sqrt(np.mean(enhanced**2))
            logger.info(f"[降噪处理] 当前 RMS: {rms:.4f}")
            
            if rms < 0.05:  # 如果 RMS 小于 0.05，提升到 0.1
                gain = 0.1 / rms
                enhanced = enhanced * gain
                logger.info(f"[降噪处理] 增益补偿: RMS {rms:.4f} -> 0.1, 增益: {gain:.2f}x")
                
                # 重新检查峰值，防止 clipping
                peak = np.max(np.abs(enhanced))
                if peak > 0.95:
                    enhanced = enhanced * (0.95 / peak)
                    logger.info(f"[降噪处理] 防止削波，Peak 限制到 0.95")

            processing_time = time.time() - start_time
            final_peak = np.max(np.abs(enhanced))
            final_rms = np.sqrt(np.mean(enhanced**2))
            logger.info(f"[降噪处理] 处理完成，耗时: {processing_time:.3f}s")
            logger.info(f"[降噪处理] 输出音频 - Peak: {final_peak:.4f}, RMS: {final_rms:.4f}")

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
        logger.info(f"[降噪处理] _parse_output 开始解析，result 类型: {type(result)}")
        
        if result is None:
            logger.warning("[降噪处理] _parse_output 收到 None，返回 fallback")
            return fallback

        # 记录 result 的详细信息
        if isinstance(result, np.ndarray):
            logger.info(f"[降噪处理] result 是 numpy 数组，形状: {result.shape}, ndim: {result.ndim}")
            logger.info(f"[降噪处理] result 范围: [{np.min(result):.6f}, {np.max(result):.6f}], 均值: {np.mean(result):.6f}")
        elif isinstance(result, list):
            logger.info(f"[降噪处理] result 是列表，长度: {len(result)}")
            if len(result) > 0:
                logger.info(f"[降噪处理] result[0] 类型: {type(result[0])}")
                if hasattr(result[0], 'shape'):
                    logger.info(f"[降噪处理] result[0] 形状: {result[0].shape}")
                if hasattr(result[0], 'min'):
                    logger.info(f"[降噪处理] result[0] 范围: [{result[0].min():.6f}, {result[0].max():.6f}]")
        elif isinstance(result, dict):
            logger.info(f"[降噪处理] result 是字典，键: {list(result.keys())}")
        
        # 语音分离模型返回多个说话人，取第一个（目标说话人）
        if isinstance(result, list):
            if len(result) > 0:
                # 直接使用 result[0]，避免 np.array() 重新创建数组导致维度问题
                first_result = result[0]
                if isinstance(first_result, np.ndarray):
                    audio = first_result.astype(np.float32)
                else:
                    audio = np.array(first_result, dtype=np.float32)
                logger.info(f"[降噪处理] 从列表中提取 result[0]，提取后形状: {audio.shape}, ndim: {audio.ndim}, 范围: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
                if audio.ndim == 1:
                    logger.info(f"[降噪处理] 语音分离输出1D数组，直接返回")
                    return audio
                elif audio.ndim == 2:
                    # 2D数组: 如果第一维很小(<=2)，取第一个元素；否则取第一列
                    if audio.shape[0] <= 2:
                        extracted = audio[0].astype(np.float32)
                        logger.info(f"[降噪处理] 语音分离输出2D数组，取 audio[0]，原形状: {audio.shape}, 提取后形状: {extracted.shape}")
                        return extracted
                    else:
                        extracted = audio[:, 0].astype(np.float32)
                        logger.info(f"[降噪处理] 语音分离输出2D数组，取 audio[:, 0]，原形状: {audio.shape}, 提取后形状: {extracted.shape}")
                        return extracted
                else:
                    logger.warning(f"[降噪处理] 语音分离输出异常维度: {audio.ndim}, 形状: {audio.shape}")
                return audio
            logger.warning("[降噪处理] 列表为空，返回 fallback")
            return fallback

        # numpy 数组
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                logger.info(f"[降噪处理] 返回 1D 数组，形状: {result.shape}")
                return result.astype(np.float32)
            elif result.ndim == 2:
                # 多维数组取第一个通道
                logger.info(f"[降噪处理] 返回 2D 数组，取 result[0]，原形状: {result.shape}")
                return result[0].astype(np.float32) if result.shape[0] <= 2 else result[:, 0].astype(np.float32)

        # dict 格式
        if isinstance(result, dict):
            for key in ["output", "enhanced", "separated", "audio"]:
                if key in result:
                    val = result[key]
                    logger.info(f"[降噪处理] 从字典中找到 key '{key}'")
                    if isinstance(val, np.ndarray):
                        logger.info(f"[降噪处理] key '{key}' 值形状: {val.shape}, ndim: {val.ndim}")
                        if val.ndim == 1:
                            return val.astype(np.float32)
                        elif val.ndim == 2:
                            # 2D数组: 如果第一维很小(<=2)，取第一个元素；否则取第一列
                            if val.shape[0] <= 2:
                                logger.info(f"[降噪处理] 2D数组，取 result[0]，原形状: {val.shape}")
                                return val[0].astype(np.float32)
                            else:
                                logger.info(f"[降噪处理] 2D数组，取 result[:, 0]，原形状: {val.shape}")
                                return val[:, 0].astype(np.float32)
            # 取第一个 numpy 值
            for val in result.values():
                if isinstance(val, np.ndarray):
                    logger.info(f"[降噪处理] 从字典值中提取数组，形状: {val.shape}")
                    if val.ndim == 1:
                        return val.astype(np.float32)
                    elif val.ndim == 2:
                        # 2D数组: 如果第一维很小(<=2)，取第一个元素；否则取第一列
                        if val.shape[0] <= 2:
                            logger.info(f"[降噪处理] 2D数组，取 result[0]，原形状: {val.shape}")
                            return val[0].astype(np.float32)
                        else:
                            logger.info(f"[降噪处理] 2D数组，取 result[:, 0]，原形状: {val.shape}")
                            return val[:, 0].astype(np.float32)

        logger.warning("[降噪处理] 无法解析 result 类型，返回 fallback")
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
