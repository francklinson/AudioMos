"""
FireRedASR2-LLM 适配器
支持 FireRedASR2-LLM (8.3B) 和 FireRedASR2-AED (1.1B)
使用 FireRedASR2S 官方推理代码加载本地模型

HuggingFace: FireRedTeam/FireRedASR2-LLM
GitHub: https://github.com/FireRedTeam/FireRedASR2S
依赖: pip install fireredasr2s 或 从源码安装
"""

import os
import logging
import tempfile
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class FireRedASR2Adapter(BaseASR):
    """FireRedASR2-LLM 适配器 — Encoder-Adapter-LLM 架构"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="firered-asr2",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )

    def _find_model_dir(self) -> Optional[str]:
        """查找本地模型目录"""
        # 1. 优先使用传入的model_dir
        if self.model_dir and os.path.exists(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
        # 2. 查找项目models目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))
        for dirname in ["firered-asr2", "FireRedASR2-LLM", "fireredasr2-llm"]:
            candidate = os.path.join(project_root, "models", "asr", dirname)
            if os.path.exists(candidate) and os.listdir(candidate):
                return candidate
        return None

    def initialize(self) -> bool:
        try:
            # 尝试导入FireRedASR2S官方库
            try:
                from fireredasr2.models.fireredasr2 import FireRedAsr2
                from fireredasr2.models.fireredasr2_config import FireRedAsr2Config
            except ImportError:
                # 回退: 尝试fasr库
                try:
                    from fasr.config import registry
                    self._use_fasr = True
                    self._registry = registry
                    return self._initialize_fasr()
                except ImportError:
                    raise ImportError(
                        "FireRedASR2未安装。请执行: "
                        "pip install fireredasr2s 或 pip install fasr-asr-firered"
                    )

            model_dir = self._find_model_dir()
            if not model_dir:
                raise FileNotFoundError(
                    "未找到FireRedASR2-LLM模型文件。请下载模型到 models/asr/firered-asr2/ 目录"
                )

            logger.info(f"[FireRedASR2] 从本地加载模型: {model_dir}")

            asr_config = FireRedAsr2Config(
                use_gpu=(self.device != "cpu"),
                beam_size=3,
                decode_min_len=0,
                repetition_penalty=1.0,
                llm_length_penalty=0.0,
                temperature=1.0,
            )

            self._model = FireRedAsr2.from_pretrained("llm", model_dir, asr_config)
            self._use_fasr = False
            self._is_initialized = True
            logger.info("[FireRedASR2] 模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"[FireRedASR2] 初始化失败: {e}")
            return False

    def _initialize_fasr(self) -> bool:
        """使用fasr库初始化"""
        model_dir = self._find_model_dir()
        if not model_dir:
            raise FileNotFoundError("未找到FireRedASR2-LLM模型文件")

        logger.info(f"[FireRedASR2] 使用fasr库从本地加载: {model_dir}")
        model_class = self._registry.asr_models.get("firered_llm")
        self._model = model_class()
        self._model.from_checkpoint(checkpoint_dir=model_dir)
        self._is_initialized = True
        logger.info("[FireRedASR2] 模型初始化成功 (fasr)")
        return True

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("FireRedASR2模型未初始化")

        # 保存临时WAV文件（FireRedASR2需要文件路径输入）
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, audio, sample_rate or self.sample_rate)

            if self._use_fasr:
                return self._transcribe_fasr(audio, sample_rate)
            else:
                return self._transcribe_official(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _transcribe_official(self, audio_path: str) -> ASRResult:
        """使用官方推理代码"""
        batch_uttid = ["utt1"]
        batch_wav_path = [audio_path]

        results = self._model.transcribe(
            batch_uttid,
            batch_wav_path,
            {
                "use_gpu": 1 if self.device != "cpu" else 0,
                "beam_size": 3,
                "decode_max_len": 0,
                "decode_min_len": 0,
                "repetition_penalty": 3.0,
                "llm_length_penalty": 1.0,
                "temperature": 1.0,
            }
        )

        text = ""
        if results and len(results) > 0:
            item = results[0]
            if isinstance(item, dict):
                text = item.get("text", "") or item.get("ref", "")
            elif hasattr(item, 'text'):
                text = item.text
            else:
                text = str(item).strip()
        text = text.strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )

    def _transcribe_fasr(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        """使用fasr库推理"""
        sr = sample_rate or self.sample_rate
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        tokens = self._model.transcribe([audio], sample_rate=sr)
        text = ""
        if tokens and len(tokens) > 0:
            token_list = tokens[0]
            if isinstance(token_list, list):
                text = "".join(token_list)
            else:
                text = str(token_list)
        text = text.strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )
