"""
FireRedASR2-AED 适配器
使用 AED (1.1B) 版本，平衡高性能与计算效率，适合3090部署

HuggingFace: FireRedTeam/FireRedASR2-AED
GitHub: https://github.com/FireRedTeam/FireRedASR2S
AED架构: Attention-based Encoder-Decoder，CER 3.05% (vs LLM 2.89%)
依赖: pip install fireredasr2s 或 pip install fasr-asr-firered
"""

import os
import logging
import tempfile
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class FireRedASR2Adapter(BaseASR):
    """FireRedASR2-AED 适配器 — Attention-based Encoder-Decoder (1.1B)"""

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
        if self.model_dir and os.path.exists(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))
        for dirname in ["firered-asr2", "FireRedASR2-AED", "fireredasr2-aed"]:
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
                    "未找到FireRedASR2-AED模型文件。请下载模型到 models/asr/firered-asr2/ 目录"
                )

            logger.info(f"[FireRedASR2-AED] 从本地加载模型: {model_dir}")

            asr_config = FireRedAsr2Config(
                use_gpu=(self.device != "cpu"),
                beam_size=3,
                nbest=1,
                decode_max_len=0,
                softmax_smoothing=1.25,
                aed_length_penalty=0.6,
                eos_penalty=1.0,
            )

            # AED模式
            self._model = FireRedAsr2.from_pretrained("aed", model_dir, asr_config)
            self._use_fasr = False
            self._is_initialized = True
            logger.info("[FireRedASR2-AED] 模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"[FireRedASR2-AED] 初始化失败: {e}")
            return False

    def _initialize_fasr(self) -> bool:
        """使用fasr库初始化"""
        model_dir = self._find_model_dir()
        if not model_dir:
            raise FileNotFoundError("未找到FireRedASR2-AED模型文件")

        logger.info(f"[FireRedASR2-AED] 使用fasr库从本地加载: {model_dir}")
        model_class = self._registry.asr_models.get("firered_aed")
        self._model = model_class()
        self._model.from_checkpoint(checkpoint_dir=model_dir)
        self._is_initialized = True
        logger.info("[FireRedASR2-AED] 模型初始化成功 (fasr)")
        return True

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("FireRedASR2-AED模型未初始化")

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
                "nbest": 1,
                "decode_max_len": 0,
                "softmax_smoothing": 1.25,
                "aed_length_penalty": 0.6,
                "eos_penalty": 1.0,
            }
        )

        text = ""
        segments = []
        if results and len(results) > 0:
            item = results[0]
            if isinstance(item, dict):
                text = item.get("text", "") or item.get("ref", "")
                # AED支持字级时间戳
                timestamp = item.get("timestamp", [])
                for ts in timestamp:
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        segments.append(ASRSegment(
                            start=ts[0] / 1000.0,
                            end=ts[1] / 1000.0,
                            text=ts[2] if len(ts) > 2 else "",
                        ))
            elif hasattr(item, 'text'):
                text = item.text
            else:
                text = str(item).strip()
        text = text.strip()

        return ASRResult(
            text=text,
            language=self.language,
            segments=segments if segments else None,
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
