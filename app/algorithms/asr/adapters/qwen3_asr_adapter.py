"""
Qwen3-ASR-1.7B 适配器
使用 qwen-asr 包加载（支持 transformers 和 vLLM 两种后端）

HuggingFace: Qwen/Qwen3-ASR-1.7B
依赖: pip install qwen-asr
"""

import os
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult

logger = logging.getLogger("audiomos")


class Qwen3ASRAdapter(BaseASR):
    """Qwen3-ASR-1.7B 适配器 — AuT音频编码器 + Qwen3 LLM"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None,
                 offline: bool = True, **kwargs):
        super().__init__(
            name="qwen3-asr",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
            offline=offline,
        )
        self._processor = None

    def _find_model_dir(self) -> Optional[str]:
        """查找本地模型目录（仅检查 self.model_dir）"""
        if self.model_dir and os.path.isdir(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
        return None

    def initialize(self) -> bool:
        try:
            from qwen_asr import Qwen3ASRModel
            import torch

            model_dir = self._find_model_dir()

            if model_dir:
                logger.info(f"[Qwen3-ASR] 从本地加载模型: {model_dir}")
                model_path = model_dir
            else:
                if self.offline:
                    raise FileNotFoundError(
                        f"[Qwen3-ASR] 离线模式：未找到本地模型，请放置在 {self.model_dir}"
                    )
                logger.info("[Qwen3-ASR] 从HuggingFace下载模型: Qwen/Qwen3-ASR-1.7B")
                model_path = "Qwen/Qwen3-ASR-1.7B"

            model_kwargs = dict(
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                max_new_tokens=512,
            )
            if self.device != "cpu":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "cuda:0"

            self._model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)

            self._is_initialized = True
            logger.info("[Qwen3-ASR] 模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"[Qwen3-ASR] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Qwen3-ASR模型未初始化")

        sr = sample_rate or self.sample_rate
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 语言映射：ISO代码 -> Qwen3-ASR 全称
        lang_map = {
            "zh": "Chinese", "en": "English", "ja": "Japanese",
            "ko": "Korean", "yue": "Cantonese", "ar": "Arabic",
            "de": "German", "fr": "French", "es": "Spanish",
            "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
            "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
        }
        qwen_lang = lang_map.get(self.language, "Chinese")

        # qwen-asr 的 transcribe 接受 (np.ndarray, sr) 元组，无需临时文件
        results = self._model.transcribe(
            (audio, sr),
            language=qwen_lang,
            return_time_stamps=False,
        )

        text = ""
        if results and len(results) > 0:
            text = (results[0].text or "").strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )
