"""
Qwen3-ASR-1.7B 适配器
使用 HuggingFace Transformers 加载

HuggingFace: Qwen/Qwen3-ASR-1.7B
依赖: pip install transformers torch soundfile librosa
"""

import os
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult

logger = logging.getLogger("audiomos")


class Qwen3ASRAdapter(BaseASR):
    """Qwen3-ASR-1.7B 适配器 — AuT音频编码器 + Qwen3 LLM"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="qwen3-asr",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )
        self._processor = None

    def _find_model_dir(self) -> Optional[str]:
        """查找本地模型目录"""
        if self.model_dir and os.path.exists(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))
        for dirname in ["qwen3-asr", "Qwen3-ASR-1.7B", "qwen3-asr-1.7b"]:
            candidate = os.path.join(project_root, "models", "asr", dirname)
            if os.path.exists(candidate) and os.listdir(candidate):
                return candidate
        return None

    def initialize(self) -> bool:
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch

            model_dir = self._find_model_dir()

            if model_dir:
                logger.info(f"[Qwen3-ASR] 从本地加载模型: {model_dir}")
                model_path = model_dir
            else:
                logger.info("[Qwen3-ASR] 从HuggingFace下载模型: Qwen/Qwen3-ASR-1.7B")
                model_path = "Qwen/Qwen3-ASR-1.7B"

            torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

            self._processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=bool(model_dir),
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=bool(model_dir),
            )
            self._model.to(self.device)
            self._model.eval()

            self._is_initialized = True
            logger.info("[Qwen3-ASR] 模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"[Qwen3-ASR] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Qwen3-ASR模型未初始化")

        import torch

        sr = sample_rate or self.sample_rate
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 预处理
        inputs = self._processor(
            audio=audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = self._model.generate(**inputs)

        # 解码
        text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )
