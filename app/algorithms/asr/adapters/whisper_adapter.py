"""
Whisper适配器
支持 OpenAI Whisper large-v3-turbo，从本地模型目录加载
"""

import os
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class WhisperAdapter(BaseASR):
    """Whisper Large-v3 Turbo 适配器"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="whisper-large-v3-turbo",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )

    def initialize(self) -> bool:
        try:
            import whisper

            model_name = "large-v3-turbo"

            # 优先使用本地模型
            if self.model_dir and os.path.exists(self.model_dir):
                logger.info(f"[Whisper] 从本地加载模型: {self.model_dir}")
                self._model = whisper.load_model(self.model_dir, device=self.device)
            else:
                # 尝试项目models目录
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )))
                local_path = os.path.join(project_root, "models", "asr", "whisper-large-v3-turbo")
                if os.path.exists(local_path):
                    logger.info(f"[Whisper] 从本地加载模型: {local_path}")
                    self._model = whisper.load_model(local_path, device=self.device)
                else:
                    logger.info(f"[Whisper] 从HuggingFace下载模型: {model_name}")
                    self._model = whisper.load_model(model_name, device=self.device)

            self._is_initialized = True
            logger.info(f"[Whisper] 模型初始化成功")
            return True
        except Exception as e:
            logger.error(f"[Whisper] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Whisper模型未初始化")

        # Whisper需要float32输入
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        result = self._model.transcribe(
            audio,
            language="zh",
            task="transcribe",
            fp16=(self.device != "cpu"),
        )

        text = result.get("text", "").strip()
        segments = []
        for seg in result.get("segments", []):
            segments.append(ASRSegment(
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", "").strip(),
            ))

        return ASRResult(
            text=text,
            language=self.language,
            segments=segments if segments else None,
            algorithm_name=self.name,
        )
