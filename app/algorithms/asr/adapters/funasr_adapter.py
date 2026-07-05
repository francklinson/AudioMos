"""
FunASR适配器
支持 Paraformer-large 和 SenseVoice-Small
直接从 models/asr/ 本地目录加载，无需 ModelScope 缓存
"""

import os
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class ParaformerAdapter(BaseASR):
    """Paraformer-large 适配器 — 通过FunASR加载"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="paraformer-large",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )

    def initialize(self) -> bool:
        try:
            from funasr import AutoModel

            # 直接从本地模型目录加载（绕开 modelscope 缓存）
            model_dir = self.model_dir
            if model_dir and os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "model.pt")):
                logger.info(f"[Paraformer] 从本地模型目录加载: {model_dir}")
                self._model = AutoModel(
                    model=model_dir,
                    device=self.device,
                    disable_update=True,
                )
            else:
                # 兜底：从 HuggingFace 加载
                logger.info("[Paraformer] 从 HuggingFace 加载: FunAudioLLM/paraformer-large")
                self._model = AutoModel(
                    model="FunAudioLLM/paraformer-large",
                    device=self.device,
                    disable_update=True,
                )
            self._is_initialized = True
            logger.info(f"[Paraformer] 模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"[Paraformer] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Paraformer模型未初始化")

        result = self._model.generate(
            input=audio,
            batch_size_s=300,
        )

        text = ""
        segments = []
        if result and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                text = item.get("text", "")
                timestamp = item.get("timestamp", [])
                for ts in timestamp:
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        segments.append(ASRSegment(
                            start=ts[0] / 1000.0,
                            end=ts[1] / 1000.0,
                            text=ts[2] if len(ts) > 2 else "",
                        ))
            else:
                text = str(item)

        return ASRResult(
            text=text,
            language=self.language,
            segments=segments if segments else None,
            algorithm_name=self.name,
        )


class SenseVoiceAdapter(BaseASR):
    """SenseVoice-Small 适配器 — 通过FunASR加载"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="sensevoice-small",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )

    def initialize(self) -> bool:
        try:
            from funasr import AutoModel

            # 直接从本地模型目录加载（绕开 modelscope 缓存）
            model_dir = self.model_dir
            if model_dir and os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "model.pt")):
                logger.info(f"[SenseVoice] 从本地模型目录加载: {model_dir}")
                self._model = AutoModel(
                    model=model_dir,
                    device=self.device,
                    disable_update=True,
                )
            else:
                # 兜底：从 HuggingFace 加载
                logger.info("[SenseVoice] 从 HuggingFace 加载: FunAudioLLM/SenseVoiceSmall")
                self._model = AutoModel(
                    model="FunAudioLLM/SenseVoiceSmall",
                    device=self.device,
                    disable_update=True,
                )
            self._is_initialized = True
            logger.info(f"[SenseVoice] 模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"[SenseVoice] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("SenseVoice模型未初始化")

        result = self._model.generate(
            input=audio,
            batch_size_s=300,
            language="auto",
            use_itn=True,
        )

        text = ""
        if result and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = str(item)

        # SenseVoice 输出可能包含特殊标记，清理
        import re
        text = re.sub(r'<\|[^|]*\|>', '', text).strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )
