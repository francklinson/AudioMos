"""
FunASR适配器
支持 Paraformer-large 和 SenseVoice-Small
FunASR新版本(1.3+)使用注册名: Paraformer, SenseVoiceSmall
模型文件优先从项目本地 models/asr/ 目录加载，否则自动从ModelScope/HuggingFace下载
"""

import os
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")

# 设置模型缓存到项目本地目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)))
_LOCAL_CACHE = os.path.join(_PROJECT_ROOT, "models", "asr", "modelscope_cache")
os.environ.setdefault("MODELSCOPE_CACHE", _LOCAL_CACHE)


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

            # FunASR 1.3+ 注册名
            model_name = "Paraformer"
            hub = "ms"

            # 优先使用本地模型目录
            if self.model_dir and os.path.exists(self.model_dir) and os.listdir(self.model_dir):
                logger.info(f"[Paraformer] 从本地加载模型: {self.model_dir}")
                model_name = self.model_dir
                hub = "local"

            self._model = AutoModel(
                model=model_name,
                device=self.device,
                disable_update=True,
                hub=hub,
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

            # FunASR 1.3+ 注册名
            model_name = "SenseVoiceSmall"
            hub = "ms"

            if self.model_dir and os.path.exists(self.model_dir) and os.listdir(self.model_dir):
                logger.info(f"[SenseVoice] 从本地加载模型: {self.model_dir}")
                model_name = self.model_dir
                hub = "local"

            self._model = AutoModel(
                model=model_name,
                device=self.device,
                disable_update=True,
                hub=hub,
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
