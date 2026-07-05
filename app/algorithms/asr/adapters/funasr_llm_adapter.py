"""
FunASR-LLM (7.7B) 适配器
通过FunASR AutoModel加载LLM-based ASR模型

ModelScope: iic/speech_seallm_asr_nat-zh-cn-16k
依赖: pip install funasr
"""

import os
import re
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult

logger = logging.getLogger("audiomos")


class FunASRLLMAdapter(BaseASR):
    """FunASR-LLM 7.7B 适配器 — LLM-based ASR with VAD+Punc+Speaker"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="funasr-llm",
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
        for dirname in ["funasr-llm", "speech_seallm_asr_nat-zh-cn-16k", "funasr-llm-7.7b"]:
            candidate = os.path.join(project_root, "models", "asr", dirname)
            if os.path.exists(candidate) and os.listdir(candidate):
                return candidate
        return None

    def initialize(self) -> bool:
        try:
            from funasr import AutoModel

            model_dir = self._find_model_dir()

            if model_dir:
                model_path = os.path.abspath(model_dir)
                logger.info(f"[FunASR-LLM] 从本地加载模型: {model_path}")
                self._model = AutoModel(
                    model=model_path,
                    hub="ms",
                    device=self.device,
                    disable_update=True,
                )
            else:
                # 使用FunASR注册名从ModelScope加载
                logger.info("[FunASR-LLM] 从ModelScope加载: iic/speech_seallm_asr_nat-zh-cn-16k")
                self._model = AutoModel(
                    model="iic/speech_seallm_asr_nat-zh-cn-16k",
                    hub="ms",
                    device=self.device,
                    disable_update=True,
                )

            self._is_initialized = True
            logger.info("[FunASR-LLM] 模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"[FunASR-LLM] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("FunASR-LLM模型未初始化")

        result = self._model.generate(
            input=audio,
            batch_size_s=300,
        )

        text = ""
        if result and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = str(item)

        # 清理特殊标记
        text = re.sub(r'<\|[^|]*\|>', '', text).strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )
