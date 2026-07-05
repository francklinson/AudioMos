"""
Fun-ASR-Nano (800M) 适配器
通过FunASR AutoModel加载，支持31种语言+7种中文方言
替代原FunASR-LLM 7.7B，更轻量适合3090部署

HuggingFace: FunAudioLLM/Fun-ASR-Nano-2512
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
    """Fun-ASR-Nano 800M 适配器 — 轻量LLM-based ASR，支持31种语言"""

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
        for dirname in ["funasr-llm", "Fun-ASR-Nano-2512", "fun-asr-nano"]:
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
                logger.info(f"[Fun-ASR-Nano] 从本地加载模型: {model_path}")
                self._model = AutoModel(
                    model=model_path,
                    trust_remote_code=True,
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=self.device,
                    disable_update=True,
                )
            else:
                # 使用HuggingFace模型名加载
                logger.info("[Fun-ASR-Nano] 从HuggingFace加载: FunAudioLLM/Fun-ASR-Nano-2512")
                self._model = AutoModel(
                    model="FunAudioLLM/Fun-ASR-Nano-2512",
                    trust_remote_code=True,
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=self.device,
                    disable_update=True,
                )

            self._is_initialized = True
            logger.info("[Fun-ASR-Nano] 模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"[Fun-ASR-Nano] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Fun-ASR-Nano模型未初始化")

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
