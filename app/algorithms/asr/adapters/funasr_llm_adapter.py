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

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None,
                 offline: bool = True, **kwargs):
        super().__init__(
            name="funasr-llm",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
            offline=offline,
        )

    def _find_model_dir(self) -> Optional[str]:
        """查找本地模型目录（仅检查 self.model_dir，不再遍历项目目录）"""
        if self.model_dir and os.path.isdir(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
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
                    device=self.device,
                    disable_update=True,
                )
            else:
                if self.offline:
                    raise FileNotFoundError(
                        f"[Fun-ASR-Nano] 离线模式：未找到本地模型，请放置在 {self.model_dir}"
                    )
                logger.info("[Fun-ASR-Nano] 从HuggingFace加载: FunAudioLLM/Fun-ASR-Nano-2512")
                self._model = AutoModel(
                    model="FunAudioLLM/Fun-ASR-Nano-2512",
                    trust_remote_code=True,
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

        # 直接传 numpy 数组，无需临时文件
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
