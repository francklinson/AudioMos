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
from typing import Optional, Any
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

    # ── 流式转录实现（全量累积 + 防抖跳过）──

    def supports_streaming(self) -> bool:
        return self._is_initialized

    def init_streaming_state(self, **kwargs) -> Any:
        return {
            "audio_buffer": np.array([], dtype=np.float32),
            "last_text": "",
            "chunk_size_sec": kwargs.get("chunk_size_sec", 1.0),
            "min_chunk_samples": int(self.sample_rate * kwargs.get("chunk_size_sec", 1.0)),
            "last_transcribe_buffer_len": 0,
        }

    def streaming_transcribe(self, audio_chunk: np.ndarray, state: Any) -> dict:
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_chunk])

        if len(state["audio_buffer"]) < state["min_chunk_samples"]:
            return {"text": state["last_text"], "language": self.language, "is_final": False}

        # ── 减少重复转录：新增音频不足 min_chunk_samples 时跳过 ──
        new_samples = len(state["audio_buffer"]) - state.get("last_transcribe_buffer_len", 0)
        if new_samples < state["min_chunk_samples"]:
            return {"text": state["last_text"], "language": self.language, "is_final": False}

        try:
            result = self.transcribe(state["audio_buffer"], self.sample_rate)
            text = result.text
            if text != state["last_text"]:
                state["last_text"] = text
            state["last_transcribe_buffer_len"] = len(state["audio_buffer"])
        except Exception as e:
            logger.warning(f"[Fun-ASR-Nano] 流式转录中间结果失败: {e}")

        return {"text": state["last_text"], "language": self.language, "is_final": False}

    def finish_streaming_transcribe(self, state: Any) -> dict:
        audio_buffer = state.get("audio_buffer", np.array([], dtype=np.float32))
        if len(audio_buffer) == 0:
            return {"text": state.get("last_text", ""), "language": self.language, "is_final": True}

        try:
            result = self.transcribe(audio_buffer, self.sample_rate)
            text = result.text
        except Exception as e:
            logger.warning(f"[Fun-ASR-Nano] 最终转录失败: {e}")
            text = state.get("last_text", "")

        return {"text": text, "language": self.language, "is_final": True}
