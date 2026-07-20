"""
FunASR适配器
支持 Paraformer-large 和 SenseVoice-Small
直接从 models/asr/ 本地目录加载，无需 ModelScope 缓存
"""

import os
import re
import logging
from typing import Optional, Any
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class ParaformerAdapter(BaseASR):
    """Paraformer-large 适配器 — 通过FunASR加载"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None,
                 offline: bool = True, **kwargs):
        super().__init__(
            name="paraformer-large",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
            offline=offline,
        )

    def initialize(self) -> bool:
        try:
            from funasr import AutoModel

            model_dir = self.model_dir
            if model_dir and os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "model.pt")):
                logger.info(f"[Paraformer] 从本地模型目录加载: {model_dir}")
                self._model = AutoModel(
                    model=model_dir,
                    device=self.device,
                    disable_update=True,
                )
            else:
                if self.offline:
                    raise FileNotFoundError(
                        f"[Paraformer] 离线模式：未找到本地模型，请放置在 {self.model_dir}"
                    )
                logger.info("[Paraformer] 从HuggingFace加载: FunAudioLLM/paraformer-large")
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

    # ── 滑动窗口流式转录（推理时间恒定，不随录音时长增长）──

    def supports_streaming(self) -> bool:
        return self._is_initialized

    def init_streaming_state(self, **kwargs) -> Any:
        window_size_sec = kwargs.get("window_size_sec", 5.0)
        return {
            "audio_buffer": np.array([], dtype=np.float32),
            "confirmed_text": "",
            "last_text": "",
            "last_window_text": "",
            "chunk_size_sec": kwargs.get("chunk_size_sec", 1.0),
            "min_chunk_samples": int(self.sample_rate * kwargs.get("chunk_size_sec", 1.0)),
            "window_size_sec": window_size_sec,
            "window_samples": int(self.sample_rate * window_size_sec),
            "last_transcribe_buffer_len": 0,
        }

    def streaming_transcribe(self, audio_chunk: np.ndarray, state: Any) -> dict:
        return self._streaming_sliding_transcribe(audio_chunk, state)

    def finish_streaming_transcribe(self, state: Any) -> dict:
        audio_buffer = state.get("audio_buffer", np.array([], dtype=np.float32))
        if len(audio_buffer) == 0:
            return {"text": state.get("confirmed_text", "") or state.get("last_text", ""),
                    "language": self.language, "is_final": True}

        try:
            result = self.transcribe(audio_buffer, self.sample_rate)
            text = result.text
        except Exception as e:
            logger.warning(f"[Paraformer] 最终转录失败: {e}")
            text = state.get("confirmed_text", "") or state.get("last_text", "")

        return {"text": text, "language": self.language, "is_final": True}


class SenseVoiceAdapter(BaseASR):
    """SenseVoice-Small 适配器 — 通过FunASR加载"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None,
                 offline: bool = True, **kwargs):
        super().__init__(
            name="sensevoice-small",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
            offline=offline,
        )

    def initialize(self) -> bool:
        try:
            from funasr import AutoModel

            model_dir = self.model_dir
            if model_dir and os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "model.pt")):
                logger.info(f"[SenseVoice] 从本地模型目录加载: {model_dir}")
                self._model = AutoModel(
                    model=model_dir,
                    device=self.device,
                    disable_update=True,
                )
            else:
                if self.offline:
                    raise FileNotFoundError(
                        f"[SenseVoice] 离线模式：未找到本地模型，请放置在 {self.model_dir}"
                    )
                logger.info("[SenseVoice] 从HuggingFace加载: FunAudioLLM/SenseVoiceSmall")
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
        text = re.sub(r'<\|[^|]*\|>', '', text).strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )

    # ── 流式转录实现（模拟流式：累积音频 + 周期性转录）──

    def supports_streaming(self) -> bool:
        return self._is_initialized

    def init_streaming_state(self, **kwargs) -> Any:
        return {
            "audio_buffer": np.array([], dtype=np.float32),
            "last_text": "",
            "chunk_size_sec": kwargs.get("chunk_size_sec", 2.0),
            "min_chunk_samples": int(self.sample_rate * kwargs.get("chunk_size_sec", 2.0)),
        }

    def streaming_transcribe(self, audio_chunk: np.ndarray, state: Any) -> dict:
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_chunk])

        if len(state["audio_buffer"]) < state["min_chunk_samples"]:
            return {"text": state["last_text"], "language": self.language, "is_final": False}

        try:
            result = self.transcribe(state["audio_buffer"], self.sample_rate)
            text = result.text
            if text != state["last_text"]:
                state["last_text"] = text
        except Exception as e:
            logger.warning(f"[SenseVoice] 流式转录中间结果失败: {e}")

        return {"text": state["last_text"], "language": self.language, "is_final": False}

    def finish_streaming_transcribe(self, state: Any) -> dict:
        audio_buffer = state.get("audio_buffer", np.array([], dtype=np.float32))
        if len(audio_buffer) == 0:
            return {"text": state.get("last_text", ""), "language": self.language, "is_final": True}

        try:
            result = self.transcribe(audio_buffer, self.sample_rate)
            text = result.text
        except Exception as e:
            logger.warning(f"[SenseVoice] 最终转录失败: {e}")
            text = state.get("last_text", "")

        return {"text": text, "language": self.language, "is_final": True}
