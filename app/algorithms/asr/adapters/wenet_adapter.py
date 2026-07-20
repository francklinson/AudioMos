"""
WeNet适配器
使用wenet.load_model()从本地模型目录加载
"""

import os
import logging
from typing import Optional, Any
import numpy as np

from ..base import BaseASR, ASRResult

logger = logging.getLogger("audiomos")


class WeNetAdapter(BaseASR):
    """WeNet U2++ 适配器 — 使用wenet.load_model()加载"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None,
                 offline: bool = True, **kwargs):
        super().__init__(
            name="wenet-u2pp",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
            offline=offline,
        )

    def _find_model_path(self) -> Optional[str]:
        """查找本地模型目录（仅检查 self.model_dir）"""
        if self.model_dir and os.path.isdir(self.model_dir):
            required_files = ["train.yaml"]
            if all(os.path.exists(os.path.join(self.model_dir, f)) for f in required_files):
                return self.model_dir
        return None

    def initialize(self) -> bool:
        try:
            import wenet

            model_path = self._find_model_path()

            if model_path:
                logger.info(f"[WeNet] 从本地加载模型: {model_path}")
                self._model = wenet.load_model(model_path, device=self.device)
            else:
                if self.offline:
                    raise FileNotFoundError(
                        f"[WeNet] 离线模式：未找到本地模型，请放置在 {self.model_dir}"
                    )
                logger.warning("[WeNet] 未找到本地模型，尝试使用默认模型")
                self._model = wenet.load_model("chinese", device=self.device)

            self._is_initialized = True
            logger.info(f"[WeNet] 模型初始化成功")
            return True
        except Exception as e:
            logger.error(f"[WeNet] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("WeNet模型未初始化")

        sr = sample_rate or self.sample_rate

        # WeNet CLI 接口接受文件路径，需要临时保存
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sr)

        try:
            result = self._model.transcribe(temp_path)
            if isinstance(result, str):
                text = result.strip()
            elif hasattr(result, 'text'):
                text = result.text.strip()
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                first = result[0]
                if hasattr(first, 'text'):
                    text = first.text.strip()
                else:
                    text = str(first).strip()
            else:
                text = str(result).strip()
        finally:
            os.unlink(temp_path)

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )

    # ── 滑动窗口流式转录（推理时间恒定，不随录音时长增长）──

    def supports_streaming(self) -> bool:
        return self._is_initialized

    def init_streaming_state(self, **kwargs) -> Any:
        window_size_sec = kwargs.get("window_size_sec", 5.0)
        return {
            "audio_buffer": np.array([], dtype=np.float32),
            "confirmed_text": "",                                      # 已确认不会回退的文本
            "last_text": "",
            "last_window_text": "",                                    # 上一窗口转录结果（用于overlap检测）
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
            logger.warning(f"[WeNet] 最终转录失败: {e}")
            text = state.get("confirmed_text", "") or state.get("last_text", "")

        return {"text": text, "language": self.language, "is_final": True}
