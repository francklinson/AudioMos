"""
Qwen3-ASR-1.7B 适配器
使用 qwen-asr 包加载（支持 transformers 和 vLLM 两种后端）

HuggingFace: Qwen/Qwen3-ASR-1.7B
依赖: pip install qwen-asr
流式依赖: pip install qwen-asr[vllm]
"""

import os
import logging
from typing import Optional, Any
import numpy as np

from ..base import BaseASR, ASRResult

logger = logging.getLogger("audiomos")


class Qwen3ASRAdapter(BaseASR):
    """Qwen3-ASR-1.7B 适配器 — AuT音频编码器 + Qwen3 LLM"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None,
                 offline: bool = True, use_vllm: bool = False, **kwargs):
        super().__init__(
            name="qwen3-asr",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
            offline=offline,
        )
        self._processor = None
        self._use_vllm = use_vllm
        self._streaming_available = False

    def _find_model_dir(self) -> Optional[str]:
        """查找本地模型目录（仅检查 self.model_dir）"""
        if self.model_dir and os.path.isdir(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
        return None

    def initialize(self) -> bool:
        try:
            from qwen_asr import Qwen3ASRModel
            import torch

            model_dir = self._find_model_dir()

            if model_dir:
                logger.info(f"[Qwen3-ASR] 从本地加载模型: {model_dir}")
                model_path = model_dir
            else:
                if self.offline:
                    raise FileNotFoundError(
                        f"[Qwen3-ASR] 离线模式：未找到本地模型，请放置在 {self.model_dir}"
                    )
                logger.info("[Qwen3-ASR] 从HuggingFace下载模型: Qwen/Qwen3-ASR-1.7B")
                model_path = "Qwen/Qwen3-ASR-1.7B"

            # 尝试使用 vLLM 后端（支持流式转录）
            if self._use_vllm:
                try:
                    self._model = Qwen3ASRModel.LLM(
                        model=model_path,
                        gpu_memory_utilization=0.8,
                        max_new_tokens=32,
                    )
                    self._streaming_available = True
                    self._is_initialized = True
                    logger.info("[Qwen3-ASR] vLLM后端初始化成功（支持流式转录）")
                    return True
                except Exception as e:
                    logger.warning(f"[Qwen3-ASR] vLLM后端初始化失败: {e}，回退到transformers后端")
                    self._use_vllm = False

            # transformers 后端（非流式）
            model_kwargs = dict(
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                max_new_tokens=512,
            )
            if self.device != "cpu":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "cuda:0"

            self._model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)

            self._is_initialized = True
            logger.info("[Qwen3-ASR] transformers后端初始化成功（不支持流式转录）")
            return True

        except Exception as e:
            logger.error(f"[Qwen3-ASR] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Qwen3-ASR模型未初始化")

        sr = sample_rate or self.sample_rate
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 语言映射：ISO代码 -> Qwen3-ASR 全称
        lang_map = {
            "zh": "Chinese", "en": "English", "ja": "Japanese",
            "ko": "Korean", "yue": "Cantonese", "ar": "Arabic",
            "de": "German", "fr": "French", "es": "Spanish",
            "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
            "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
        }
        qwen_lang = lang_map.get(self.language, "Chinese")

        # qwen-asr 的 transcribe 接受 (np.ndarray, sr) 元组，无需临时文件
        results = self._model.transcribe(
            (audio, sr),
            language=qwen_lang,
            return_time_stamps=False,
        )

        text = ""
        if results and len(results) > 0:
            text = (results[0].text or "").strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )

    # ── 流式转录实现（vLLM 后端原生流式 + transformers 模拟流式回退）──

    def supports_streaming(self) -> bool:
        """是否支持流式转录"""
        return self._is_initialized

    def init_streaming_state(self, **kwargs) -> Any:
        """
        初始化流式转录状态

        vLLM 后端：使用原生流式状态
        transformers 后端：使用模拟流式（累积音频 + 周期性转录）
        """
        if self._streaming_available:
            # vLLM 原生流式
            state = self._model.init_streaming_state(
                unfixed_chunk_num=kwargs.get("unfixed_chunk_num", 2),
                unfixed_token_num=kwargs.get("unfixed_token_num", 5),
                chunk_size_sec=kwargs.get("chunk_size_sec", 2.0),
            )
            state._backend = "vllm"
            return state
        else:
            # transformers 模拟流式
            return {
                "_backend": "transformers",
                "audio_buffer": np.array([], dtype=np.float32),
                "last_text": "",
                "chunk_size_sec": kwargs.get("chunk_size_sec", 2.0),
                "min_chunk_samples": int(self.sample_rate * kwargs.get("chunk_size_sec", 2.0)),
            }

    def streaming_transcribe(self, audio_chunk: np.ndarray, state: Any) -> dict:
        """
        流式转录：送入一个音频块，返回增量识别结果

        vLLM 后端：原生逐块转录
        transformers 后端：累积后周期性转录
        """
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        backend = getattr(state, '_backend', None) or state.get('_backend', 'transformers')

        if backend == "vllm":
            # vLLM 原生流式
            self._model.streaming_transcribe(audio_chunk, state)
            return {
                "text": state.text or "",
                "language": state.language or "",
                "is_final": False,
            }
        else:
            # transformers 模拟流式：累积音频块
            state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_chunk])

            if len(state["audio_buffer"]) < state["min_chunk_samples"]:
                return {"text": state["last_text"], "language": self.language, "is_final": False}

            try:
                lang_map = {
                    "zh": "Chinese", "en": "English", "ja": "Japanese",
                    "ko": "Korean", "yue": "Cantonese", "ar": "Arabic",
                    "de": "German", "fr": "French", "es": "Spanish",
                    "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
                    "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
                }
                qwen_lang = lang_map.get(self.language, "Chinese")

                results = self._model.transcribe(
                    (state["audio_buffer"], self.sample_rate),
                    language=qwen_lang,
                    return_time_stamps=False,
                )
                text = ""
                if results and len(results) > 0:
                    text = (results[0].text or "").strip()
                if text != state["last_text"]:
                    state["last_text"] = text
            except Exception as e:
                logger.warning(f"[Qwen3-ASR] 流式转录中间结果失败: {e}")

            return {"text": state["last_text"], "language": self.language, "is_final": False}

    def finish_streaming_transcribe(self, state: Any) -> dict:
        """
        结束流式转录，获取最终结果

        vLLM 后端：原生 finish
        transformers 后端：对全部累积音频做最终转录
        """
        backend = getattr(state, '_backend', None) or state.get('_backend', 'transformers')

        if backend == "vllm":
            self._model.finish_streaming_transcribe(state)
            return {
                "text": state.text or "",
                "language": state.language or "",
                "is_final": True,
            }
        else:
            audio_buffer = state.get("audio_buffer", np.array([], dtype=np.float32))
            if len(audio_buffer) == 0:
                return {"text": state.get("last_text", ""), "language": self.language, "is_final": True}

            try:
                lang_map = {
                    "zh": "Chinese", "en": "English", "ja": "Japanese",
                    "ko": "Korean", "yue": "Cantonese", "ar": "Arabic",
                    "de": "German", "fr": "French", "es": "Spanish",
                    "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
                    "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
                }
                qwen_lang = lang_map.get(self.language, "Chinese")

                results = self._model.transcribe(
                    (audio_buffer, self.sample_rate),
                    language=qwen_lang,
                    return_time_stamps=False,
                )
                text = ""
                if results and len(results) > 0:
                    text = (results[0].text or "").strip()
            except Exception as e:
                logger.warning(f"[Qwen3-ASR] 最终转录失败: {e}")
                text = state.get("last_text", "")

            return {"text": text, "language": self.language, "is_final": True}
