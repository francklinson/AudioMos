"""
VibeVoice-ASR 适配器
支持 VibeVoice 官方包 和 transformers 两种加载方式

HuggingFace: microsoft/VibeVoice-ASR-HF
依赖: pip install vibevoice-asr 或 pip install transformers
"""

import os
import logging
import tempfile
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class VibeVoiceAdapter(BaseASR):
    """VibeVoice-ASR 适配器"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None,
                 offline: bool = True, **kwargs):
        super().__init__(
            name="vibevoice-asr",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
            offline=offline,
        )
        self._load_mode = None
        self._processor = None

    def _find_model_dir(self) -> Optional[str]:
        """查找本地模型目录（仅检查 self.model_dir）"""
        if self.model_dir and os.path.isdir(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
        return None

    def initialize(self) -> bool:
        try:
            model_dir = self._find_model_dir()

            if model_dir is None:
                if self.offline:
                    raise FileNotFoundError(
                        f"[VibeVoice] 离线模式：未找到本地模型，请放置在 {self.model_dir}"
                    )

            # 尝试使用 vibevoice 官方包
            if model_dir is None:
                model_dir = "microsoft/VibeVoice-ASR-HF"

            try:
                import vibevoice_asr
                logger.info(f"[VibeVoice] 使用VibeVoice官方包加载: {model_dir}")
                self._model = vibevoice_asr.VibeVoiceAsrForConditionalGeneration.from_pretrained(
                    model_dir,
                )
                self._processor = vibevoice_asr.VibeVoiceProcessor.from_pretrained(model_dir)
                self._load_mode = "vibevoice"
            except ImportError:
                logger.info("[VibeVoice] vibevoice_asr未安装，使用transformers加载")
                from transformers import AutoModelForCausalLM, AutoProcessor
                import torch

                if model_dir is None:
                    model_dir = "microsoft/VibeVoice-ASR-7B"

                self._model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self._processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                self._load_mode = "transformers"

            self._is_initialized = True
            logger.info(f"[VibeVoice] 模型初始化成功 (mode={self._load_mode})")
            return True

        except Exception as e:
            logger.error(f"[VibeVoice] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("VibeVoice模型未初始化")

        sr = sample_rate or self.sample_rate

        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, audio, sr)

            if self._load_mode == "vibevoice":
                return self._transcribe_native(temp_path)
            else:
                return self._transcribe_pkg(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _transcribe_native(self, audio_path: str) -> ASRResult:
        """使用VibeVoiceAsrForConditionalGeneration推理"""
        import torch

        inputs = self._processor.apply_transcription_request(
            audio=audio_path,
            prompt=None,
        ).to(self._model.device, torch.float16 if self.device != "cpu" else torch.float32)

        output_ids = self._model.generate(**inputs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        try:
            result = self._processor.decode(generated_ids, return_format="parsed")[0]
            text_parts = []
            segments = []
            for seg in result:
                if isinstance(seg, dict):
                    content = seg.get("Content", seg.get("text", ""))
                    text_parts.append(content)
                    segments.append(ASRSegment(
                        start=seg.get("Start", 0),
                        end=seg.get("End", 0),
                        text=content,
                    ))
            text = "".join(text_parts).strip()
        except Exception:
            text = self._processor.decode(generated_ids[0], skip_special_tokens=True).strip()

        return ASRResult(
            text=text,
            language=self.language,
            segments=segments if segments else None,
            algorithm_name=self.name,
        )

    def _transcribe_pkg(self, audio_path: str) -> ASRResult:
        """使用transformers加载的模型推理"""
        import torch

        inputs = self._processor.apply_transcription_request(
            audio=audio_path,
            prompt=None,
        ).to(self._model.device, torch.float16 if self.device != "cpu" else torch.float32)

        output_ids = self._model.generate(**inputs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        try:
            result = self._processor.decode(generated_ids, return_format="parsed")[0]
            text_parts = []
            for seg in result:
                if isinstance(seg, dict):
                    text_parts.append(seg.get("Content", seg.get("text", "")))
            text = "".join(text_parts).strip()
        except Exception:
            text = self._processor.decode(generated_ids[0], skip_special_tokens=True).strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )
