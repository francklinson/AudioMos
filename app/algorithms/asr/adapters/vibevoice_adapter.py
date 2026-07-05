"""
VibeVoice-ASR-7B 适配器
超低帧率分词器+LLM，单次处理60分钟长音频
ICLR 2026 Oral

HuggingFace: microsoft/VibeVoice-ASR-HF
GitHub: https://github.com/microsoft/VibeVoice
依赖: pip install transformers torch soundfile
  需要transformers版本支持VibeVoiceAsrForConditionalGeneration
"""

import os
import logging
import tempfile
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class VibeVoiceAdapter(BaseASR):
    """VibeVoice-ASR-7B 适配器 — 超低帧率分词器 + LLM"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="vibevoice-asr",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )
        self._processor = None

    def _find_model_dir(self) -> Optional[str]:
        """查找本地模型目录"""
        if self.model_dir and os.path.exists(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))
        for dirname in ["vibevoice-asr", "VibeVoice-ASR-HF", "VibeVoice-ASR-7B"]:
            candidate = os.path.join(project_root, "models", "asr", dirname)
            if os.path.exists(candidate) and os.listdir(candidate):
                return candidate
        return None

    def initialize(self) -> bool:
        try:
            import torch

            model_dir = self._find_model_dir()

            # 尝试使用VibeVoice专用transformers类
            try:
                from transformers import AutoProcessor
                from transformers import VibeVoiceAsrForConditionalGeneration

                if model_dir:
                    logger.info(f"[VibeVoice] 从本地加载模型: {model_dir}")
                    model_path = model_dir
                else:
                    logger.info("[VibeVoice] 从HuggingFace下载: microsoft/VibeVoice-ASR-HF")
                    model_path = "microsoft/VibeVoice-ASR-HF"

                torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

                self._processor = AutoProcessor.from_pretrained(
                    model_path,
                    local_files_only=bool(model_dir),
                )
                self._model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    local_files_only=bool(model_dir),
                )
                self._load_mode = "vibevoice"
                self._is_initialized = True
                logger.info("[VibeVoice] 模型初始化成功 (VibeVoiceAsrForConditionalGeneration)")
                return True

            except (ImportError, AttributeError):
                # VibeVoiceAsrForConditionalGeneration不可用，尝试vibevoice包
                pass

            # 回退: 使用vibevoice社区包
            try:
                from vibevoice import VibeVoiceASR

                if model_dir:
                    logger.info(f"[VibeVoice] 使用vibevoice包从本地加载: {model_dir}")
                    self._model = VibeVoiceASR.from_pretrained(model_dir)
                else:
                    logger.info("[VibeVoice] 使用vibevoice包从HuggingFace下载")
                    self._model = VibeVoiceASR.from_pretrained("microsoft/VibeVoice-ASR-7B")

                self._load_mode = "vibevoice_pkg"
                self._is_initialized = True
                logger.info("[VibeVoice] 模型初始化成功 (vibevoice包)")
                return True

            except ImportError:
                raise ImportError(
                    "VibeVoice未安装。请执行:\n"
                    "  pip install transformers>=4.51.0 (支持VibeVoiceAsrForConditionalGeneration)\n"
                    "  或 pip install vibevoice"
                )

        except Exception as e:
            logger.error(f"[VibeVoice] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("VibeVoice模型未初始化")

        sr = sample_rate or self.sample_rate

        # VibeVoice需要文件路径输入
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
        ).to(self._model.device, self._model.dtype)

        output_ids = self._model.generate(**inputs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # 尝试解析格式
        try:
            result = self._processor.decode(generated_ids, return_format="parsed")[0]
            # 返回的是分段列表
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
            # 回退到纯文本解码
            text = self._processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            segments = None

        return ASRResult(
            text=text,
            language=self.language,
            segments=segments,
            algorithm_name=self.name,
        )

    def _transcribe_pkg(self, audio_path: str) -> ASRResult:
        """使用vibevoice社区包推理"""
        result = self._model.transcribe(audio_path, language="zh")

        text = ""
        segments = None
        if isinstance(result, dict):
            text = result.get("text", "")
            segs = result.get("segments", [])
            if segs:
                segments = [
                    ASRSegment(
                        start=s.get("start", 0),
                        end=s.get("end", 0),
                        text=s.get("text", ""),
                    )
                    for s in segs
                ]
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)

        return ASRResult(
            text=text.strip(),
            language=self.language,
            segments=segments,
            algorithm_name=self.name,
        )
