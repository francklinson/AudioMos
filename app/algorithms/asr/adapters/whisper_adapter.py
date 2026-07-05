"""
Whisper适配器
支持 OpenAI Whisper large-v3-turbo，优先从本地模型目录加载
本地safetensors模型通过transformers加载，远程模型通过openai-whisper加载
"""

import os
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class WhisperAdapter(BaseASR):
    """Whisper Large-v3 Turbo 适配器"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="whisper-large-v3-turbo",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )
        self._processor = None

    def initialize(self) -> bool:
        try:
            model_dir = self.model_dir

            # 查找本地模型目录
            if not model_dir or not os.path.exists(model_dir):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )))
                candidate = os.path.join(project_root, "models", "asr", "whisper-large-v3-turbo")
                if os.path.exists(candidate):
                    model_dir = candidate

            # 方式1: 如果本地目录包含safetensors，使用transformers加载
            if model_dir and os.path.exists(os.path.join(model_dir, "model.safetensors")):
                logger.info(f"[Whisper] 从本地safetensors加载: {model_dir}")
                self._load_from_transformers(model_dir)
            # 方式2: 如果本地目录包含pt文件，使用openai-whisper加载
            elif model_dir and os.path.exists(os.path.join(model_dir, "model.pt")):
                logger.info(f"[Whisper] 从本地pt文件加载: {model_dir}")
                import whisper
                self._model = whisper.load_model(os.path.join(model_dir, "model.pt"), device=self.device)
            # 方式3: 使用模型名称从HuggingFace缓存加载
            elif model_dir and os.path.exists(model_dir) and os.listdir(model_dir):
                logger.info(f"[Whisper] 尝试transformers加载目录: {model_dir}")
                self._load_from_transformers(model_dir)
            # 方式4: 使用openai-whisper的模型名称下载
            else:
                logger.info(f"[Whisper] 从HuggingFace下载模型: large-v3-turbo")
                import whisper
                self._model = whisper.load_model("large-v3-turbo", device=self.device)

            self._is_initialized = True
            logger.info(f"[Whisper] 模型初始化成功")
            return True
        except Exception as e:
            logger.error(f"[Whisper] 初始化失败: {e}")
            return False

    def _load_from_transformers(self, model_dir: str):
        """使用transformers加载safetensors格式的Whisper模型"""
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True,
        ).to(self.device)
        self._processor = WhisperProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self._model.eval()

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Whisper模型未初始化")

        # float32输入
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 判断是transformers模型还是openai-whisper模型
        if self._processor is not None:
            return self._transcribe_transformers(audio)
        else:
            return self._transcribe_openai_whisper(audio)

    def _transcribe_transformers(self, audio: np.ndarray) -> ASRResult:
        """使用transformers pipeline识别"""
        import torch

        # 预处理
        input_features = self._processor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.to(self.device)

        # 生成
        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, language="chinese", task="transcribe")

        # 解码
        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        # 尝试获取分段（transformers不直接提供时间戳分段）
        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )

    def _transcribe_openai_whisper(self, audio: np.ndarray) -> ASRResult:
        """使用openai-whisper识别"""
        result = self._model.transcribe(
            audio,
            language="zh",
            task="transcribe",
            fp16=(self.device != "cpu"),
        )

        text = result.get("text", "").strip()
        segments = []
        for seg in result.get("segments", []):
            segments.append(ASRSegment(
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", "").strip(),
            ))

        return ASRResult(
            text=text,
            language=self.language,
            segments=segments if segments else None,
            algorithm_name=self.name,
        )
