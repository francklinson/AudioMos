"""
Step-Audio-2-mini 适配器
支持 step_audio 官方包 和 transformers 两种加载方式

HuggingFace: stepfun-ai/Step-Audio-2-mini
依赖: pip install step-audio 或 pip install transformers
"""

import os
import logging
import tempfile
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult

logger = logging.getLogger("audiomos")


class StepAudioAdapter(BaseASR):
    """Step-Audio-2-mini 适配器"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None,
                 offline: bool = True, **kwargs):
        super().__init__(
            name="step-audio-2-mini",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
            offline=offline,
        )
        self._load_mode = None
        self._tokenizer = None

    def _find_model_dir(self) -> Optional[str]:
        """查找本地模型目录（仅检查 self.model_dir）"""
        if self.model_dir and os.path.isdir(self.model_dir) and os.listdir(self.model_dir):
            return self.model_dir
        return None

    def initialize(self) -> bool:
        try:
            model_dir = self._find_model_dir()

            # 尝试使用 step_audio 官方包
            if model_dir is None and not self.offline:
                model_dir = "stepfun-ai/Step-Audio-2-mini"

            if model_dir is None:
                raise FileNotFoundError(
                    f"[Step-Audio] 离线模式：未找到本地模型，请放置在 {self.model_dir}"
                )

            try:
                import step_audio
                logger.info(f"[Step-Audio] 使用step_audio官方包加载: {model_dir}")
                self._model = step_audio.AutoModel.from_pretrained(model_dir)
                self._load_mode = "step_audio"
            except ImportError:
                logger.info("[Step-Audio] step_audio未安装，使用transformers加载")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                self._model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map="auto" if self.device != "cpu" else None,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                self._load_mode = "transformers"

            self._is_initialized = True
            logger.info(f"[Step-Audio] 模型初始化成功 (mode={self._load_mode})")
            return True

        except Exception as e:
            logger.error(f"[Step-Audio] 初始化失败: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Step-Audio模型未初始化")

        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            sr = sample_rate or self.sample_rate
            sf.write(temp_path, audio, sr)

            if self._load_mode == "step_audio":
                return self._transcribe_step_audio(temp_path)
            else:
                return self._transcribe_transformers(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _transcribe_step_audio(self, audio_path: str) -> ASRResult:
        """使用step_audio官方包推理"""
        result = self._model.transcribe(audio_path, language="zh")

        text = ""
        if isinstance(result, dict):
            text = result.get("text", "")
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)
        text = text.strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )

    def _transcribe_transformers(self, audio_path: str) -> ASRResult:
        """使用transformers推理"""
        import torch

        prompt = "请将以下音频内容转换为文字，只输出识别结果："
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=2048,
            )

        result_ids = generated_ids[:, model_inputs["input_ids"].shape[1]:]
        result_text = self._tokenizer.batch_decode(result_ids, skip_special_tokens=True)[0].strip()

        return ASRResult(
            text=result_text,
            language=self.language,
            algorithm_name=self.name,
        )
