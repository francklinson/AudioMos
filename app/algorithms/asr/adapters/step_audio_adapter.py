"""
Step-Audio-2-mini 适配器
端到端语音大模型，支持情感识别和工具调用

HuggingFace: stepfun-ai/Step-Audio-2-mini
GitHub: https://github.com/stepfun-ai/Step-Audio2
依赖: 从源码安装 Step-Audio2 或 pip install stepaudio
"""

import os
import logging
import tempfile
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult

logger = logging.getLogger("audiomos")


class StepAudioAdapter(BaseASR):
    """Step-Audio-2-mini 适配器 — 端到端语音大模型"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="step-audio-2-mini",
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
        for dirname in ["step-audio-2-mini", "Step-Audio-2-mini"]:
            candidate = os.path.join(project_root, "models", "asr", dirname)
            if os.path.exists(candidate) and os.listdir(candidate):
                return candidate
        return None

    def initialize(self) -> bool:
        try:
            # Step-Audio2 有自己的加载方式
            try:
                # 优先尝试step_audio包
                from step_audio import StepAudioModel
                return self._init_with_step_audio(StepAudioModel)
            except ImportError:
                pass

            # 回退: 使用transformers加载
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
                return self._init_with_transformers()
            except ImportError:
                pass

            raise ImportError(
                "Step-Audio未安装。请执行: "
                "pip install stepaudio 或从 https://github.com/stepfun-ai/Step-Audio2 安装"
            )

        except Exception as e:
            logger.error(f"[Step-Audio] 初始化失败: {e}")
            return False

    def _init_with_step_audio(self, ModelClass) -> bool:
        """使用step_audio官方包加载"""
        model_dir = self._find_model_dir()
        if not model_dir:
            raise FileNotFoundError("未找到Step-Audio-2-mini模型文件")

        logger.info(f"[Step-Audio] 使用step_audio从本地加载: {model_dir}")
        self._model = ModelClass.from_pretrained(
            model_dir,
            device=self.device,
        )
        self._load_mode = "step_audio"
        self._is_initialized = True
        logger.info("[Step-Audio] 模型初始化成功 (step_audio)")
        return True

    def _init_with_transformers(self) -> bool:
        """使用transformers加载"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = self._find_model_dir()
        if model_dir:
            logger.info(f"[Step-Audio] 使用transformers从本地加载: {model_dir}")
            model_path = model_dir
        else:
            logger.info("[Step-Audio] 从HuggingFace下载: stepfun-ai/Step-Audio-2-mini")
            model_path = "stepfun-ai/Step-Audio-2-mini"

        torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=bool(model_dir),
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            local_files_only=bool(model_dir),
            trust_remote_code=True,
        )
        self._model.eval()
        self._load_mode = "transformers"
        self._is_initialized = True
        logger.info("[Step-Audio] 模型初始化成功 (transformers)")
        return True

    def transcribe(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> ASRResult:
        if not self._is_initialized:
            raise RuntimeError("Step-Audio模型未初始化")

        # 保存临时WAV文件
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

        # 构建ASR请求的prompt
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

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        result_text = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return ASRResult(
            text=result_text,
            language=self.language,
            algorithm_name=self.name,
        )
