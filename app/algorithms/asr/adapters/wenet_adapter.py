"""
WeNet适配器
复用项目已有的wenet模块，从本地模型目录加载
"""

import os
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")


class WeNetAdapter(BaseASR):
    """WeNet U2++ 适配器 — 复用项目已有wenet模块"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="wenet-u2pp",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )

    def initialize(self) -> bool:
        try:
            import wenet

            # 查找模型文件
            model_path = self.model_dir
            if not model_path or not os.path.exists(model_path):
                # 尝试项目默认模型路径
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )))
                candidate_paths = [
                    os.path.join(project_root, "models", "asr", "wenet-u2pp"),
                    os.path.join(project_root, "models", "asr", "wenet"),
                ]
                for p in candidate_paths:
                    if os.path.exists(p):
                        model_path = p
                        break

            if not model_path or not os.path.exists(model_path):
                logger.warning(f"[WeNet] 未找到本地模型，将使用wenet hub下载")
                self._model = wenet.CLModel(language="zh")
            else:
                logger.info(f"[WeNet] 从本地加载模型: {model_path}")
                # 查找zip或目录中的模型文件
                model_file = self._find_model_file(model_path)
                if model_file:
                    self._model = wenet.CLModel(model_path=model_file)
                else:
                    self._model = wenet.CLModel(language="zh")

            self._is_initialized = True
            logger.info(f"[WeNet] 模型初始化成功")
            return True
        except Exception as e:
            logger.error(f"[WeNet] 初始化失败: {e}")
            return False

    def _find_model_file(self, model_dir: str) -> Optional[str]:
        """在模型目录中查找模型文件"""
        # WeNet模型通常是 final.zip 或 avg_best.pt
        for name in ["final.zip", "avg_best.pt", "model.zip", "model.pt"]:
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                return path
        # 查找子目录
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                if f.endswith(".zip") or f.endswith(".pt"):
                    return os.path.join(root, f)
        return None

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
            text = result.strip() if isinstance(result, str) else str(result)
        finally:
            os.unlink(temp_path)

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )
