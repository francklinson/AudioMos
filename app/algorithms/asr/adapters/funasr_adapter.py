"""
FunASR适配器
支持 Paraformer-large 和 SenseVoice-Small
模型文件从项目本地 models/asr/ 目录加载

FunASR 1.3+ 本地加载方式:
  1. 设置 MODELSCOPE_CACHE 环境变量指向 models/asr/modelscope_cache
  2. 使用注册名 + hub='ms' 加载
  3. FunASR会自动从 MODELSCOPE_CACHE 目录查找已下载的模型
"""

import os
import logging
from typing import Optional
import numpy as np

from ..base import BaseASR, ASRResult, ASRSegment

logger = logging.getLogger("audiomos")

# 设置模型缓存到项目本地目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)))
_LOCAL_CACHE = os.path.join(_PROJECT_ROOT, "models", "asr", "modelscope_cache")
os.environ["MODELSCOPE_CACHE"] = _LOCAL_CACHE

# ModelScope模型ID到FunASR注册名的映射
_MODEL_ID_MAP = {
    "paraformer-large": {
        "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "register_name": "Paraformer",  # FunASR注册名
    },
    "sensevoice-small": {
        "model_id": "iic/SenseVoiceSmall",
        "register_name": "SenseVoiceSmall",  # FunASR注册名
    },
}


def _setup_local_model_cache(algo_name: str, model_dir: str):
    """
    将本地模型目录链接到MODELSCOPE_CACHE/hub/<model_id>/，
    使FunASR AutoModel可以用hub='local'加载
    """
    info = _MODEL_ID_MAP.get(algo_name)
    if not info:
        return model_dir

    model_id = info["model_id"]
    cache_model_dir = os.path.join(_LOCAL_CACHE, "hub", model_id)

    if os.path.exists(cache_model_dir) and os.listdir(cache_model_dir):
        return model_dir  # 缓存目录已有模型

    # 从model_dir创建软链接到缓存目录
    os.makedirs(os.path.dirname(cache_model_dir), exist_ok=True)

    if model_dir and os.path.exists(model_dir) and os.listdir(model_dir):
        if os.path.islink(cache_model_dir):
            os.unlink(cache_model_dir)
        elif os.path.exists(cache_model_dir):
            import shutil
            shutil.rmtree(cache_model_dir)
        os.symlink(os.path.abspath(model_dir), cache_model_dir)
        logger.info(f"[FunASR] 已链接本地模型: {model_dir} → {cache_model_dir}")

    return model_dir


class ParaformerAdapter(BaseASR):
    """Paraformer-large 适配器 — 通过FunASR加载"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="paraformer-large",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )

    def initialize(self) -> bool:
        try:
            from funasr import AutoModel

            # 使用ModelScope模型ID加载（不是FunASR注册名）
            info = _MODEL_ID_MAP["paraformer-large"]
            logger.info(f"[Paraformer] 从ModelScope缓存加载: {info['model_id']}")
            logger.info(f"[Paraformer] MODELSCOPE_CACHE: {_LOCAL_CACHE}")
            
            # 直接使用ModelScope模型ID
            self._model = AutoModel(
                model=info["model_id"],  # 使用完整的ModelScope ID
                hub="ms",
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


class SenseVoiceAdapter(BaseASR):
    """SenseVoice-Small 适配器 — 通过FunASR加载"""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None, **kwargs):
        super().__init__(
            name="sensevoice-small",
            sample_rate=16000,
            device=device,
            language="zh",
            model_dir=model_dir,
        )

    def initialize(self) -> bool:
        try:
            from funasr import AutoModel

            # 使用ModelScope模型ID加载（不是FunASR注册名）
            info = _MODEL_ID_MAP["sensevoice-small"]
            logger.info(f"[SenseVoice] 从ModelScope缓存加载: {info['model_id']}")
            logger.info(f"[SenseVoice] MODELSCOPE_CACHE: {_LOCAL_CACHE}")
            
            # 直接使用ModelScope模型ID
            self._model = AutoModel(
                model=info["model_id"],  # 使用完整的ModelScope ID
                hub="ms",
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
        import re
        text = re.sub(r'<\|[^|]*\|>', '', text).strip()

        return ASRResult(
            text=text,
            language=self.language,
            algorithm_name=self.name,
        )
