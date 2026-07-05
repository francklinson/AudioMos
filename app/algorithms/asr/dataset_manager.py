"""
ASR标准数据集管理模块
支持AISHELL-1、WenetSpeech、Thchs30等标准中文ASR测试数据集
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger("audiomos")


@dataclass
class DatasetSample:
    """数据集单条样本"""
    audio_path: str          # 音频文件路径
    reference_text: str      # 参考文本
    speaker_id: str = ""     # 说话人ID
    utterance_id: str = ""   # 语句ID
    duration: float = 0.0    # 音频时长(秒)
    extra: Dict = None       # 额外信息

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class BaseASRDataset(ABC):
    """ASR数据集基类"""

    def __init__(self, name: str, data_dir: str):
        self.name = name
        self.data_dir = data_dir

    @abstractmethod
    def get_test_split(self, max_samples: Optional[int] = None) -> List[DatasetSample]:
        """获取测试集"""
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """获取数据集元信息"""
        pass

    def is_available(self) -> bool:
        """检查数据集是否可用"""
        return os.path.exists(self.data_dir)


class AISHELL1Dataset(BaseASRDataset):
    """AISHELL-1 数据集适配器"""

    def __init__(self, data_dir: str):
        super().__init__("AISHELL-1", data_dir)

    def get_test_split(self, max_samples: Optional[int] = None) -> List[DatasetSample]:
        test_dir = os.path.join(self.data_dir, "test", "wav")
        transcript_file = os.path.join(self.data_dir, "test", "transcript", "aishell1_test.txt")

        if not os.path.exists(test_dir):
            logger.warning(f"[AISHELL-1] 测试集目录不存在: {test_dir}")
            return []

        # 解析transcript
        transcripts = {}
        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        utt_id = parts[0]
                        text = " ".join(parts[1:])
                        transcripts[utt_id] = text

        samples = []
        for root, dirs, files in os.walk(test_dir):
            for f in sorted(files):
                if f.endswith(".wav"):
                    utt_id = os.path.splitext(f)[0]
                    audio_path = os.path.join(root, f)
                    ref_text = transcripts.get(utt_id, "")

                    samples.append(DatasetSample(
                        audio_path=audio_path,
                        reference_text=ref_text,
                        utterance_id=utt_id,
                    ))

                    if max_samples and len(samples) >= max_samples:
                        return samples

        logger.info(f"[AISHELL-1] 加载 {len(samples)} 条测试样本")
        return samples

    def get_info(self) -> dict:
        return {
            "name": "AISHELL-1",
            "description": "AISHELL-1 中文普通话语音语料库，170小时朗读语音",
            "hours": 170,
            "speakers": 400,
            "type": "朗读",
            "license": "Apache 2.0",
            "available": self.is_available(),
        }


class WenetSpeechDataset(BaseASRDataset):
    """WenetSpeech 数据集适配器"""

    def __init__(self, data_dir: str):
        super().__init__("WenetSpeech", data_dir)

    def get_test_split(self, max_samples: Optional[int] = None) -> List[DatasetSample]:
        # WenetSpeech使用M4A格式，需先转换为WAV
        meta_file = os.path.join(self.data_dir, "wenetspeech_test.json")

        if not os.path.exists(meta_file):
            logger.warning(f"[WenetSpeech] 测试集元数据不存在: {meta_file}")
            return []

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        samples = []
        for item in meta.get("utts", []):
            audio_path = item.get("audio", "")
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(self.data_dir, audio_path)

            samples.append(DatasetSample(
                audio_path=audio_path,
                reference_text=item.get("text", ""),
                utterance_id=item.get("utt_id", ""),
                duration=item.get("duration", 0),
            ))

            if max_samples and len(samples) >= max_samples:
                return samples

        logger.info(f"[WenetSpeech] 加载 {len(samples)} 条测试样本")
        return samples

    def get_info(self) -> dict:
        return {
            "name": "WenetSpeech",
            "description": "WenetSpeech 10000+小时中文语音语料库，含朗读/演讲/会议",
            "hours": 10000,
            "speakers": 20000,
            "type": "朗读+演讲+会议",
            "license": "CC BY-NC-ND 4.0",
            "available": self.is_available(),
        }


class Thchs30Dataset(BaseASRDataset):
    """THCHS-30 数据集适配器"""

    def __init__(self, data_dir: str):
        super().__init__("THCHS-30", data_dir)

    def get_test_split(self, max_samples: Optional[int] = None) -> List[DatasetSample]:
        test_dir = os.path.join(self.data_dir, "test")
        if not os.path.exists(test_dir):
            # THCHS-30可能直接在data_dir下
            test_dir = self.data_dir

        samples = []
        for f in sorted(os.listdir(test_dir)):
            if f.endswith(".wav"):
                base = os.path.splitext(f)[0]
                audio_path = os.path.join(test_dir, f)
                trn_path = os.path.join(test_dir, base + ".trn")

                ref_text = ""
                if os.path.exists(trn_path):
                    with open(trn_path, "r", encoding="utf-8") as tf:
                        lines = tf.readlines()
                        if lines:
                            ref_text = lines[0].strip()

                samples.append(DatasetSample(
                    audio_path=audio_path,
                    reference_text=ref_text,
                    utterance_id=base,
                ))

                if max_samples and len(samples) >= max_samples:
                    return samples

        logger.info(f"[THCHS-30] 加载 {len(samples)} 条测试样本")
        return samples

    def get_info(self) -> dict:
        return {
            "name": "THCHS-30",
            "description": "THCHS-30 清华大学30小时中文语音数据集",
            "hours": 30,
            "speakers": 30,
            "type": "朗读",
            "license": "免费学术使用",
            "available": self.is_available(),
        }


class BuiltInTestDataset(BaseASRDataset):
    """项目内置测试集 — 用于快速验证"""

    def __init__(self, data_dir: str):
        super().__init__("内置测试集", data_dir)

    def get_test_split(self, max_samples: Optional[int] = None) -> List[DatasetSample]:
        if not os.path.exists(self.data_dir):
            logger.warning(f"[内置测试集] 目录不存在: {self.data_dir}")
            return []

        samples = []
        for f in sorted(os.listdir(self.data_dir)):
            if f.endswith(".wav"):
                audio_path = os.path.join(self.data_dir, f)
                # 查找对应文本文件
                base = os.path.splitext(f)[0]
                txt_path = os.path.join(self.data_dir, base + ".txt")
                ref_text = ""
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as tf:
                        ref_text = tf.read().strip()

                samples.append(DatasetSample(
                    audio_path=audio_path,
                    reference_text=ref_text,
                    utterance_id=base,
                ))

                if max_samples and len(samples) >= max_samples:
                    return samples

        logger.info(f"[内置测试集] 加载 {len(samples)} 条测试样本")
        return samples

    def get_info(self) -> dict:
        return {
            "name": "内置测试集",
            "description": "项目内置ASR测试数据，用于快速验证",
            "hours": 0.5,
            "type": "混合",
            "available": self.is_available(),
        }


# ==================== 数据集管理器 ====================

class DatasetManager:
    """数据集管理器 — 统一管理所有数据集"""

    def __init__(self, datasets_config: Optional[dict] = None):
        self._datasets: Dict[str, BaseASRDataset] = {}
        self._config = datasets_config or {}

    def register_dataset(self, name: str, dataset: BaseASRDataset):
        """注册数据集"""
        self._datasets[name] = dataset
        logger.info(f"数据集已注册: {name}")

    def get_dataset(self, name: str) -> Optional[BaseASRDataset]:
        """获取数据集"""
        return self._datasets.get(name)

    def list_datasets(self) -> List[dict]:
        """列出所有数据集"""
        return [ds.get_info() for ds in self._datasets.values()]

    def get_test_samples(self, name: str, max_samples: Optional[int] = None) -> List[DatasetSample]:
        """获取指定数据集的测试样本"""
        ds = self._datasets.get(name)
        if ds:
            return ds.get_test_split(max_samples)
        return []

    @classmethod
    def from_config(cls, config: dict, project_root: str = ".") -> "DatasetManager":
        """从配置创建数据集管理器"""
        manager = cls(config)
        datasets_dir = os.path.join(project_root, "data", "datasets")

        # 注册标准数据集
        dataset_map = {
            "aishell1_test": AISHELL1Dataset,
            "wenetspeech_test": WenetSpeechDataset,
            "thchs30_test": Thchs30Dataset,
            "builtin": BuiltInTestDataset,
        }

        for key, cls_type in dataset_map.items():
            ds_config = config.get(key, {})
            ds_dir = ds_config.get("path", os.path.join(datasets_dir, key))
            manager.register_dataset(key, cls_type(ds_dir))

        return manager
