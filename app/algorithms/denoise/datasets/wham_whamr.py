"""
WHAM! / WHAMR! 语音分离数据集

WHAM! (WSJ0 Hipster Ambient Mixtures):
- 基于WSJ0-2mix的噪声增强版本
- 环境噪声混合

WHAMR! (WHAM with Reverb):
- 在WHAM!基础上添加混响

参考:
- Wichern et al., "WHAM!: Extending Speech Separation to Noisy Environments", INTERSPEECH 2019
- Maciejewski et al., "WHAMR!: Noisy and Reverberant Single-Channel Speech Separation", ICASSP 2020
- 数据集: https://wham.whisper.ai/

注意事项:
- WHAM! 基于 WSJ0 数据集，需要先获取 WSJ0 License (LDC)
- WHAM! 噪声部分可根据 CC BY-NC 4.0 许可使用
"""

import os
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import librosa

from .base import (
    BaseDataset,
    DatasetMeta,
    DatasetStatistics,
    DatasetCategory,
    SamplePair,
    SampleMetadata,
    NoiseType,
    SceneType,
    EvaluationSetConfig,
)

logger = logging.getLogger(__name__)


class WhamDataset(BaseDataset):
    """
    WHAM! / WHAMR! 数据集

    数据集组成:
    - wsj0-2mix/: WSJ0 2说话人混合 (需要LDC License)
    - wham_noise/: WHAM!环境噪声 (CC BY-NC 4.0)

    使用方式:
        dataset = WhamDataset(data_dir="./data/datasets/wham")
        if not dataset.is_downloaded:
            dataset.download()
        pairs = dataset.prepare_evaluation_set()
    """

    DEFAULT_SAMPLE_RATE = 8000  # WHAM! 默认使用8kHz

    KNOWN_BASELINES = {
        "sepformer_wham": {"sisdr": 20.8, "sdr": 21.2, "pesq": 3.11},
        "sepformer_whamr": {"sisdr": 14.0, "sdr": 14.4, "pesq": 2.52},
        "dprnn": {"sisdr": 18.8, "sdr": 19.1},
        "conv_tasnet": {"sisdr": 16.5, "sdr": 16.9},
    }

    def __init__(self, data_dir: str, sample_rate: int = 8000):
        super().__init__(data_dir, sample_rate)

    def get_meta(self) -> DatasetMeta:
        return DatasetMeta(
            name="WHAM! / WHAMR!",
            version="1.0",
            organization="MERL & Mitsubishi Electric",
            description="WSJ0混合数据集，WHAM!含环境噪声，WHAMR!额外添加混响",
            url="https://wham.whisper.ai/",
            license_info="WSJ0部分需LDC License；WHAM噪声CC BY-NC 4.0",
            citation="Wichern et al., 'WHAM!: Extending Speech Separation to Noisy Environments', INTERSPEECH 2019",
            total_duration_hours=40,
            total_files=30000,
            sample_rate=8000,
            categories=[DatasetCategory.CLEAN_SPEECH, DatasetCategory.NOISE, DatasetCategory.MIXED],
            known_baselines=self.KNOWN_BASELINES,
        )

    def download(self, force: bool = False) -> bool:
        """下载WHAM数据集（提供指引）"""
        if self.is_downloaded and not force:
            logger.info("WHAM 数据集已存在")
            return True

        os.makedirs(self.data_dir, exist_ok=True)
        logger.warning("WHAM! 数据集需要手动下载（依赖WSJ0 License）")
        logger.warning(self._get_download_instructions())

        print(self._get_download_instructions())
        self.save_metadata(self.get_meta())

        return self.is_downloaded

    def validate(self) -> Dict[str, bool]:
        results = {
            "directory_exists": os.path.exists(self.data_dir),
            "has_mixes": False,
            "has_noise": False,
        }

        if not results["directory_exists"]:
            return results

        for subdir in ["wsj0-2mix", "wham_noise"]:
            path = os.path.join(self.data_dir, subdir)
            if os.path.exists(path):
                files = self.scan_audio_files(path)
                results[f"has_{subdir.split('-')[-1]}"] = len(files) > 0

        return results

    def get_statistics(self) -> DatasetStatistics:
        stats = DatasetStatistics(
            name="WHAM!",
            version="1.0",
            data_dir=self.data_dir,
            exists=os.path.exists(self.data_dir),
        )

        if not stats.exists:
            return stats

        for subdir in ["wsj0-2mix", "wham_noise"]:
            path = os.path.join(self.data_dir, subdir)
            if os.path.exists(path):
                files = self.scan_audio_files(path)
                stats.categories[subdir] = {
                    "file_count": len(files),
                    "total_duration_hours": 0,  # 需要扫描
                }

        return stats

    def prepare_evaluation_set(
        self, config: Optional[EvaluationSetConfig] = None
    ) -> List[SamplePair]:
        """
        WHAM! 评测集准备

        如果没有下载WHAM数据，回退到内置合成数据。
        """
        if config is None:
            config = EvaluationSetConfig(
                n_samples=50, snr_levels=[0, 5, 10, 15], target_sample_rate=8000,
            )

        if not self.is_downloaded:
            logger.warning("WHAM 数据集未下载，回退到内置合成数据")
            from .dns import _BuiltinGenerator
            gen = _BuiltinGenerator(self.data_dir, self.sample_rate)
            return gen.generate(config)

        # 尝试收集已有的混合文件
        return self._collect_mixes(config)

    def _collect_mixes(self, config: EvaluationSetConfig) -> List[SamplePair]:
        """收集已有的混合音频配对"""
        pairs = []
        mix_dir = os.path.join(self.data_dir, "wsj0-2mix")

        if not os.path.exists(mix_dir):
            logger.warning("WSJ0混合数据不存在")
            return pairs

        # WHAM! 标准结构: mix/ s1/ s2/ noise/
        # 为简化，收集mix目录下的文件
        mix_files = self.scan_audio_files(os.path.join(mix_dir, "mix"))
        s1_files = self.scan_audio_files(os.path.join(mix_dir, "s1"))

        import random
        random.seed(config.seed)

        for i, mix_file in enumerate(mix_files[: config.n_samples]):
            clean_file = s1_files[i] if i < len(s1_files) else None

            pairs.append(SamplePair(
                noisy_path=mix_file,
                clean_path=clean_file,
                metadata=SampleMetadata(
                    noise_type=NoiseType.NON_STATIONARY,
                    original_source="wham",
                ),
            ))

        return pairs

    def _get_download_instructions(self) -> str:
        return f"""
# WHAM! / WHAMR! 数据集下载指引

## 前置条件
1. 获取 WSJ0 数据集 License:
   - 访问 https://catalog.ldc.upenn.edu/LDC93S6A
   - WSJ0 是商业数据集，需要购买

## WHAM! 下载
1. 访问 https://wham.whisper.ai/
2. 按照网站的生成脚本生成 WHAM! 数据
3. 或直接使用项目内置合成数据进行替代测试

## WHAMR! 下载
1. 访问 https://wham.whisper.ai/whamr/
2. 使用提供的Python脚本生成 WHAMR! 混响增强数据

## 替代方案
如果不能获取WSJ0许可证，可使用项目内置的 BuiltinDataset:
  from datasets.dns import BuiltinDataset
  dataset = BuiltinDataset()

## 预期目录结构
{self.data_dir}/
├── wsj0-2mix/
│   ├── mix/       # 混合音频
│   ├── s1/        # 说话人1
│   ├── s2/        # 说话人2
│   └── noise/     # 环境噪声
├── wham_noise/
│   └── ...
└── dataset_meta.json
"""
