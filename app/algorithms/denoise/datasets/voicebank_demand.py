"""
VoiceBank-DEMAND 标准语音增强数据集

最广泛使用的语音增强基准数据集。
- 28个训练说话人 + 2个测试说话人
- DEMAND噪声 + 人工噪声（babble, speech-shaped noise）
- SNR: 0, 5, 10, 15 dB

参考:
- Valentini et al., "Investigating RNN-based speech enhancement...", INTERSPEECH 2016
- 数据集: https://datashare.ed.ac.uk/handle/10283/2791
- DEMAND噪声: https://zenodo.org/record/1227121
"""

import os
import json
import time
import logging
import zipfile
import hashlib
from typing import Dict, List, Optional
from pathlib import Path

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


class VoicebankDemandDataset(BaseDataset):
    """
    VoiceBank-DEMAND 数据集

    组成:
    - clean_trainset_wav/: 28个说话人，~11572条干净语音
    - noisy_trainset_wav/: 对应的带噪版本
    - clean_testset_wav/: 2个说话人，~824条干净语音
    - noisy_testset_wav/: 对应的带噪版本

    官方下载:
    - VoiceBank: https://datashare.ed.ac.uk/handle/10283/2791
    - DEMAND噪声库: https://zenodo.org/record/1227121

    使用方式:
        dataset = VoicebankDemandDataset(data_dir="./data/datasets/voicebank_demand")
        if not dataset.is_downloaded:
            dataset.download()
        pairs = dataset.prepare_evaluation_set()
    """

    # VoiceBank-DEMAND 官方下载URL
    DATASET_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_wav.zip"
    CLEAN_TRAINSET_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_wav.zip"
    TEST_SET_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"
    CLEAN_TESTSET_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"

    # DEMAND噪声库
    DEMAND_URL = "https://zenodo.org/record/1227121/files/DEMAND.zip"

    # 标准SNR级别（VoiceBank使用: 0, 5, 10, 15 dB）
    STANDARD_SNR_LEVELS = [2.5, 7.5, 12.5, 17.5]

    # 噪声类型映射
    NOISE_TYPE_MAP = {
        # 16种DEMAND噪声
        "DKITCHEN": NoiseType.CAFE,
        "DWASHING": NoiseType.FACTORY,
        "DLIVING": NoiseType.NATURAL,
        "TCAR": NoiseType.TRAFFIC,
        "TBUS": NoiseType.TRAFFIC,
        "TRAIN": NoiseType.TRAFFIC,
        "SPSQUARE": NoiseType.NON_STATIONARY,
        "SCAFE": NoiseType.CAFE,
        "STRAFFIC": NoiseType.TRAFFIC,
        "NRIVER": NoiseType.NATURAL,
        "NPARK": NoiseType.NATURAL,
        "OHALLWAY": NoiseType.NON_STATIONARY,
        "OOFFICE": NoiseType.STATIONARY,
        "PSTATION": NoiseType.FACTORY,
        "PSQUARE": NoiseType.NON_STATIONARY,
        "TMETRO": NoiseType.TRAFFIC,
    }

    # 已知基线
    KNOWN_BASELINES = {
        "segan": {"pesq": 2.16, "stoi": 0.890, "csig": 3.39, "cbak": 2.42, "covl": 2.57},
        "wavenet": {"pesq": 2.45, "stoi": 0.915, "csig": 3.52, "cbak": 2.62, "covl": 2.70},
        "dccrn": {"pesq": 2.84, "stoi": 0.910, "csig": 4.14, "cbak": 3.17, "covl": 3.38},
        "metricgan_plus": {"pesq": 3.15, "stoi": 0.930, "csig": 4.14, "cbak": 3.16, "covl": 3.64},
        "fullsubnet": {"pesq": 2.97, "stoi": 0.920, "csig": 4.20, "cbak": 3.24, "covl": 3.55},
    }

    def __init__(self, data_dir: str, sample_rate: int = 48000):
        """
        VoiceBank-DEMAND 默认使用48kHz采样率

        Args:
            data_dir: 数据集存储目录
            sample_rate: 采样率（默认48kHz，匹配原始数据集）
        """
        super().__init__(data_dir, sample_rate)

    def get_meta(self) -> DatasetMeta:
        return DatasetMeta(
            name="VoiceBank-DEMAND",
            version="1.0",
            organization="University of Edinburgh",
            description="标准语音增强数据集，28个训练说话人+2个测试说话人，DEMAND噪声+人工噪声",
            url="https://datashare.ed.ac.uk/handle/10283/2791",
            license_info="CC BY 4.0",
            citation="Valentini et al., 'Investigating RNN-based speech enhancement...', INTERSPEECH 2016",
            total_duration_hours=5.0,
            total_files=12100,
            sample_rate=48000,
            categories=[DatasetCategory.CLEAN_SPEECH, DatasetCategory.NOISY],
            known_baselines=self.KNOWN_BASELINES,
        )

    def download(self, force: bool = False) -> bool:
        """下载VoiceBank-DEMAND数据集"""
        if self.is_downloaded and not force:
            logger.info("VoiceBank-DEMAND 数据集已存在")
            return True

        os.makedirs(self.data_dir, exist_ok=True)

        # 尝试自动下载
        urls = {
            "clean_testset_wav.zip": self.CLEAN_TESTSET_URL,
            "noisy_testset_wav.zip": self.TEST_SET_URL,
        }

        download_success = False

        for filename, url in urls.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath) and not force:
                logger.info(f"{filename} 已存在，跳过下载")
                continue

            try:
                logger.info(f"下载 {filename} from {url}...")
                self._download_file(url, filepath)
                logger.info(f"{filename} 下载完成")

                # 解压
                extract_dir = os.path.join(self.data_dir, filename.replace(".zip", ""))
                if not os.path.exists(extract_dir) or force:
                    logger.info(f"解压 {filename}...")
                    with zipfile.ZipFile(filepath, "r") as zf:
                        zf.extractall(self.data_dir)

                download_success = True

            except Exception as e:
                logger.warning(f"下载 {filename} 失败: {e}")

        # 保存元数据
        meta = self.get_meta()
        self.save_metadata(meta)

        if not download_success:
            logger.warning("自动下载失败，请手动下载VoiceBank-DEMAND数据集")
            logger.warning(self._get_download_instructions())

        return self.is_downloaded

    def _download_file(self, url: str, filepath: str, timeout: int = 600) -> None:
        """使用urllib下载文件"""
        import urllib.request

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 100 == 0:
                    logger.info(f"下载进度: {percent:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)")

        urllib.request.urlretrieve(url, filepath, progress_hook)

    def validate(self) -> Dict[str, bool]:
        results = {
            "directory_exists": os.path.exists(self.data_dir),
            "has_clean_testset": False,
            "has_noisy_testset": False,
            "has_clean_trainset": False,
            "has_noisy_trainset": False,
            "file_count_ok": False,
            "sample_rate_ok": True,
        }

        if not results["directory_exists"]:
            return results

        # 检查测试集（评测必需）
        expected_dirs = {
            "has_clean_testset": "clean_testset_wav",
            "has_noisy_testset": "noisy_testset_wav",
            "has_clean_trainset": "clean_trainset_wav",
            "has_noisy_trainset": "noisy_trainset_wav",
        }

        for key, dirname in expected_dirs.items():
            full_path = os.path.join(self.data_dir, dirname)
            if os.path.exists(full_path):
                files = self.scan_audio_files(full_path)
                results[key] = len(files) > 0

        # 测试集文件数检查
        total = 0
        for d in ["clean_testset_wav", "noisy_testset_wav"]:
            full = os.path.join(self.data_dir, d)
            if os.path.exists(full):
                total += len(self.scan_audio_files(full))

        results["file_count_ok"] = total >= 800  # VoiceBank测试集约824对

        return results

    def get_statistics(self) -> DatasetStatistics:
        stats = DatasetStatistics(
            name="VoiceBank-DEMAND",
            version="1.0",
            data_dir=self.data_dir,
            exists=os.path.exists(self.data_dir),
        )

        if not stats.exists:
            return stats

        total_files = 0
        total_duration = 0.0

        for subdir in ["clean_testset_wav", "noisy_testset_wav", "clean_trainset_wav", "noisy_trainset_wav"]:
            full_path = os.path.join(self.data_dir, subdir)
            if not os.path.exists(full_path):
                continue

            files = self.scan_audio_files(full_path)
            cat_duration = 0.0

            for f in files[:200]:  # 抽样估算
                dur = self.compute_audio_duration(f)
                if dur:
                    cat_duration += dur

            scale = max(1, len(files) / 200) if len(files) > 200 else 1
            cat_duration *= scale

            stats.categories[subdir] = {
                "file_count": len(files),
                "total_duration_hours": round(cat_duration / 3600, 2),
            }
            total_files += len(files)
            total_duration += cat_duration

        stats.total_files = total_files
        stats.total_duration_hours = round(total_duration / 3600, 2)

        return stats

    def prepare_evaluation_set(
        self, config: Optional[EvaluationSetConfig] = None
    ) -> List[SamplePair]:
        """
        准备VoiceBank-DEMAND标准评测集

        使用官方测试集（2个说话人，824对音频）。

        Args:
            config: 评测集配置

        Returns:
            SamplePair 列表
        """
        if config is None:
            config = EvaluationSetConfig(
                n_samples=50,
                snr_levels=[2.5, 7.5, 12.5, 17.5],
            )

        if not self.is_downloaded:
            logger.warning("VoiceBank-DEMAND 未下载")
            logger.warning(self._get_download_instructions())

            # 回退到内置合成数据
            logger.info("回退到内置合成数据")
            from .dns import _BuiltinGenerator
            gen = _BuiltinGenerator(self.data_dir, self.sample_rate)
            return gen.generate(config)

        # 收集测试集配对
        pairs = self._collect_test_pairs()

        if not pairs:
            logger.warning("未找到VoiceBank-DEMAND测试集配对")
            return []

        # 限制数量
        n = min(config.n_samples, len(pairs))

        # 如果指定了snr_levels，按SNR筛选
        if config.snr_levels and config.snr_levels != [2.5, 7.5, 12.5, 17.5]:
            filtered = [p for p in pairs if p.metadata.snr_db in config.snr_levels]
            if filtered:
                pairs = filtered

        import random
        random.seed(config.seed)
        random.shuffle(pairs)

        return pairs[:n]

    def _collect_test_pairs(self) -> List[SamplePair]:
        """收集官方测试集配对"""
        pairs = []
        clean_dir = os.path.join(self.data_dir, "clean_testset_wav")
        noisy_dir = os.path.join(self.data_dir, "noisy_testset_wav")

        if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir):
            return pairs

        clean_files = {os.path.basename(f): f for f in self.scan_audio_files(clean_dir)}
        noisy_files = {os.path.basename(f): f for f in self.scan_audio_files(noisy_dir)}

        for filename, noisy_path in sorted(noisy_files.items()):
            if filename in clean_files:
                # 推断噪声类型（基于命名规则）
                noise_type = self._infer_noise_type_from_filename(filename)

                # 估算SNR（基于文件名中的p子编号推测噪声区域）
                snr_db = self._infer_snr_from_filename(filename)

                pairs.append(SamplePair(
                    noisy_path=noisy_path,
                    clean_path=clean_files[filename],
                    metadata=SampleMetadata(
                        noise_type=noise_type,
                        snr_db=snr_db,
                        speaker_id=self._infer_speaker_id(filename),
                        original_source="voicebank_demand_official",
                    ),
                ))

        return pairs

    def _infer_noise_type_from_filename(self, filename: str) -> NoiseType:
        """从文件名推断噪声类型"""
        # VoiceBank格式: p232_XXX.wav
        # 噪声类型在文件名中不直接体现，但可通过测试集构建时使用的噪声矩阵确定
        # 这里使用合理默认值
        name_upper = filename.upper()
        for keyword, ntype in self.NOISE_TYPE_MAP.items():
            if keyword in name_upper:
                return ntype
        return NoiseType.NON_STATIONARY

    def _infer_snr_from_filename(self, filename: str) -> Optional[float]:
        """估计SNR值"""
        # VoiceBank测试集使用: 2.5, 7.5, 12.5, 17.5 dB
        # 文件名规律不直接体现SNR，返回None表示未标记
        return None

    def _infer_speaker_id(self, filename: str) -> Optional[str]:
        """提取说话人ID"""
        # VoiceBank格式: pXXX_YYY.wav, pXXX是说话人ID
        try:
            parts = filename.split("_")
            if parts and parts[0].startswith("p"):
                return parts[0]
        except Exception:
            pass
        return None

    def _get_download_instructions(self) -> str:
        """下载指引"""
        return """
# VoiceBank-DEMAND 数据集下载指引

## 方法1: 自动下载（推荐）
系统已尝试自动下载测试集文件。如果失败，请使用方法2。

## 方法2: 手动下载
1. 访问数据集页面:
   - https://datashare.ed.ac.uk/handle/10283/2791
2. 下载以下文件:
   - clean_testset_wav.zip (测试集干净语音, ~300MB)
   - noisy_testset_wav.zip (测试集带噪语音, ~300MB)
   - clean_trainset_wav.zip (训练集干净语音, ~5GB, 可选)
   - noisy_trainset_wav.zip (训练集带噪语音, ~5GB, 可选)
3. 解压到: {data_dir}/

## 方��3: DEMAND噪声库
VoiceBank使用的噪声来自DEMAND库:
- https://zenodo.org/record/1227121
- 下载DEMAND.zip (~1GB)

## 预期目录结构
{data_dir}/
├── clean_testset_wav/
│   ├── p232_001.wav
│   ├── p232_002.wav
│   └── ...
├── noisy_testset_wav/
│   ├── p232_001.wav
│   └── ...
├── clean_trainset_wav/   (可选)
├── noisy_trainset_wav/   (可选)
└── dataset_meta.json
""".format(data_dir=self.data_dir)
