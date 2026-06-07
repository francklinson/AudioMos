"""
DNS Challenge 标准测试数据集

DNS (Deep Noise Suppression) Challenge 是由Microsoft组织的
语音降噪/增强领域最权威的评测基准。
支持 DNS1 - DNS5 版本。

参考: https://github.com/microsoft/DNS-Challenge
"""

import os
import json
import time
import logging
import urllib.request
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

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


class DNSChallengeDataset(BaseDataset):
    """
    DNS Challenge 数据集

    数据集组成:
    - 干净语音: 多语言朗读语音 (~60GB)
    - 噪声: 各种环境噪声 (~15GB)
    - 房间冲激响应(RIR): 用于生成混响 (~500MB)

    使用方式:
        dataset = DNSChallengeDataset(data_dir="./data/datasets/dns_challenge")
        if not dataset.is_downloaded:
            print(dataset._get_download_instructions())
            dataset.download()
        pairs = dataset.prepare_evaluation_set(EvaluationSetConfig(n_samples=50))
    """

    DEFAULT_SAMPLE_RATE = 16000

    def __init__(self, data_dir: str, sample_rate: int = 16000):
        super().__init__(data_dir, sample_rate)

    def get_meta(self) -> DatasetMeta:
        return DatasetMeta(
            name="DNS Challenge",
            version="DNS5 (2023)",
            organization="Microsoft Research",
            description="业界标准的深度降噪评估基准，包含多语言干净语音、150+类噪声和房间冲激响应",
            url="https://github.com/microsoft/DNS-Challenge",
            license_info="需注册Microsoft DNS Challenge获取Azure Blob访问权限",
            citation="Reddy et al., 'The INTERSPEECH 2020 Deep Noise Suppression Challenge', INTERSPEECH 2020",
            total_duration_hours=550,
            total_files=75000,
            sample_rate=16000,
            categories=[DatasetCategory.CLEAN_SPEECH, DatasetCategory.NOISE, DatasetCategory.RIR],
            known_baselines={
                "dns_challenge_baseline": {"pesq": 2.15, "stoi": 0.91, "sisdr": 15.0},
                "dccrn": {"pesq": 2.84, "stoi": 0.92, "dnsmos_ovrl": 3.05},
            },
        )

    def download(self, force: bool = False) -> bool:
        """
        下载 DNS Challenge 数据集

        由于数据集托管在Azure Blob Storage且需要注册访问，
        此方法提供详细的下载指引并尝试验证已有的数据。
        """
        if self.is_downloaded and not force:
            logger.info("DNS Challenge数据集已存在")
            return True

        instructions = self._get_download_instructions()
        logger.info(instructions)
        print(instructions)

        # 尝试创建基础目录结构
        for subdir in ["clean_speech", "noise", "rir", "synthesized"]:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)

        # 保存元数据（下载后手动更新）
        meta = self.get_meta()
        self.save_metadata(meta)

        return self.is_downloaded

    def validate(self) -> Dict[str, bool]:
        results = {
            "directory_exists": os.path.exists(self.data_dir),
            "has_clean_speech": False,
            "has_noise": False,
            "has_rir": False,
            "file_count_ok": False,
            "sample_rate_ok": True,
        }

        if not results["directory_exists"]:
            return results

        # 检查各子目录
        for category in ["clean_speech", "noise", "rir"]:
            cat_dir = os.path.join(self.data_dir, category)
            if os.path.exists(cat_dir):
                files = self.scan_audio_files(cat_dir)
                results[f"has_{category}"] = len(files) > 0

                # 检查采样率
                for f in files[:10]:  # 抽样检查
                    try:
                        info = sf.info(f)
                        if info.samplerate != self.DEFAULT_SAMPLE_RATE:
                            results["sample_rate_ok"] = False
                            break
                    except Exception:
                        pass

        # 粗略的文件数检查
        total_files = sum(
            len(self.scan_audio_files(os.path.join(self.data_dir, c)))
            for c in ["clean_speech", "noise", "rir"]
            if os.path.exists(os.path.join(self.data_dir, c))
        )
        results["file_count_ok"] = total_files > 100  # DNS至少应该有数百个文件

        return results

    def get_statistics(self) -> DatasetStatistics:
        stats = DatasetStatistics(
            name="DNS Challenge",
            version="DNS5",
            data_dir=self.data_dir,
            exists=os.path.exists(self.data_dir),
        )

        if not stats.exists:
            return stats

        total_files = 0
        total_duration = 0.0

        for category in ["clean_speech", "noise", "rir"]:
            cat_dir = os.path.join(self.data_dir, category)
            if not os.path.exists(cat_dir):
                continue

            files = self.scan_audio_files(cat_dir)
            cat_duration = 0.0

            for f in files[:500]:  # 抽样估算时长
                dur = self.compute_audio_duration(f)
                if dur:
                    cat_duration += dur

            # 如果有更多文件，按比例估算
            if len(files) > 500:
                scale = len(files) / 500
                cat_duration *= scale

            stats.categories[category] = {
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
        if config is None:
            config = EvaluationSetConfig()

        # 检查数据
        if not self.is_downloaded:
            logger.warning("DNS Challenge 数据集未下载")
            print(self._get_download_instructions())

            # 回退到内置数据
            logger.info("回退到内置合成数据")
            return _BuiltinGenerator(self.data_dir, self.sample_rate).generate(config)

        # 从已有数据合成评测集
        return self._synthesize_pairs(config)

    def _synthesize_pairs(self, config: EvaluationSetConfig) -> List[SamplePair]:
        """从DNS数据集合成带噪-干净音频对"""
        import random
        random.seed(config.seed)
        np.random.seed(config.seed)

        output_dir = os.path.join(self.data_dir, "evaluation_set")
        os.makedirs(output_dir, exist_ok=True)

        # 扫描可用数据
        clean_files = self.scan_audio_files(os.path.join(self.data_dir, "clean_speech"))
        noise_files = self.scan_audio_files(os.path.join(self.data_dir, "noise"))
        rir_files = self.scan_audio_files(os.path.join(self.data_dir, "rir"))

        if not clean_files or not noise_files:
            logger.warning("DNS数据不完整，回退到内置合成")
            return _BuiltinGenerator(self.data_dir, self.sample_rate).generate(config)

        pairs = []
        samples_per_snr = max(1, config.n_samples // len(config.snr_levels))

        for snr_db in config.snr_levels:
            snr_dir = os.path.join(output_dir, f"snr_{int(snr_db)}dB")
            os.makedirs(snr_dir, exist_ok=True)

            for i in range(samples_per_snr):
                try:
                    # 随机选择文件
                    clean_file = random.choice(clean_files)
                    noise_file = random.choice(noise_files)

                    # 加载音频
                    clean_audio, sr = sf.read(clean_file)
                    if sr != config.target_sample_rate:
                        clean_audio = librosa.resample(
                            clean_audio.astype(np.float64),
                            orig_sr=sr,
                            target_sr=config.target_sample_rate,
                        )

                    noise_audio, n_sr = sf.read(noise_file)
                    if n_sr != config.target_sample_rate:
                        noise_audio = librosa.resample(
                            noise_audio.astype(np.float64),
                            orig_sr=n_sr,
                            target_sr=config.target_sample_rate,
                        )

                    # 单声道处理
                    if clean_audio.ndim > 1:
                        clean_audio = clean_audio[:, 0]
                    if noise_audio.ndim > 1:
                        noise_audio = noise_audio[:, 0]

                    # 长度对齐
                    target_len = int(
                        config.target_sample_rate
                        * random.uniform(config.min_duration, config.max_duration)
                    )
                    if len(clean_audio) > target_len:
                        start = random.randint(0, len(clean_audio) - target_len)
                        clean_audio = clean_audio[start: start + target_len]
                    else:
                        clean_audio = np.pad(clean_audio, (0, max(0, target_len - len(clean_audio))))

                    if len(noise_audio) < len(clean_audio):
                        repeats = len(clean_audio) // len(noise_audio) + 1
                        noise_audio = np.tile(noise_audio, repeats)
                    noise_audio = noise_audio[: len(clean_audio)]

                    # 可选混响
                    if config.use_reverb and rir_files and random.random() < config.reverb_probability:
                        rir_file = random.choice(rir_files)
                        rir, rir_sr = sf.read(rir_file)
                        if rir.ndim > 1:
                            rir = rir[:, 0]
                        if rir_sr != config.target_sample_rate:
                            rir = librosa.resample(
                                rir.astype(np.float64),
                                orig_sr=rir_sr,
                                target_sr=config.target_sample_rate,
                            )
                        clean_audio = np.convolve(clean_audio, rir, mode="full")[: len(clean_audio)]

                    # 调整SNR
                    clean_power = np.mean(clean_audio ** 2)
                    noise_power = np.mean(noise_audio ** 2)
                    desired_noise_power = clean_power / (10 ** (snr_db / 10) + 1e-10)
                    noise_audio = noise_audio * np.sqrt(desired_noise_power / (noise_power + 1e-10))

                    noisy_audio = clean_audio + noise_audio

                    # 防削波
                    max_val = np.max(np.abs(noisy_audio))
                    if max_val > 0.99:
                        scale = 0.99 / max_val
                        noisy_audio *= scale
                        clean_audio *= scale

                    # 保存
                    idx = len(pairs)
                    clean_path = os.path.join(snr_dir, f"clean_{idx:04d}.wav")
                    noisy_path = os.path.join(snr_dir, f"noisy_{idx:04d}.wav")

                    sf.write(clean_path, clean_audio.astype(np.float32), config.target_sample_rate)
                    sf.write(noisy_path, noisy_audio.astype(np.float32), config.target_sample_rate)

                    # 推断噪声类型（基于文件名关键词）
                    noise_type = self._infer_noise_type(noise_file)

                    pairs.append(SamplePair(
                        noisy_path=noisy_path,
                        clean_path=clean_path,
                        metadata=SampleMetadata(
                            noise_type=noise_type,
                            snr_db=snr_db,
                            original_source="dns_challenge",
                        ),
                    ))

                except Exception as e:
                    logger.warning(f"合成样本失败 [{snr_db}dB #{i}]: {e}")
                    continue

        # 保存元数据
        self._save_pair_list(output_dir, pairs)

        return pairs

    def _infer_noise_type(self, filepath: str) -> NoiseType:
        """根据文件名推断噪声类型"""
        name_lower = filepath.lower()
        keywords = {
            NoiseType.BABBLE: ["babble", "speech", "crowd"],
            NoiseType.TRAFFIC: ["traffic", "car", "bus", "street", "vehicle"],
            NoiseType.CAFE: ["cafe", "cafeteria", "restaurant", "coffee"],
            NoiseType.FACTORY: ["factory", "machinery", "industrial", "engine"],
            NoiseType.MUSIC: ["music", "song", "instrument"],
            NoiseType.NATURAL: ["wind", "rain", "thunder", "water", "nature"],
            NoiseType.STATIONARY: ["white", "pink", "static", "stationary"],
        }

        for noise_type, kws in keywords.items():
            if any(kw in name_lower for kw in kws):
                return noise_type

        return NoiseType.NON_STATIONARY

    def _save_pair_list(self, output_dir: str, pairs: List[SamplePair]):
        """保存评测对列表"""
        pair_list = []
        for i, p in enumerate(pairs):
            pair_list.append({
                "index": i,
                "noisy": p.noisy_path,
                "clean": p.clean_path,
                "snr_db": p.metadata.snr_db,
                "noise_type": p.metadata.noise_type.value,
            })

        with open(os.path.join(output_dir, "pair_list.json"), "w") as f:
            json.dump(pair_list, f, indent=2, ensure_ascii=False)

    def _get_download_instructions(self) -> str:
        """获取下载指引文本（供子类DatasetManager调用）"""
        return """
# DNS Challenge 数据集下载指引

## 方法1: 官方下载（推荐）
1. 访问 https://github.com/microsoft/DNS-Challenge
2. 注册并获取Azure Blob Storage访问权限
3. 使用azcopy或wget下载数据集

## 方法2: 使用项目内置合成数据
如果无法获取完整DNS数据集，可使用项目内置的 BuiltinDataset:
  from datasets.dns import BuiltinDataset
  dataset = BuiltinDataset(data_dir="./data/datasets/builtin")
  pairs = dataset.prepare_evaluation_set(EvaluationSetConfig(n_samples=50))

## 预期目录结构
{data_dir}/
├── clean_speech/
│   ├── read_speech/
│   ├── singing_voice/
│   └── ...
├── noise/
│   ├── stationary/
│   ├── non_stationary/
│   └── ...
├── rir/
│   └── room_impulse_responses/
└── dataset_meta.json
""".format(data_dir=self.data_dir)


# ===========================
# 内置数据集生成器
# ===========================


class BuiltinDataset(BaseDataset):
    """
    项目内置测试数据集

    使用项目已有的测试音频或合成简单测试数据。
    适用于快速功能验证。
    """

    def __init__(self, data_dir: str = None, sample_rate: int = 16000):
        """初始化内置数据集"""
        if data_dir is None:
            # 使用项目根目录下的data目录
            import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ))))
        self._project_root = project_root
        if data_dir is None:
            data_dir = os.path.join(project_root, "data", "datasets", "builtin")
        super().__init__(data_dir, sample_rate)

    def get_meta(self) -> DatasetMeta:
        return DatasetMeta(
            name="AudioMos内置测试集",
            version="1.0",
            organization="AudioMos项目",
            description="使用项目已有测试音频和合成数据，支持快速功能验证",
            url=None,
            license_info="项目内部使用",
            citation="",
            total_files=50,
            sample_rate=self.sample_rate,
            categories=[DatasetCategory.NOISY],
        )

    def download(self, force: bool = False) -> bool:
        """内置数据集不需要下载"""
        os.makedirs(self.data_dir, exist_ok=True)
        self.save_metadata(self.get_meta())
        return True

    def validate(self) -> Dict[str, bool]:
        return {
            "directory_exists": True,
            "always_valid": True,
        }

    def get_statistics(self) -> DatasetStatistics:
        gen = _BuiltinGenerator(self.data_dir, self.sample_rate)
        return DatasetStatistics(
            name="AudioMos内置测试集",
            version="1.0",
            data_dir=self.data_dir,
            exists=True,
            total_files=50,
            total_duration_hours=0.05,
            categories={"synthetic": {"file_count": 50, "total_duration_hours": 0.05}},
        )

    def prepare_evaluation_set(
        self, config: Optional[EvaluationSetConfig] = None
    ) -> List[SamplePair]:
        if config is None:
            config = EvaluationSetConfig()

        # 首先尝试使用项目已有的测试数据
        pairs = self._collect_project_test_data()
        if pairs and len(pairs) >= config.n_samples:
            return pairs[: config.n_samples]

        # 不足则用合成数据补充
        logger.info("项目测试数据不足，使用合成数据补充")
        gen = _BuiltinGenerator(self.data_dir, self.sample_rate)
        return gen.generate(config)

    def _collect_project_test_data(self) -> List[SamplePair]:
        """收集项目中已有的测试音频"""
        pairs = []

        # 扫描测试目录
        test_dirs = [
            os.path.join(self._project_root, "test_data", "enhanced"),
            os.path.join(self._project_root, "test_data", "reference"),
            os.path.join(self._project_root, "data", "ref"),
        ]

        noisy_files = []
        clean_files = []

        for test_dir in test_dirs:
            if not os.path.exists(test_dir):
                continue
            for root, _, filenames in os.walk(test_dir):
                for f in filenames:
                    if f.lower().endswith(".wav"):
                        full = os.path.join(root, f)
                        if "ref" in root.lower() or "reference" in root.lower() or "clean" in f.lower():
                            clean_files.append(full)
                        else:
                            noisy_files.append(full)

        # 配对
        for i, noisy in enumerate(noisy_files):
            clean = clean_files[i] if i < len(clean_files) else None
            pairs.append(SamplePair(
                noisy_path=noisy,
                clean_path=clean,
                metadata=SampleMetadata(
                    noise_type=NoiseType.UNKNOWN,
                    scene_type=SceneType.UNKNOWN_SCENE,
                    original_source="project_test_data",
                ),
            ))

        return pairs

    def _get_download_instructions(self) -> str:
        return """内置数据集无需下载，自动使用项目测试数据。"""


# ===========================
# 内置合成器（共享组�）
# ===========================


class _BuiltinGenerator:
    """内置合成测试数据生成器"""

    # 预定义的合成语音模板
    _VOICE_TEMPLATES = {
        "male_speech": {
            "freqs": [120, 240, 360, 500, 700, 1000, 1400, 2000, 2800, 3500],
            "amps": [0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.02],
        },
        "female_speech": {
            "freqs": [200, 400, 600, 800, 1100, 1500, 2000, 2800, 3500, 4200],
            "amps": [0.25, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.02],
        },
        "child_speech": {
            "freqs": [250, 500, 750, 1000, 1500, 2000, 2800, 3500, 4200, 5000],
            "amps": [0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01],
        },
    }

    # 噪声生成器
    _NOISE_GENERATORS = {
        NoiseType.STATIONARY: lambda t: np.random.randn(len(t)),
        NoiseType.BABBLE: lambda t: 0.3 * (
            np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t)
        ) * np.random.randn(len(t)),
        NoiseType.TRAFFIC: lambda t: 0.5 * np.sin(2 * np.pi * 50 * t) * (
            1 + 0.5 * np.sin(2 * np.pi * 5 * t)
        ) + 0.1 * np.random.randn(len(t)),
        NoiseType.CAFE: lambda t: 0.2 * np.sin(2 * np.pi * 300 * t) * np.random.randn(len(t))
        + 0.15 * np.random.randn(len(t)),
        NoiseType.FACTORY: lambda t: 0.3 * np.sin(2 * np.pi * 60 * t)
        + 0.2 * np.sin(2 * np.pi * 120 * t)
        + 0.1 * np.sin(2 * np.pi * 240 * t)
        + 0.2 * np.random.randn(len(t)) * np.abs(np.sin(2 * np.pi * 0.5 * t)),
        NoiseType.NATURAL: lambda t: 0.3 * np.random.randn(len(t)) * (
            1 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
        ),
    }

    def __init__(self, output_dir: str, sample_rate: int = 16000):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, config: EvaluationSetConfig) -> List[SamplePair]:
        """生成合成测试数据"""
        import random
        random.seed(config.seed)
        np.random.seed(config.seed)

        pairs = []
        output_dir = os.path.join(self.output_dir, "synthetic_test")
        os.makedirs(output_dir, exist_ok=True)

        noise_types = config.noise_types or list(NoiseType)
        if NoiseType.UNKNOWN in noise_types:
            noise_types = [n for n in noise_types if n != NoiseType.UNKNOWN]

        samples_per_config = max(1, config.n_samples // max(1, len(config.snr_levels) * len(noise_types)))

        for snr_db in config.snr_levels:
            for noise_type in noise_types:
                for i in range(samples_per_config):
                    try:
                        # 生成语音
                        voice_template = random.choice(list(self._VOICE_TEMPLATES.values()))
                        duration = random.uniform(config.min_duration, min(config.max_duration, 10.0))
                        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

                        clean = np.zeros_like(t)
                        for freq, amp in zip(voice_template["freqs"], voice_template["amps"]):
                            # 添加频率微动（模拟语音的自然变化）
                            freq_jitter = freq * (1 + 0.01 * np.sin(2 * np.pi * 4 * t))
                            phase = random.uniform(0, 2 * np.pi)
                            clean += amp * np.sin(2 * np.pi * freq_jitter * t + phase)

                        # 添加颤音（vibrato）
                        vibrato_rate = random.uniform(4, 8)
                        clean *= 1 + 0.05 * np.sin(2 * np.pi * vibrato_rate * t)

                        # 归一化
                        clean /= np.max(np.abs(clean)) + 1e-10
                        clean *= 0.8

                        # 生成噪声
                        if noise_type in self._NOISE_GENERATORS:
                            noise = self._NOISE_GENERATORS[noise_type](t)
                        else:
                            noise = np.random.randn(len(t))

                        noise /= np.max(np.abs(noise)) + 1e-10
                        noise *= 0.8

                        # 调整SNR
                        clean_power = np.mean(clean ** 2)
                        noise_power = np.mean(noise ** 2)
                        desired_noise_power = clean_power / (10 ** (snr_db / 10) + 1e-10)
                        noise = noise * np.sqrt(desired_noise_power / (noise_power + 1e-10))

                        noisy = clean + noise

                        # 防削波
                        max_val = np.max(np.abs(noisy))
                        if max_val > 0.99:
                            noisy /= max_val / 0.99
                            clean /= max_val / 0.99

                        # 保存
                        idx = len(pairs)
                        clean_path = os.path.join(output_dir, f"clean_snr{int(snr_db)}_{noise_type.value}_{idx:04d}.wav")
                        noisy_path = os.path.join(output_dir, f"noisy_snr{int(snr_db)}_{noise_type.value}_{idx:04d}.wav")

                        sf.write(clean_path, clean.astype(np.float32), self.sample_rate)
                        sf.write(noisy_path, noisy.astype(np.float32), self.sample_rate)

                        pairs.append(SamplePair(
                            noisy_path=noisy_path,
                            clean_path=clean_path,
                            metadata=SampleMetadata(
                                noise_type=noise_type,
                                snr_db=snr_db,
                                scene_type=SceneType.UNKNOWN_SCENE,
                                original_source="builtin_synthetic",
                            ),
                        ))

                    except Exception as e:
                        logger.warning(f"生成合成样本失败 [{noise_type} {snr_db}dB #{i}]: {e}")
                        continue

        return pairs[: config.n_samples]
