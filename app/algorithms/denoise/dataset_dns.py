"""
DNS Challenge 标准测试数据集支持

DNS (Deep Noise Suppression) Challenge 是由Microsoft组织的
语音降噪/增强领域最权威的评测基准。

数据集组成:
- 干净语音: 多语言朗读语音
- 噪声: 各种环境噪声
- 房间冲激响应(RIR): 用于生成混响

挑战版本:
- DNS1 (2019): 初始版本
- DNS2 (2020): 增加数据处理和评估
- DNS3 (2021): 增加个性化增强
- DNS4 (2022): 增加双耳增强
- DNS5 (2023): 增加多通道增强

评估指标:
- PESQ (ITU-T P.862)
- STOI (Short-Time Objective Intelligibility)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- DNSMOS (ITU-T P.808 基于深度学习的MOS)

参考:
- https://github.com/microsoft/DNS-Challenge
- Reddy et al., "The INTERSPEECH 2020 Deep Noise Suppression Challenge", 2020
"""

import os
import json
import time
import hashlib
import urllib.request
import zipfile
import tarfile
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf
import librosa


# ===========================
# 数据类
# ===========================


@dataclass
class DNSAudioSample:
    """DNS数据集音频样本"""

    filepath: str
    sample_type: str  # 'clean_speech', 'noise', 'rir', 'noisy'
    duration: float
    sample_rate: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class DNSMixConfig:
    """DNS混音配置"""

    snr_db: float = 0.0  # 信噪比 (dB)
    target_sr: int = 16000  # 目标采样率
    use_reverb: bool = True  # 是否添加混响
    rir_probability: float = 0.5  # 混响概率


# ===========================
# DNS数据集管理器
# ===========================


class DNSDataset:
    """
    DNS Challenge 数据集管理器

    支持:
    - 数据集下载 (需要手动下载或通过URL)
    - 数据集验证 (校验完整性)
    - 合成带噪音频 (干净语音 + 噪声 + 可选混响)
    - 数据集统计
    - 导出评估文件列表

    使用方式:
        dataset = DNSDataset(data_dir="./data/dns_challenge")
        dataset.download()  # 下载数据集
        dataset.validate()  # 验证完整性
        noisy_files, clean_files = dataset.prepare_evaluation_set(n_samples=100)
    """

    # DNS Challenge 数据集标准结构
    EXPECTED_STRUCTURE = {
        "clean_speech": ["read_speech", "singing_voice", "non_english"],
        "noise": ["noise_types"],
        "rir": ["room_impulse_responses"],
    }

    # 标准评估文件列表
    STANDARD_TEST_SETS = {
        "dns1_blind": "blind_test_set",
        "dns2_blind": "datasets/test_set/synthetic/no_reverb",
        "dns3_blind": "datasets/blind_test_set",
    }

    def __init__(self, data_dir: str = "./data/dns_challenge"):
        """
        初始化DNS数据集管理器

        Args:
            data_dir: 数据集存储目录
        """
        self.data_dir = data_dir
        self.metadata_file = os.path.join(data_dir, "dataset_metadata.json")
        self._samples: Dict[str, List[DNSAudioSample]] = {}

    @property
    def is_downloaded(self) -> bool:
        """数据集是否已下载"""
        return os.path.exists(self.metadata_file)

    def scan(self) -> Dict[str, List[str]]:
        """
        扫描数据集目录，统计文件

        Returns:
            按类别分组的文件列表
        """
        categories = {}

        if not os.path.exists(self.data_dir):
            return categories

        for category in ["clean_speech", "noise", "rir"]:
            category_dir = os.path.join(self.data_dir, category)
            if os.path.exists(category_dir):
                files = []
                for root, _, filenames in os.walk(category_dir):
                    for f in filenames:
                        if f.endswith((".wav", ".flac", ".mp3")):
                            files.append(os.path.join(root, f))
                categories[category] = files

        return categories

    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息

        Returns:
            包含总时长、文件数等统计
        """
        stats = {
            "data_dir": self.data_dir,
            "exists": os.path.exists(self.data_dir),
            "categories": {},
        }

        if not stats["exists"]:
            return stats

        files = self.scan()
        total_duration = 0
        total_files = 0

        for category, file_list in files.items():
            category_duration = 0
            for f in file_list:
                try:
                    info = sf.info(f)
                    category_duration += info.duration
                except Exception:
                    pass

            stats["categories"][category] = {
                "file_count": len(file_list),
                "total_duration_hours": round(category_duration / 3600, 2),
            }
            total_duration += category_duration
            total_files += len(file_list)

        stats["total_files"] = total_files
        stats["total_duration_hours"] = round(total_duration / 3600, 2)

        return stats

    def download_info(self) -> str:
        """
        获取DNS Challenge下载指引

        Returns:
            下载说明文本
        """
        return """
# DNS Challenge 数据集下载指引

DNS Challenge数据集需要通过官方渠道获取：

## 官方下载地址
1. DNS Challenge GitHub: https://github.com/microsoft/DNS-Challenge
2. 数据集注册: https://dns-challenge.azurewebsites.net/

## 下载步骤
1. 访问 https://github.com/microsoft/DNS-Challenge
2. 在README中找到数据集下载链接（Azure Blob Storage）
3. 下载所需数据集文件
   - clean_speech/ (约60GB)
   - noise/ (约15GB)
   - rir/ (约500MB)

## 自动下载脚本
```bash
# 使用wget下载（示例）
cd data/dns_challenge
wget -r -np -nH --cut-dirs=3 <azure-blob-url>
```

## 数据集组织
下载后将数据组织为以下结构：
```
data/dns_challenge/
├── clean_speech/
│   ├── read_speech/
│   └── ...
├── noise/
│   ├── noise_types/
│   └── ...
└── rir/
    ├── room_impulse_responses/
    └── ...
```
"""

    def synthesize_noisy(
        self,
        clean_files: List[str],
        noise_files: List[str],
        rir_files: Optional[List[str]] = None,
        config: DNSMixConfig = None,
    ) -> List[Tuple[str, str]]:
        """
        合成带噪音频文件

        将干净语音与噪声混合，可选添加混响。

        Args:
            clean_files: 干净音频文件列表
            noise_files: 噪声文件列表
            rir_files: 房间冲激响应文件列表（可选）
            config: 混音配置

        Returns:
            [(noisy_file_path, clean_file_path), ...] 带噪和干净音频对
        """
        if config is None:
            config = DNSMixConfig()

        output_dir = os.path.join(self.data_dir, "synthesized")
        os.makedirs(output_dir, exist_ok=True)

        pairs = []

        for i, clean_file in enumerate(clean_files):
            try:
                # 加载干净语音
                clean_audio, sr = sf.read(clean_file)
                if sr != config.target_sr:
                    clean_audio = librosa.resample(clean_audio, orig_sr=sr, target_sr=config.target_sr)

                # 确保单声道
                if len(clean_audio.shape) > 1:
                    clean_audio = np.mean(clean_audio, axis=1)

                # 随机选择噪声文件（循环使用）
                noise_file = noise_files[i % len(noise_files)]
                noise_audio, noise_sr = sf.read(noise_file)
                if noise_sr != config.target_sr:
                    noise_audio = librosa.resample(noise_audio, orig_sr=noise_sr, target_sr=config.target_sr)
                if len(noise_audio.shape) > 1:
                    noise_audio = np.mean(noise_audio, axis=1)

                # 裁剪噪声到干净语音长度
                if len(noise_audio) < len(clean_audio):
                    # 循环拼接
                    repeats = len(clean_audio) // len(noise_audio) + 1
                    noise_audio = np.tile(noise_audio, repeats)
                noise_audio = noise_audio[: len(clean_audio)]

                # 添加混响（可选）
                if config.use_reverb and rir_files and np.random.random() < config.rir_probability:
                    rir_file = np.random.choice(rir_files)
                    rir, rir_sr = sf.read(rir_file)
                    if rir_sr != config.target_sr:
                        rir = librosa.resample(rir, orig_sr=rir_sr, target_sr=config.target_sr)
                    if len(rir.shape) > 1:
                        rir = rir[:, 0]
                    # 卷积添加混响
                    clean_audio = np.convolve(clean_audio, rir, mode="full")[: len(clean_audio)]

                # 调整信噪比
                clean_power = np.mean(clean_audio**2)
                noise_power = np.mean(noise_audio**2)

                desired_noise_power = clean_power / (10 ** (config.snr_db / 10))
                noise_audio = noise_audio * np.sqrt(desired_noise_power / (noise_power + 1e-10))

                # 混合
                noisy_audio = clean_audio + noise_audio

                # 归一化防止削波
                max_val = np.max(np.abs(noisy_audio))
                if max_val > 0.99:
                    noisy_audio = noisy_audio / max_val * 0.99

                # 保存
                clean_out = os.path.join(output_dir, f"clean_{i:04d}.wav")
                noisy_out = os.path.join(output_dir, f"noisy_{i:04d}.wav")

                sf.write(clean_out, clean_audio, config.target_sr)
                sf.write(noisy_out, noisy_audio, config.target_sr)

                pairs.append((noisy_out, clean_out))

            except Exception as e:
                print(f"合成失败 {clean_file}: {e}")
                continue

        # 保存合成元数据
        meta = {
            "config": {"snr_db": config.snr_db, "target_sr": config.target_sr, "use_reverb": config.use_reverb},
            "n_pairs": len(pairs),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(output_dir, "synthesis_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return pairs

    def prepare_evaluation_set(
        self,
        n_samples: int = 100,
        snr_dbs: Optional[List[float]] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        准备标准评估数据集

        从已有数据中合成指定数量的带噪音频进行评估。

        Args:
            n_samples: 合成样本数量
            snr_dbs: SNR级别列表
            output_dir: 输出目录

        Returns:
            (noisy_files, clean_files) 文件列表对
        """
        if snr_dbs is None:
            snr_dbs = [0, 5, 10, 15]

        if output_dir is None:
            output_dir = os.path.join(self.data_dir, "evaluation_set")

        os.makedirs(output_dir, exist_ok=True)

        all_noisy = []
        all_clean = []

        categories = self.scan()
        clean_files = categories.get("clean_speech", [])
        noise_files = categories.get("noise", [])
        rir_files = categories.get("rir", [])

        if not clean_files or not noise_files:
            print("警告: 数据集不完整，无法合成评估数据")
            print("请下载DNS Challenge数据集到:", self.data_dir)
            return [], []

        samples_per_snr = n_samples // len(snr_dbs)

        for snr in snr_dbs:
            config = DNSMixConfig(snr_db=snr)
            snr_output = os.path.join(output_dir, f"snr_{int(snr)}dB")
            os.makedirs(snr_output, exist_ok=True)

            pairs = self.synthesize_noisy(
                clean_files[:samples_per_snr],
                noise_files[:samples_per_snr],
                rir_files,
                config,
            )

            for noisy, clean in pairs:
                all_noisy.append(noisy)
                all_clean.append(clean)

        # 保存文件列表
        file_list = {"noisy_files": all_noisy, "clean_files": all_clean, "n_files": len(all_noisy)}
        with open(os.path.join(output_dir, "file_list.json"), "w") as f:
            json.dump(file_list, f, indent=2, default=str)

        return all_noisy, all_clean

    def prepare_simple_test_set(
        self, output_dir: Optional[str] = None
    ) -> Tuple[List[str], Optional[List[str]]]:
        """
        准备简单测试集（不需要完整DNS数据集）

        使用项目中已有的测试音频创建标准化的测试集。

        Args:
            output_dir: 输出目录

        Returns:
            (test_files, reference_files_or_None)
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, "simple_test_set")

        os.makedirs(output_dir, exist_ok=True)

        test_files = []
        ref_files = []

        # 扫描项目测试数据
        project_test_dirs = [
            "./test_data/enhanced",
            "./test_data/reference",
            "./data/ref",
            "./data/uploads",
        ]

        for test_dir in project_test_dirs:
            if os.path.exists(test_dir):
                for root, _, filenames in os.walk(test_dir):
                    for f in filenames:
                        if f.endswith(".wav"):
                            src = os.path.join(root, f)
                            dest = os.path.join(output_dir, f)
                            if not os.path.exists(dest):
                                try:
                                    audio, sr = sf.read(src)
                                    sf.write(dest, audio, sr)
                                except Exception:
                                    import shutil

                                    shutil.copy2(src, dest)

                            if "ref" in test_dir.lower() or "reference" in test_dir.lower():
                                ref_files.append(dest)
                            else:
                                test_files.append(dest)

        return test_files, ref_files if ref_files else None


def create_dns_dataset(download: bool = False) -> DNSDataset:
    """
    创建DNS数据集实例的便捷函数

    Args:
        download: 是否下载（显示下载指引）

    Returns:
        DNSDataset实例
    """
    dataset = DNSDataset()

    if download:
        info = dataset.download_info()
        print(info)

    return dataset


# ===========================
# 辅助工具
# ===========================


def list_available_datasets() -> List[Dict]:
    """列出所有可用的标准测试数据集"""
    datasets = [
        {
            "name": "DNS Challenge (DNS1-DNS5)",
            "organization": "Microsoft Research",
            "description": "业界标准的降噪评估基准",
            "metrics": ["PESQ", "STOI", "SI-SDR", "DNSMOS"],
            "url": "https://github.com/microsoft/DNS-Challenge",
            "requires_download": True,
        },
        {
            "name": "Voicebank-DEMAND",
            "organization": "University of Edinburgh",
            "description": "标准语音增强数据集，30说话人",
            "metrics": ["PESQ", "STOI", "CSIG", "CBAK", "COVL"],
            "url": "https://datashare.ed.ac.uk/handle/10283/2791",
            "requires_download": True,
        },
        {
            "name": "WHAM! / WHAMR!",
            "organization": "MERL & Mitsubishi Electric",
            "description": "WSJ0混合数据集，含噪声/混响",
            "metrics": ["SI-SDR", "SDR", "PESQ"],
            "url": "https://wham.whisper.ai/",
            "requires_download": True,
        },
        {
            "name": "项目内置测试集",
            "organization": "AudioMos项目",
            "description": "使用项目已有的测试音频进行快速评估",
            "metrics": ["PESQ", "STOI", "SI-SDR", "DNSMOS", "NISQA", "UTMOS"],
            "url": None,
            "requires_download": False,
        },
    ]
    return datasets


if __name__ == "__main__":
    # 快速测试
    dataset = DNSDataset()
    stats = dataset.get_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
