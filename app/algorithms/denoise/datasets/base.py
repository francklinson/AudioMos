"""
降噪测评数据集抽象基类

定义数据集统一接口，确保所有数据集实现一致的操作方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import hashlib
import os


# ===========================
# 枚举定义
# ===========================


class NoiseType(str, Enum):
    """噪声类型枚举"""
    STATIONARY = "stationary"           # 平稳噪声 (白噪声、粉红噪声)
    NON_STATIONARY = "non_stationary"   # 非平稳噪声
    BABBLE = "babble"                   # 多人说话
    TRAFFIC = "traffic"                 # 交通噪声
    CAFE = "cafe"                       # 餐厅噪声
    FACTORY = "factory"                 # 工业噪声
    MUSIC = "music"                     # 音乐噪声
    NATURAL = "natural"                 # 自然环境噪声 (风、雨)
    UNKNOWN = "unknown"                 # 未知


class SceneType(str, Enum):
    """场景类型枚举"""
    INDOOR_MEETING = "indoor_meeting"   # 室内会议
    IN_CAR = "in_car"                   # 车载
    STREET = "street"                   # 街道
    RESTAURANT = "restaurant"           # 餐厅
    FACTORY_SCENE = "factory"           # 工厂
    OUTDOOR = "outdoor"                 # 户外
    TELEPHONE = "telephone"             # 电话信道
    REVERB = "reverb"                   # 强混响
    UNKNOWN_SCENE = "unknown"           # 未知


class DatasetCategory(str, Enum):
    """数据集类别"""
    CLEAN_SPEECH = "clean_speech"
    NOISE = "noise"
    RIR = "rir"                         # 房间冲激响应
    NOISY = "noisy"
    MIXED = "mixed"


SNR_LEVELS = [-5, 0, 5, 10, 15, 20, 25]


# ===========================
# 数据类
# ===========================


@dataclass
class DatasetMeta:
    """数据集元数据"""
    name: str                           # 数据集名称
    version: str = "1.0"                # 版本
    organization: str = ""              # 发布组织
    description: str = ""               # 描述
    url: Optional[str] = None           # 下载URL
    license_info: str = ""              # 许可证信息
    citation: str = ""                  # 引用论文
    total_duration_hours: float = 0.0   # 总时长(小时)
    total_files: int = 0                # 总文件数
    sample_rate: int = 16000            # 采样率
    categories: List[DatasetCategory] = field(default_factory=list)
    known_baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 已知基线: {algorithm_name: {metric_name: value}}


@dataclass
class SampleMetadata:
    """音频样本元数据"""
    noise_type: NoiseType = NoiseType.UNKNOWN
    scene_type: SceneType = SceneType.UNKNOWN_SCENE
    snr_db: Optional[float] = None
    speaker_id: Optional[str] = None
    transcription: Optional[str] = None
    original_source: Optional[str] = None
    extras: Dict = field(default_factory=dict)


@dataclass
class SamplePair:
    """评测用的音频样本对"""
    noisy_path: str                     # 带噪音频路径
    clean_path: Optional[str] = None    # 干净参考音频路径
    metadata: SampleMetadata = field(default_factory=SampleMetadata)
    sample_rate: int = 16000


@dataclass
class EvaluationSetConfig:
    """评测集配置"""
    n_samples: int = 100                # 需要的样本数
    snr_levels: List[float] = field(default_factory=lambda: [0, 5, 10, 15])
    noise_types: Optional[List[NoiseType]] = None
    scene_types: Optional[List[SceneType]] = None
    target_sample_rate: int = 16000
    seed: int = 42
    min_duration: float = 2.0           # 最短音频时长(秒)
    max_duration: float = 30.0          # 最长音频时长(秒)
    use_reverb: bool = True
    reverb_probability: float = 0.5


@dataclass
class DatasetStatistics:
    """数据集统计信息"""
    name: str
    version: str
    data_dir: str
    exists: bool
    total_files: int = 0
    total_duration_hours: float = 0.0
    categories: Dict[str, Dict] = field(default_factory=dict)
    # categories: {category_name: {file_count, duration_hours}}


# ===========================
# 抽象基类
# ===========================


class BaseDataset(ABC):
    """
    降噪测评数据集抽象基类

    所有标准数据集（DNS Challenge, VoiceBank-DEMAND, WHAM!等）
    都应继承此类并实现抽象方法。

    使用方式:
        dataset = VoiceBankDemand(data_dir="./data/datasets/voicebank_demand")
        if not dataset.is_downloaded:
            dataset.download()
        pairs = dataset.prepare_evaluation_set(EvaluationSetConfig(n_samples=50))
    """

    def __init__(self, data_dir: str, sample_rate: int = 16000):
        """
        初始化数据集

        Args:
            data_dir: 数据集存储根目录
            sample_rate: 目标采样率
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        os.makedirs(data_dir, exist_ok=True)
        self._metadata_file = os.path.join(data_dir, "dataset_meta.json")

    # ===========================
    # 抽象方法
    # ===========================

    @abstractmethod
    def download(self, force: bool = False) -> bool:
        """
        下载数据集

        Args:
            force: 是否强制重新下载

        Returns:
            是否下载成功
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, bool]:
        """
        验证数据集完整性

        Returns:
            {check_name: passed} 字典

        检查项:
        - file_count: 文件数量是否正确
        - checksums: 校验和是否匹配
        - sample_rates: 采样率是否符合预期
        - directory_structure: 目录结构是否完整
        """
        pass

    @abstractmethod
    def get_statistics(self) -> DatasetStatistics:
        """
        获取数据集统计信息

        Returns:
            DatasetStatistics 对象
        """
        pass

    @abstractmethod
    def get_meta(self) -> DatasetMeta:
        """
        获取数据集元数据

        Returns:
            DatasetMeta 对象
        """
        pass

    @abstractmethod
    def prepare_evaluation_set(
        self, config: Optional[EvaluationSetConfig] = None
    ) -> List[SamplePair]:
        """
        准备评测用的样本对列表

        Args:
            config: 评测集配置

        Returns:
            SamplePair 列表
        """
        pass

    # ===========================
    # 通用方法
    # ===========================

    @property
    def is_downloaded(self) -> bool:
        """检查数据集是否已下载"""
        return os.path.exists(self._metadata_file)

    def save_metadata(self, meta: DatasetMeta):
        """保存数据集元数据到磁盘"""
        os.makedirs(os.path.dirname(self._metadata_file), exist_ok=True)
        with open(self._metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._meta_to_dict(meta), f, indent=2, ensure_ascii=False)

    def load_metadata(self) -> Optional[DatasetMeta]:
        """从磁盘加载数据集元数据"""
        if not os.path.exists(self._metadata_file):
            return None
        try:
            with open(self._metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._dict_to_meta(data)
        except Exception:
            return None

    def scan_audio_files(self, directory: str, extensions: tuple = (".wav", ".flac", ".mp3")) -> List[str]:
        """
        扫描目录中的音频文件

        Args:
            directory: 目标目录
            extensions: 支持的扩展名

        Returns:
            音频文件路径列表
        """
        files = []
        if not os.path.exists(directory):
            return files
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if f.lower().endswith(extensions):
                    files.append(os.path.join(root, f))
        return sorted(files)

    def compute_audio_duration(self, filepath: str) -> Optional[float]:
        """计算音频文件时长"""
        try:
            import soundfile as sf
            info = sf.info(filepath)
            return info.duration
        except Exception:
            return None

    def compute_checksum(self, filepath: str, algorithm: str = "md5") -> str:
        """计算文件校验和"""
        hash_obj = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def verify_checksums(self, checksum_file: str) -> Dict[str, bool]:
        """
        验证校验和文件

        Args:
            checksum_file: 校验和文件路径 (每行: checksum  filepath)

        Returns:
            {filepath: valid} 字典
        """
        results = {}
        if not os.path.exists(checksum_file):
            return results

        base_dir = os.path.dirname(checksum_file)
        with open(checksum_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    expected_hash, rel_path = parts[0], parts[-1]
                    file_path = os.path.join(base_dir, rel_path)
                    if os.path.exists(file_path):
                        actual_hash = self.compute_checksum(file_path)
                        results[rel_path] = actual_hash == expected_hash
                    else:
                        results[rel_path] = False
        return results

    @staticmethod
    def _meta_to_dict(meta: DatasetMeta) -> dict:
        """DatasetMeta 转字典"""
        return {
            "name": meta.name,
            "version": meta.version,
            "organization": meta.organization,
            "description": meta.description,
            "url": meta.url,
            "license_info": meta.license_info,
            "citation": meta.citation,
            "total_duration_hours": meta.total_duration_hours,
            "total_files": meta.total_files,
            "sample_rate": meta.sample_rate,
            "categories": [c.value for c in meta.categories],
            "known_baselines": meta.known_baselines,
        }

    @staticmethod
    def _dict_to_meta(data: dict) -> DatasetMeta:
        """字典转 DatasetMeta"""
        categories = []
        for c in data.get("categories", []):
            try:
                categories.append(DatasetCategory(c))
            except ValueError:
                pass
        return DatasetMeta(
            name=data.get("name", ""),
            version=data.get("version", "1.0"),
            organization=data.get("organization", ""),
            description=data.get("description", ""),
            url=data.get("url"),
            license_info=data.get("license_info", ""),
            citation=data.get("citation", ""),
            total_duration_hours=data.get("total_duration_hours", 0.0),
            total_files=data.get("total_files", 0),
            sample_rate=data.get("sample_rate", 16000),
            categories=categories,
            known_baselines=data.get("known_baselines", {}),
        )
