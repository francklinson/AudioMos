"""
降噪测评数据集统一管理器

提供统一的数据集注册、发现、下载和评测集准备入口。

使用方式:
    manager = DatasetManager(storage_dir="./data/datasets")
    manager.list_datasets()
    dataset = manager.get_dataset("voicebank_demand")
    pairs = manager.prepare_test_suite("standard_benchmark", n_samples=50)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

from .datasets.base import (
    BaseDataset,
    DatasetMeta,
    SamplePair,
    EvaluationSetConfig,
    DatasetStatistics,
    NoiseType,
    SceneType,
    SNR_LEVELS,
)

logger = logging.getLogger(__name__)


# ===========================
# 数据集注册信息
# ===========================

DATASET_REGISTRY: Dict[str, Dict] = {
    "dns_challenge": {
        "name": "DNS Challenge (DNS1-DNS5)",
        "organization": "Microsoft Research",
        "description": "业界标准的深度降噪评估基准，包含多语言干净语音和丰富噪声库",
        "url": "https://github.com/microsoft/DNS-Challenge",
        "license_info": "需通过Microsoft DNS Challenge官网申请访问",
        "citation": "Reddy et al., 'The INTERSPEECH 2020 Deep Noise Suppression Challenge', 2020",
        "module": ".datasets.dns",
        "class": "DNSChallengeDataset",
        "recommended_metrics": ["pesq", "stoi", "sisdr", "dnsmos_ovrl"],
        "estimated_size": ">80GB",
        "known_baselines": {
            "dns_challenge_baseline": {"pesq": 2.15, "stoi": 0.91, "sisdr": 15.0},
        },
    },
    "voicebank_demand": {
        "name": "VoiceBank-DEMAND",
        "organization": "University of Edinburgh",
        "description": "标准语音增强数据集，28个训练说话人+2个测试说话人，含DEMAND和人工噪声",
        "url": "https://datashare.ed.ac.uk/handle/10283/2791",
        "license_info": "CC BY 4.0",
        "citation": "Valentini et al., 'Investigating RNN-based speech enhancement...', 2016",
        "module": ".datasets.voicebank_demand",
        "class": "VoicebankDemandDataset",
        "recommended_metrics": ["pesq", "stoi", "sisdr", "csig", "cbak", "covl"],
        "estimated_size": "~8GB",
        "known_baselines": {
            "segan": {"pesq": 2.16, "stoi": 0.890},
            "metricgan_plus": {"pesq": 3.15, "stoi": 0.930},
            "dccrn": {"pesq": 2.84, "stoi": 0.910},
        },
    },
    "wham": {
        "name": "WHAM! / WHAMR!",
        "organization": "MERL & Mitsubishi Electric",
        "description": "WSJ0混合数据集，WHAM!含环境噪声，WHAMR!额外添加混响",
        "url": "https://wham.whisper.ai/",
        "license_info": "需先获取WSJ0许可证（LDC），WHAM部分CC BY-NC 4.0",
        "citation": "Wichern et al., 'WHAM!: Extending Speech Separation...', 2019",
        "module": ".datasets.wham_whamr",
        "class": "WhamDataset",
        "recommended_metrics": ["sisdr", "sdr", "pesq", "stoi"],
        "estimated_size": "~20GB (不含WSJ0)",
        "known_baselines": {
            "sepformer_wham": {"sisdr": 20.8, "sdr": 21.2},
            "sepformer_whamr": {"sisdr": 14.0, "sdr": 14.4},
        },
    },
    "builtin": {
        "name": "项目内置测试集",
        "organization": "AudioMos项目",
        "description": "使用项目已有的测试音频和合成数据，支持快速功能验证",
        "url": None,
        "license_info": "项目内部使用",
        "citation": "",
        "module": ".datasets.dns",
        "class": "BuiltinDataset",
        "recommended_metrics": ["pesq", "stoi", "sisdr", "dnsmos_ovrl", "nisqa_mos", "utmos"],
        "estimated_size": "<100MB",
        "known_baselines": {},
    },
}


# ===========================
# 预定义测试套件
# ===========================

PREDEFINED_TEST_SUITES: Dict[str, Dict] = {
    "quick_check": {
        "description": "快速功能验证（内置合成数据，10个样本）",
        "dataset": "builtin",
        "config": {"n_samples": 10, "snr_levels": [5, 10], "use_reverb": False},
    },
    "standard_benchmark": {
        "description": "标准基准测试（使用项目测试数据，30个样本）",
        "dataset": "builtin",
        "config": {"n_samples": 30, "snr_levels": [0, 5, 10, 15], "use_reverb": True},
    },
    "dns_full": {
        "description": "DNS Challenge完整评测（100个样本，4级SNR）",
        "dataset": "dns_challenge",
        "config": {"n_samples": 100, "snr_levels": [0, 5, 10, 15], "use_reverb": True},
    },
    "voicebank_standard": {
        "description": "VoiceBank-DEMAND标准评测（官方测试集）",
        "dataset": "voicebank_demand",
        "config": {"n_samples": 50, "snr_levels": [2.5, 7.5, 12.5, 17.5]},
    },
    "robustness_test": {
        "description": "鲁棒性测试（多噪声类型×多SNR，50个样本）",
        "dataset": "builtin",
        "config": {
            "n_samples": 50,
            "snr_levels": [-5, 0, 5, 10, 15, 20],
            "noise_types": ["stationary", "babble", "traffic", "cafe"],
            "use_reverb": True,
        },
    },
    "efficiency_test": {
        "description": "计算效率专项测试（不同时长音频，20个样本）",
        "dataset": "builtin",
        "config": {
            "n_samples": 20,
            "snr_levels": [10],
            "min_duration": 1.0,
            "max_duration": 60.0,
            "use_reverb": False,
        },
    },
}


# ===========================
# 数据集管理器
# ===========================


class DatasetManager:
    """
    数据集统一管理器

    负责数据集的注册、发现、懒加载和评测集准备。
    """

    def __init__(self, storage_dir: str = "./data/datasets"):
        """
        初始化数据集管理器

        Args:
            storage_dir: 数据集存储根目录
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._loaded_datasets: Dict[str, BaseDataset] = {}

    def list_datasets(self) -> List[Dict]:
        """
        列出所有已注册的数据集信息

        Returns:
            数据集信息字典列表
        """
        result = []
        for key, info in DATASET_REGISTRY.items():
            ds_dir = os.path.join(self.storage_dir, key)
            is_available = os.path.exists(ds_dir) and os.listdir(ds_dir)

            result.append({
                "key": key,
                "name": info["name"],
                "organization": info["organization"],
                "description": info["description"],
                "url": info.get("url"),
                "license_info": info.get("license_info", ""),
                "citation": info.get("citation", ""),
                "estimated_size": info.get("estimated_size", "未知"),
                "recommended_metrics": info.get("recommended_metrics", []),
                "downloaded": is_available,
                "known_baselines": info.get("known_baselines", {}),
            })
        return result

    def get_dataset(self, key: str) -> Optional[BaseDataset]:
        """
        获取数据集实例（懒加载）

        Args:
            key: 数据集键名

        Returns:
            BaseDataset 实例或 None
        """
        if key in self._loaded_datasets:
            return self._loaded_datasets[key]

        if key not in DATASET_REGISTRY:
            logger.error(f"未知数据集: {key}，可用: {list(DATASET_REGISTRY.keys())}")
            return None

        info = DATASET_REGISTRY[key]
        ds_dir = os.path.join(self.storage_dir, key)

        # 动态导入并实例化
        try:
            module_path = info["module"]
            class_name = info["class"]

            # 直接导入具体模块而非使用importlib（确保包上下文正确）
            if key == "dns_challenge" or key == "builtin":
                from .datasets.dns import DNSChallengeDataset, BuiltinDataset
                dataset_class = BuiltinDataset if class_name == "BuiltinDataset" else DNSChallengeDataset
            elif key == "voicebank_demand":
                from .datasets.voicebank_demand import VoicebankDemandDataset
                dataset_class = VoicebankDemandDataset
            elif key == "wham":
                from .datasets.wham_whamr import WhamDataset
                dataset_class = WhamDataset
            else:
                # 回退: 尝试 importlib（带完整包名）
                import importlib
                full_module = f"denoise{module_path}"  # .datasets.dns → denoise.datasets.dns
                module = importlib.import_module(full_module)
                dataset_class = getattr(module, class_name)

            dataset = dataset_class(data_dir=ds_dir)
            self._loaded_datasets[key] = dataset
            return dataset
        except Exception as e:
            logger.error(f"加载数据集 {key} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def download_if_needed(self, key: str, force: bool = False) -> bool:
        """
        按需下载数据集

        Args:
            key: 数据集键名
            force: 是否强制重新下载

        Returns:
            是否下载成功（或已存在）
        """
        dataset = self.get_dataset(key)
        if dataset is None:
            return False

        if dataset.is_downloaded and not force:
            logger.info(f"数据集 {key} 已下载")
            return True

        logger.info(f"开始下载数据集: {key}...")
        success = dataset.download(force=force)
        if success:
            logger.info(f"数据集 {key} 下载完成")
        else:
            logger.error(f"数据集 {key} 下载失败")

        return success

    def get_statistics(self, key: str) -> Optional[DatasetStatistics]:
        """获取数据集统计信息"""
        dataset = self.get_dataset(key)
        if dataset is None:
            return None
        return dataset.get_statistics()

    def prepare_evaluation_set(
        self, key: str, config: Optional[EvaluationSetConfig] = None
    ) -> List[SamplePair]:
        """
        从指定数据集准备评测集

        Args:
            key: 数据集键名
            config: 评测集配置

        Returns:
            SamplePair 列表
        """
        dataset = self.get_dataset(key)
        if dataset is None:
            return []

        if not dataset.is_downloaded:
            logger.warning(f"数据集 {key} 未下载，尝试下载...")
            if not self.download_if_needed(key):
                logger.error(f"数据集 {key} 不可用")
                return []

        return dataset.prepare_evaluation_set(config)

    def prepare_test_suite(
        self, suite_name: str, **overrides
    ) -> Tuple[str, List[SamplePair], Dict]:
        """
        按预定义测试套件准备评测数据

        Args:
            suite_name: 套件名称（见 PREDEFINED_TEST_SUITES）
            **overrides: 覆盖配置参数

        Returns:
            (dataset_key, sample_pairs, suite_config) 元组

        可用套件: quick_check, standard_benchmark, dns_full,
                  voicebank_standard, robustness_test, efficiency_test
        """
        if suite_name not in PREDEFINED_TEST_SUITES:
            available = list(PREDEFINED_TEST_SUITES.keys())
            logger.error(f"未知测试套件: {suite_name}，可用: {available}")
            # 回退到 quick_check
            suite_name = "quick_check"

        suite_info = PREDEFINED_TEST_SUITES[suite_name]

        # 合并覆盖参数
        config_dict = suite_info.get("config", {}).copy()
        config_dict.update(overrides)

        # 如果有noise_types字符串列表，转换为枚举
        if "noise_types" in config_dict and isinstance(config_dict["noise_types"], list):
            config_dict["noise_types"] = [
                NoiseType(n) if isinstance(n, str) else n
                for n in config_dict["noise_types"]
            ]

        config = EvaluationSetConfig(**config_dict)

        dataset_key = suite_info["dataset"]
        pairs = self.prepare_evaluation_set(dataset_key, config)

        logger.info(
            f"测试套件 [{suite_name}] 准备完成: "
            f"数据集={dataset_key}, 样本数={len(pairs)}, "
            f"有参考={sum(1 for p in pairs if p.clean_path is not None)}"
        )

        return dataset_key, pairs, suite_info

    def list_test_suites(self) -> List[Dict]:
        """列出所有预定义测试套件"""
        return [
            {
                "name": name,
                "description": info["description"],
                "dataset": info["dataset"],
                "config": info.get("config", {}),
            }
            for name, info in PREDEFINED_TEST_SUITES.items()
        ]

    def validate_dataset(self, key: str) -> Dict[str, bool]:
        """验证数据集完整性"""
        dataset = self.get_dataset(key)
        if dataset is None:
            return {"error": False}
        return dataset.validate()

    def clear_cache(self):
        """清除所有已加载的数据集缓存"""
        self._loaded_datasets.clear()

    def get_download_info(self, key: str) -> str:
        """获取数据集的下载指引"""
        if key not in DATASET_REGISTRY:
            return f"未知数据集: {key}"

        info = DATASET_REGISTRY[key]
        parts = [
            f"# {info['name']} 下载指引",
            f"",
            f"- 发布组织: {info['organization']}",
            f"- 许可证: {info['license_info']}",
            f"- 估计大小: {info['estimated_size']}",
            f"- 论文引用: {info['citation']}",
        ]
        if info.get("url"):
            parts.append(f"- 项目地址: {info['url']}")

        dataset = self.get_dataset(key)
        if dataset is not None:
            parts.extend(dataset._get_download_instructions())

        return "\n".join(parts)
