"""
场景化测试数据构建器

按噪声类型、SNR级别、场景类型组织评测数据。
支持:
- 按噪声类型标签筛选/聚合
- 按SNR级别分层采样
- 按场景类型分类

使用方式:
    builder = SceneBuilder(output_dir="./data/datasets/scenes")
    pairs = builder.build_scene_suite(
        noise_types=[NoiseType.BABBLE, NoiseType.TRAFFIC],
        snr_levels=[0, 5, 10],
        n_samples_per_condition=10,
    )
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from .base import BaseDataset, SamplePair, SampleMetadata, NoiseType, SceneType, SNR_LEVELS

logger = logging.getLogger(__name__)


@dataclass
class SceneBuildConfig:
    """场景构建配置"""
    noise_types: List[NoiseType] = field(default_factory=lambda: [NoiseType.BABBLE])
    scene_types: List[SceneType] = field(default_factory=lambda: [SceneType.UNKNOWN_SCENE])
    snr_levels: List[float] = field(default_factory=lambda: [0, 5, 10])
    n_samples_per_condition: int = 5
    min_duration: float = 2.0
    max_duration: float = 10.0
    sample_rate: int = 16000
    seed: int = 42


class SceneBuilder:
    """
    场景化测试数据构建器

    将标准数据集中的样本按场景标签组织和聚合。
    """

    def __init__(self, output_dir: str = "./data/datasets/scenes"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build_scene_suite(self, config: SceneBuildConfig) -> Dict[str, List[SamplePair]]:
        """
        构建场景化测试套件

        Args:
            config: 构建配置

        Returns:
            {condition_key: sample_pairs} 字典
            例如: {"babble_5dB": [...], "traffic_10dB": [...]}
        """
        from .dns import _BuiltinGenerator

        suites = {}
        gen = _BuiltinGenerator(self.output_dir, config.sample_rate)

        for noise_type in config.noise_types:
            for snr in config.snr_levels:
                condition_key = f"{noise_type.value}_{int(snr)}dB"

                from .base import EvaluationSetConfig
                eval_config = EvaluationSetConfig(
                    n_samples=config.n_samples_per_condition,
                    snr_levels=[snr],
                    noise_types=[noise_type],
                    min_duration=config.min_duration,
                    max_duration=config.max_duration,
                    target_sample_rate=config.sample_rate,
                    seed=config.seed,
                    use_reverb=True,
                )

                pairs = gen.generate(eval_config)
                suites[condition_key] = pairs

                logger.info(f"场景 [{condition_key}]: 生成 {len(pairs)} 个样本")

        return suites

    def tag_samples(
        self, pairs: List[SamplePair],
        noise_type: Optional[NoiseType] = None,
        scene_type: Optional[SceneType] = None,
        snr_db: Optional[float] = None,
    ) -> List[SamplePair]:
        """为样本打标签"""
        for p in pairs:
            if noise_type and p.metadata.noise_type == NoiseType.UNKNOWN:
                p.metadata.noise_type = noise_type
            if scene_type and p.metadata.scene_type == SceneType.UNKNOWN_SCENE:
                p.metadata.scene_type = scene_type
            if snr_db is not None and p.metadata.snr_db is None:
                p.metadata.snr_db = snr_db
        return pairs

    def group_by_noise_type(self, pairs: List[SamplePair]) -> Dict[NoiseType, List[SamplePair]]:
        """按噪声类型分组"""
        groups = {}
        for p in pairs:
            nt = p.metadata.noise_type
            groups.setdefault(nt, []).append(p)
        return groups

    def group_by_snr(self, pairs: List[SamplePair]) -> Dict[float, List[SamplePair]]:
        """按SNR分组"""
        groups = {}
        for p in pairs:
            snr = p.metadata.snr_db
            if snr is not None:
                groups.setdefault(round(snr), []).append(p)
        return groups

    def group_by_scene(self, pairs: List[SamplePair]) -> Dict[SceneType, List[SamplePair]]:
        """按场景类型分组"""
        groups = {}
        for p in pairs:
            st = p.metadata.scene_type
            groups.setdefault(st, []).append(p)
        return groups

    def get_statistics(self, pairs: List[SamplePair]) -> Dict:
        """获取样本集的场景统计"""
        stats = {
            "total_pairs": len(pairs),
            "has_clean_reference": sum(1 for p in pairs if p.clean_path is not None),
            "by_noise_type": {},
            "by_snr": {},
        }

        for nt, group in self.group_by_noise_type(pairs).items():
            stats["by_noise_type"][nt.value] = len(group)

        for snr, group in self.group_by_snr(pairs).items():
            stats["by_snr"][str(snr)] = len(group)

        return stats


def build_standard_scene_suite(
    output_dir: str = "./data/datasets/scenes",
    n_per_condition: int = 10,
) -> Dict[str, List[SamplePair]]:
    """
    构建标准场景测试套件（便捷函数）

    覆盖4种常见噪声×4级SNR = 16个场景条件

    Args:
        output_dir: 输出目录
        n_per_condition: 每个条件的样本数

    Returns:
        场景套件
    """
    builder = SceneBuilder(output_dir)
    config = SceneBuildConfig(
        noise_types=[NoiseType.STATIONARY, NoiseType.BABBLE, NoiseType.TRAFFIC, NoiseType.CAFE],
        snr_levels=[0, 5, 10, 15],
        n_samples_per_condition=n_per_condition,
    )
    return builder.build_scene_suite(config)
