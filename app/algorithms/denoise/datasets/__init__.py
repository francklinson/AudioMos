"""
降噪测评标准数据集模块

提供统一的数据集管理接口，支持:
- DNS Challenge (Microsoft, 降噪领域最权威基准)
- VoiceBank-DEMAND (Edinburgh, 标准语音增强基准)
- WHAM! / WHAMR! (WSJ0混合数据集)
- 场景化测试数据构建器
"""

from .base import (
    BaseDataset,
    DatasetMeta,
    SamplePair,
    SampleMetadata,
    EvaluationSetConfig,
    DatasetCategory,
    NoiseType,
    SceneType,
)

__all__ = [
    'BaseDataset',
    'DatasetMeta',
    'SamplePair',
    'SampleMetadata',
    'EvaluationSetConfig',
    'DatasetCategory',
    'NoiseType',
    'SceneType',
]
