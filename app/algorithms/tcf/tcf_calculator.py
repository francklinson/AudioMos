"""
@file: tcf_calculator.py
@time: 2026/5/28
@desc: TCF 音色还原度计算模块（已迁移）

注意：此文件已迁移到 app/core/calculator/mos_calculator.py
保留此文件是为了向后兼容
"""
import warnings
import sys
import os

# 添加 core 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core'))

# 从新的位置导入所有内容
from calculator.mos_calculator import (
    ParallelMOSCompute,
    OptimizedDNSMOScore,
    OptimizedNisqaMosScore,
    OptimizedScoreqScore,
    OptimizedRefScore,
    OptimizedWerScore,
    OptimizedToneColorFidelityScore,
    AudioCache,
    PerformanceTimer,
)

# 发出弃用警告
warnings.warn(
    "tcf_calculator.py 已迁移到 app/core/calculator/mos_calculator.py，"
    "请更新导入路径。此文件将在未来版本中移除。",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'ParallelMOSCompute',
    'OptimizedDNSMOScore',
    'OptimizedNisqaMosScore',
    'OptimizedScoreqScore',
    'OptimizedRefScore',
    'OptimizedWerScore',
    'OptimizedToneColorFidelityScore',
    'AudioCache',
    'PerformanceTimer',
]
