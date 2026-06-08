"""
降噪算法模块
提供多种业界先进的音频降噪/增强算法

支持算法:
- 传统方法: 谱减法, 维纳滤波
- SpeechBrain: MetricGAN+, SepFormer
- ClearerVoice: FRCRN, MossFormer, MossFormer2
- DCCRN: 复数卷积循环网络
- FullSubNet: 全带子带融合网络

新增能力:
- 数据集管理: DNS Challenge, VoiceBank-DEMAND, WHAM!, 场景化构建
- 统计检验: t检验, Wilcoxon, Bootstrap CI, Cohen's d
- 可视化: 雷达图, 箱线图, 柱状图, 热力图, 仪表盘
- 效率分析: RTF, 内存/GPU监控, 模型参数量
- 实验管理: 配置模板, SQLite持久化, 历史对比
- 插件系统: 自研算法接入规范, ONNX/TensorRT适配
"""

from .base import BaseDenoiser, DenoiseResult
from .registry import DenoiserRegistry, get_available_denoisers, get_denoiser_description
from .speechbrain_denoiser import SpeechBrainDenoiser
from .traditional_denoiser import TraditionalDenoiser

# ── 测评引擎 ──
from .evaluator import DenoiseEvaluator, DenoiseMetrics, DenoiseEvaluation, BatchEvaluator
from .benchmark import BenchmarkRunner, BenchmarkResult, AlgorithmBenchmarkResult, run_quick_benchmark
from .report_generator import ReportGenerator

# ── 新增: 统计检验 ──
try:
    from .significance import (
        StatisticalAnalyzer,
        TTestResult,
        WilcoxonResult,
        EffectSize,
        MetricComparison,
        ComparisonReport,
        FullComparisonMatrix,
    )
    SIGNIFICANCE_AVAILABLE = True
except ImportError:
    SIGNIFICANCE_AVAILABLE = False

# ── 新增: 可视化 ──
try:
    from .visualizer import BenchmarkVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

# ── 新增: 效率分析 ──
try:
    from .efficiency_profiler import EfficiencyProfiler, EfficiencyMetrics, EfficiencyReport, quick_efficiency_check
    EFFICIENCY_AVAILABLE = True
except ImportError:
    EFFICIENCY_AVAILABLE = False

# ── 新增: 数据集管理 ──
try:
    from .dataset_manager import DatasetManager
    from .datasets import (
        BaseDataset,
        DatasetMeta,
        SamplePair,
        SampleMetadata,
        EvaluationSetConfig,
        NoiseType,
        SceneType,
    )
    DATASET_MANAGER_AVAILABLE = True
except ImportError:
    DATASET_MANAGER_AVAILABLE = False

# ── 新增: 实验管理 ──
try:
    from .experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult, run_quick_experiment
    from .experiment_db import ExperimentDB, ExperimentSummary
    from .experiment_comparator import ExperimentComparator, TrendReport
    EXPERIMENT_AVAILABLE = True
except ImportError:
    EXPERIMENT_AVAILABLE = False

# ── 新增: 配置模板 ──
try:
    from .config_templates import (
        ConfigTemplateManager,
        PRESET_TEMPLATES,
        get_recommended_template,
    )
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False

# ── 新增: 插件系统 ──
try:
    from .plugin_spec import (
        PluginMetadata,
        CustomDenoiserPlugin,
        PluginLoader,
        register_plugin,
    )
    PLUGIN_AVAILABLE = True
except ImportError:
    PLUGIN_AVAILABLE = False

try:
    from .model_format_adapter import ModelFormatAdapter
    FORMAT_ADAPTER_AVAILABLE = True
except ImportError:
    FORMAT_ADAPTER_AVAILABLE = False

# ── ClearVoice-Studio 降噪算法 ──
try:
    from .clearervoice_denoiser import (
        ClearVoiceWrapperDenoiser,
        FRCRNSE16KDenoiser,
        MossFormer2SE48KDenoiser,
        MossFormerGANSE16KDenoiser,
        MossFormer2SS16KDenoiser,
        MossFormer2SR48KDenoiser,
        # 向后兼容
        FRCRNDenoiser,
        MossFormerDenoiser,
        MossFormer2Denoiser,
        ClearerVoiceDenoiser,
        CLEARVOICE_MODEL_SPECS,
    )
    CLEARERVOICE_AVAILABLE = True
except ImportError:
    CLEARERVOICE_AVAILABLE = False


__all__ = [
    # 核心
    'BaseDenoiser',
    'DenoiseResult',
    'DenoiserRegistry',
    'get_available_denoisers',
    'get_denoiser_description',
    'SpeechBrainDenoiser',
    'TraditionalDenoiser',
    # 测评
    'DenoiseEvaluator',
    'DenoiseMetrics',
    'DenoiseEvaluation',
    'BatchEvaluator',
    'BenchmarkRunner',
    'BenchmarkResult',
    'AlgorithmBenchmarkResult',
    'run_quick_benchmark',
    'ReportGenerator',
]

if CLEARERVOICE_AVAILABLE:
    __all__.extend([
        'ClearVoiceWrapperDenoiser',
        'FRCRNSE16KDenoiser',
        'MossFormer2SE48KDenoiser',
        'MossFormerGANSE16KDenoiser',
        'MossFormer2SS16KDenoiser',
        'MossFormer2SR48KDenoiser',
        'FRCRNDenoiser',
        'MossFormerDenoiser',
        'MossFormer2Denoiser',
        'ClearerVoiceDenoiser',
        'CLEARVOICE_MODEL_SPECS',
    ])

# 新增模块
if SIGNIFICANCE_AVAILABLE:
    __all__.extend(['StatisticalAnalyzer', 'ComparisonReport'])
if VISUALIZER_AVAILABLE:
    __all__.append('BenchmarkVisualizer')
if EFFICIENCY_AVAILABLE:
    __all__.extend(['EfficiencyProfiler', 'EfficiencyMetrics', 'quick_efficiency_check'])
if DATASET_MANAGER_AVAILABLE:
    __all__.extend(['DatasetManager', 'BaseDataset', 'SamplePair', 'NoiseType', 'SceneType'])
if EXPERIMENT_AVAILABLE:
    __all__.extend(['ExperimentRunner', 'ExperimentConfig', 'ExperimentDB', 'ExperimentComparator'])
if TEMPLATES_AVAILABLE:
    __all__.extend(['ConfigTemplateManager', 'get_recommended_template'])
if PLUGIN_AVAILABLE:
    __all__.extend(['CustomDenoiserPlugin', 'PluginLoader', 'register_plugin'])
if FORMAT_ADAPTER_AVAILABLE:
    __all__.append('ModelFormatAdapter')
