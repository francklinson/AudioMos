"""
降噪测评配置模板

提供预设的实验配置模板，支持一键启动标准测评流程。

可用模板:
- quick_check: 快速功能验证（10个样本，内置数据）
- standard_comparison: 标准基准对比（30个样本，内置数据）
- voicebank_benchmark: VoiceBank-DEMAND 标准评测
- dns_benchmark: DNS Challenge 完整评测
- robustness_analysis: 鲁棒性分析（多噪声×多SNR）
- efficiency_test: 计算效率专项测试
- custom_algorithm_validation: 自研算法快速验证
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


# ===========================
# 预设模板
# ===========================


@dataclass
class ConfigTemplate:
    """配置模板"""
    name: str
    description: str
    dataset_key: str = "builtin"
    algorithms: Optional[List[str]] = None  # None = 使用所有已注册算法
    dataset_config: Dict = field(default_factory=dict)
    significance_test: bool = True
    visualization: bool = True
    efficiency_profile: bool = False
    tags: List[str] = field(default_factory=list)
    recommended_metrics: List[str] = field(default_factory=list)
    notes: str = ""


# ===========================
# 预定义模板集合
# ===========================


PRESET_TEMPLATES: Dict[str, ConfigTemplate] = {
    # ── 快速验证 ──
    "quick_check": ConfigTemplate(
        name="快速功能验证",
        description="使用内置合成数据，快速验证所有降噪算法的基本功能是否正常",
        dataset_key="builtin",
        dataset_config={"n_samples": 10, "snr_levels": [5, 10], "use_reverb": False},
        significance_test=False,
        visualization=False,
        efficiency_profile=False,
        tags=["quick", "validation"],
        recommended_metrics=["pesq", "stoi", "sisdr"],
        notes="适用于CI/CD或快速回归测试。运行时间: ~2分钟",
    ),

    # ── 标准对比 ──
    "standard_comparison": ConfigTemplate(
        name="标准基准对比",
        description="使用内置测试数据，对多个降噪算法进行标准化对比测评",
        dataset_key="builtin",
        dataset_config={
            "n_samples": 30,
            "snr_levels": [0, 5, 10, 15],
            "use_reverb": True,
            "noise_types": ["stationary", "babble", "traffic"],
        },
        significance_test=True,
        visualization=True,
        efficiency_profile=False,
        tags=["standard", "comparison"],
        recommended_metrics=["pesq", "stoi", "sisdr", "dnsmos_ovrl"],
        notes="适合日常开发中的算法验证和对比。运行时间: ~10-30分钟",
    ),

    # ── VoiceBank-DEMAND ──
    "voicebank_benchmark": ConfigTemplate(
        name="VoiceBank-DEMAND 标准评测",
        description="在VoiceBank-DEMAND官方测试集上评测，输出可与论文基线对比的结果",
        dataset_key="voicebank_demand",
        dataset_config={
            "n_samples": 50,
            "snr_levels": [2.5, 7.5, 12.5, 17.5],
            "min_duration": 1.0,
            "max_duration": 10.0,
        },
        significance_test=True,
        visualization=True,
        efficiency_profile=True,
        tags=["standard", "voicebank", "paper_baseline"],
        recommended_metrics=["pesq", "stoi", "sisdr", "csig", "cbak", "covl"],
        notes="VoiceBank-DEMAND 是语音增强领域最常用的基准。"
              "运行时间: ~30-60分钟（取决于算法数量）",
    ),

    # ── DNS Challenge ──
    "dns_benchmark": ConfigTemplate(
        name="DNS Challenge 完整评测",
        description="在DNS Challenge标准测试集上评测，业界最权威的降噪基准",
        dataset_key="dns_challenge",
        dataset_config={
            "n_samples": 100,
            "snr_levels": [0, 5, 10, 15],
            "use_reverb": True,
            "min_duration": 2.0,
            "max_duration": 30.0,
        },
        significance_test=True,
        visualization=True,
        efficiency_profile=True,
        tags=["standard", "dns_challenge", "authoritative"],
        recommended_metrics=["pesq", "stoi", "sisdr", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"],
        notes="DNS Challenge是降噪领域最权威的评测标准。"
              "需要下载完整DNS数据集(>80GB)。运行时间: 数小时",
    ),

    # ── 鲁棒性分析 ──
    "robustness_analysis": ConfigTemplate(
        name="鲁棒性分析",
        description="评测算法在不同噪声类型和SNR条件下的鲁棒性",
        dataset_key="builtin",
        dataset_config={
            "n_samples": 60,
            "snr_levels": [-5, 0, 5, 10, 15, 20],
            "noise_types": ["stationary", "babble", "traffic", "cafe", "factory"],
            "use_reverb": True,
        },
        significance_test=True,
        visualization=True,
        efficiency_profile=False,
        tags=["analysis", "robustness"],
        recommended_metrics=["pesq", "stoi", "sisdr", "dnsmos_ovrl"],
        notes="用于发现算法在不同条件下的优势和劣势。"
              "结果中会按噪声类型和SNR级别分别汇总。运行时间: ~30-60分钟",
    ),

    # ── 效率测试 ──
    "efficiency_test": ConfigTemplate(
        name="计算效率专项测试",
        description="系统性评估算法的计算资源消耗（RTF、内存、GPU利用率）",
        dataset_key="builtin",
        dataset_config={
            "n_samples": 20,
            "snr_levels": [10],
            "min_duration": 1.0,
            "max_duration": 60.0,
            "use_reverb": False,
        },
        significance_test=False,
        visualization=True,
        efficiency_profile=True,
        tags=["efficiency", "performance"],
        recommended_metrics=["processing_time", "rtf"],
        notes="测试不同音频时长下的RTF和内存使用。"
              "适合评估算法是否满足实时处理要求(RTF<1)。运行时间: ~10-20分钟",
    ),

    # ── 自研算法验证 ──
    "custom_algorithm_validation": ConfigTemplate(
        name="自研算法验证",
        description="针对单个自研算法的快速迭代测评",
        dataset_key="builtin",
        dataset_config={
            "n_samples": 20,
            "snr_levels": [0, 5, 10, 15],
            "use_reverb": True,
        },
        significance_test=False,  # 自研算法通常不需要与其他对比
        visualization=True,
        efficiency_profile=True,
        tags=["custom", "development"],
        recommended_metrics=["pesq", "stoi", "sisdr", "dnsmos_ovrl", "rtf"],
        notes="适用于自研算法开发过程中的快速迭代验证。"
              "建议只选择1-2个对比算法。运行时间: ~5-15分钟",
    ),

    # ── 论文基线复现 ──
    "paper_reproduction": ConfigTemplate(
        name="论文基线复现",
        description="复现论文中的基线结果，含完整统计检验和可视化",
        dataset_key="voicebank_demand",
        dataset_config={
            "n_samples": 50,
            "snr_levels": [2.5, 7.5, 12.5, 17.5],
            "min_duration": 1.0,
            "max_duration": 10.0,
        },
        significance_test=True,
        visualization=True,
        efficiency_profile=True,
        tags=["paper", "reproduction", "academic"],
        recommended_metrics=["pesq", "stoi", "sisdr", "csig", "cbak", "covl", "rtf"],
        notes="输出可直接用于论文的实验结果。"
              "包含完整的统计检验报告和可视化图表。",
    ),
}


# ===========================
# 模板管理器
# ===========================


class ConfigTemplateManager:
    """
    配置模板管理器

    提供模板的加载、保存和应用。

    使用方式:
        manager = ConfigTemplateManager()
        templates = manager.list_templates()
        config = manager.apply_template("standard_comparison", algorithms=["algo_a", "algo_b"])
        runner.run(config)
    """

    def __init__(self, user_templates_dir: Optional[str] = None):
        self.user_templates_dir = user_templates_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "templates"
        )
        os.makedirs(self.user_templates_dir, exist_ok=True)

    def list_templates(self) -> List[Dict]:
        """列出所有可用模板"""
        templates = []

        # 预置模板
        for name, tmpl in PRESET_TEMPLATES.items():
            templates.append({
                "name": name,
                "description": tmpl.description,
                "dataset_key": tmpl.dataset_key,
                "tags": tmpl.tags,
                "notes": tmpl.notes,
                "source": "builtin",
            })

        # 用户自定义模板
        if os.path.exists(self.user_templates_dir):
            for f in sorted(os.listdir(self.user_templates_dir)):
                if f.endswith(".json"):
                    name = f[:-5]
                    templates.append({
                        "name": name,
                        "description": f"用户自定义模板: {name}",
                        "source": "user",
                        "path": os.path.join(self.user_templates_dir, f),
                    })

        return templates

    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """获取指定模板"""
        if name in PRESET_TEMPLATES:
            return PRESET_TEMPLATES[name]

        # 尝试加载用户模板
        user_path = os.path.join(self.user_templates_dir, f"{name}.json")
        if os.path.exists(user_path):
            return self.load_template(name)

        logger.error(f"模板不存在: {name}")
        return None

    def apply_template(
        self, template_name: str, **overrides
    ) -> Optional['ExperimentConfig']:
        """
        应用模板生成实验配置

        Args:
            template_name: 模板名称
            **overrides: 覆盖参数

        Returns:
            ExperimentConfig 或 None
        """
        from ..experiment_runner import ExperimentConfig, EvaluatorConfig

        tmpl = self.get_template(template_name)
        if tmpl is None:
            return None

        # 构建配置
        config = ExperimentConfig(
            name=tmpl.name,
            description=tmpl.description,
            algorithms=tmpl.algorithms or [],
            dataset_key=tmpl.dataset_key,
            dataset_config=dict(tmpl.dataset_config),
            significance_test=tmpl.significance_test,
            visualization=tmpl.visualization,
            efficiency_profile=tmpl.efficiency_profile,
            tags=list(tmpl.tags),
        )

        # 应用覆盖
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif key in tmpl.dataset_config:
                config.dataset_config[key] = value

        return config

    def save_template(self, name: str, config: 'ExperimentConfig') -> str:
        """
        保存当前配置为用户模板

        Args:
            name: 模板名称
            config: 实验配置

        Returns:
            保存路径
        """
        template_dict = {
            "name": name,
            "description": config.description,
            "dataset_key": config.dataset_key,
            "algorithms": config.algorithms,
            "dataset_config": config.dataset_config,
            "significance_test": config.significance_test,
            "visualization": config.visualization,
            "efficiency_profile": config.efficiency_profile,
            "tags": config.tags,
        }

        save_path = os.path.join(self.user_templates_dir, f"{name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(template_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"模板已保存: {save_path}")
        return save_path

    def load_template(self, name: str) -> Optional[ConfigTemplate]:
        """加载用户自定义模板"""
        path = os.path.join(self.user_templates_dir, f"{name}.json")
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return ConfigTemplate(
                name=data.get("name", name),
                description=data.get("description", ""),
                dataset_key=data.get("dataset_key", "builtin"),
                algorithms=data.get("algorithms"),
                dataset_config=data.get("dataset_config", {}),
                significance_test=data.get("significance_test", True),
                visualization=data.get("visualization", True),
                efficiency_profile=data.get("efficiency_profile", False),
                tags=data.get("tags", []),
            )
        except Exception as e:
            logger.error(f"加载模板失败: {e}")
            return None

    def delete_template(self, name: str) -> bool:
        """删除用户模板"""
        path = os.path.join(self.user_templates_dir, f"{name}.json")
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"模板已删除: {name}")
            return True
        return False


# ===========================
# 便捷函数
# ===========================


def get_recommended_template(use_case: str) -> Optional[str]:
    """
    根据使用场景推荐模板

    Args:
        use_case: 使用场景
            - "quick" / "dev" → quick_check
            - "standard" / "benchmark" → standard_comparison
            - "paper" / "academic" → paper_reproduction
            - "efficiency" / "perf" → efficiency_test
            - "robustness" → robustness_analysis

    Returns:
        推荐的模板名称
    """
    mapping = {
        "quick": "quick_check",
        "dev": "quick_check",
        "development": "quick_check",
        "standard": "standard_comparison",
        "benchmark": "standard_comparison",
        "paper": "paper_reproduction",
        "academic": "paper_reproduction",
        "reproduction": "paper_reproduction",
        "efficiency": "efficiency_test",
        "perf": "efficiency_test",
        "performance": "efficiency_test",
        "robustness": "robustness_analysis",
        "custom": "custom_algorithm_validation",
    }

    template_name = mapping.get(use_case.lower())
    if template_name and template_name in PRESET_TEMPLATES:
        return template_name

    return "standard_comparison"  # 默认
