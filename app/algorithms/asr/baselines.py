"""
ASR 业界公开基准数据
用作业内对比参照，来源为各模型官方论文/模型卡/官方 Benchmark

数据说明：
- CER: 字错误率 (百分比)，越低越好
- 标注 "-" 表示该模型在该数据集上无公开数据
- 各论文使用不同数据版本，横向对比仅供参考
"""
from typing import Dict, Optional

# 基准数据: {算法名: {数据集名: {"cer": float, "source": str}}}
BASELINES: Dict[str, Dict[str, dict]] = {
    # ── 项目已注册算法 ──
    "paraformer-large": {
        "aishell1_test": {"cer": 1.94, "source": "FunASR Official Benchmark"},
        "wenetspeech_test": {"cer": 6.66, "source": "FunASR Benchmark Pipeline"},
        "wenetspeech_test_meeting": {"cer": 7.01, "source": "FunASR Benchmark Pipeline"},
        "thchs30_test": {"cer": None, "source": ""},
    },
    "sensevoice-small": {
        "aishell1_test": {"cer": 3.03, "source": "SenseVoice Paper (arXiv 2407.04051)"},
        "wenetspeech_test": {"cer": None, "source": ""},
        "wenetspeech_test_meeting": {"cer": None, "source": ""},
        "thchs30_test": {"cer": None, "source": ""},
    },
    "wenet-u2pp": {
        "aishell1_test": {"cer": 4.63, "source": "WeNet 2.0 (Interspeech 2022)"},
        "wenetspeech_test": {"cer": 9.25, "source": "WeNet 2.0 (MER, not CER)"},
        "wenetspeech_test_meeting": {"cer": 16.18, "source": "WeNet 2.0 (MER, not CER)"},
        "thchs30_test": {"cer": None, "source": ""},
    },
    "whisper-large-v3-turbo": {
        "aishell1_test": {"cer": 8.64, "source": "BELLE-2 Evaluation"},
        "wenetspeech_test": {"cer": None, "source": ""},
        "wenetspeech_test_meeting": {"cer": None, "source": ""},
        "thchs30_test": {"cer": None, "source": ""},
    },
    "firered-asr2": {
        "aishell1_test": {"cer": 0.57, "source": "FireRedASR2 Official (FireRedASR2-AED)"},
        "wenetspeech_test": {"cer": None, "source": ""},
        "wenetspeech_test_meeting": {"cer": None, "source": ""},
        "thchs30_test": {"cer": None, "source": ""},
    },
    "qwen3-asr": {
        "aishell1_test": {"cer": 1.50, "source": "Qwen3-ASR-1.7B Model Card"},
        "wenetspeech_test": {"cer": 4.55, "source": "Qwen3-ASR-1.7B Model Card"},
        "wenetspeech_test_meeting": {"cer": 4.69, "source": "Qwen3-ASR-1.7B Model Card"},
        "thchs30_test": {"cer": None, "source": ""},
    },
    "funasr-llm": {
        "aishell1_test": {"cer": 1.81, "source": "Fun-ASR-Nano Model Card (HF)"},
        "wenetspeech_test": {"cer": 6.33, "source": "Fun-ASR-Nano Model Card"},
        "wenetspeech_test_meeting": {"cer": 6.73, "source": "Fun-ASR-Nano Model Card"},
        "thchs30_test": {"cer": None, "source": ""},
    },
    # ── 未注册但代码已存在的适配器 ──
    "step-audio-2-mini": {
        "aishell1_test": {"cer": None, "source": ""},
    },
    "vibevoice-asr": {
        "aishell1_test": {"cer": None, "source": ""},
    },
}

# 数据集显示名称映射
DATASET_DISPLAY_NAMES = {
    "aishell1_test": "AISHELL-1",
    "wenetspeech_test": "WenetSpeech Net",
    "wenetspeech_test_meeting": "WenetSpeech Meeting",
    "thchs30_test": "THCHS-30",
    "builtin": "内置测试集",
}


def get_baseline(algorithm: str, dataset: str) -> Optional[dict]:
    """获取指定算法在指定数据集上的公开基准"""
    algo_data = BASELINES.get(algorithm, {})
    return algo_data.get(dataset)


def format_baseline_table(algorithms: list, dataset: str) -> list:
    """返回基准数据表格行，用于报告生成"""
    rows = []
    for algo in algorithms:
        baseline = get_baseline(algo, dataset)
        if baseline and baseline["cer"] is not None:
            rows.append({
                "algorithm": algo,
                "expected_cer": baseline["cer"],
                "source": baseline["source"],
            })
    return rows
