"""
ASR 算法单元测试
测试 BaseASR、ASRRegistry、适配器初始化等核心功能

运行方式:
  pytest tests/algorithms/test_asr.py -v -m "not slow"
  pytest tests/algorithms/test_asr.py -v                        # 含模型加载测试
"""

import pytest
import os
import sys
import json
import numpy as np
from pathlib import Path

# ==================== 基础数据 ====================

EXPECTED_ALGORITHMS = [
    "paraformer-large",
    "sensevoice-small",
    "wenet-u2pp",
    "whisper-large-v3-turbo",
    "firered-asr2",
    "qwen3-asr",
    "funasr-llm",
]

EXPECTED_PRELOAD = ["paraformer-large", "sensevoice-small", "wenet-u2pp"]

# 每个模型必须的关键文件（相对于 models/asr/<name>/）
MODEL_KEY_FILES = {
    "paraformer-large": ["model.pt", "config.yaml", "tokens.json"],
    "sensevoice-small": ["model.pt", "config.yaml", "tokens.json"],
    "wenet-u2pp": ["final.pt", "train.yaml", "units.txt"],
    "whisper-large-v3-turbo": ["model.safetensors", "config.json", "tokenizer.json"],
    "firered-asr2": ["model.pth.tar"],
    "qwen3-asr": ["model.safetensors.index.json", "config.json"],
    "funasr-llm": ["model.pt", "config.yaml", "configuration.json"],
}

REF_AUDIO = "data/ref/ref_001.wav"
GROUND_TRUTH = "他为儿子买了一整根甘蔗市区的停车收费将大幅提高他醒来后发现自己脸上有黑眼圈"


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def models_dir(project_root):
    return project_root / "models" / "asr"


@pytest.fixture(scope="session")
def ref_audio_path(project_root):
    return project_root / REF_AUDIO


@pytest.fixture(scope="session")
def ref_audio_numpy(ref_audio_path):
    """加载参考音频为 numpy 数组"""
    import soundfile as sf
    data, sr = sf.read(str(ref_audio_path))
    return data, sr


# ==================== 测试: 模型文件完整性 ====================

class TestModelFiles:
    """验证 models/asr/ 下所有模型文件存在且非空"""

    def test_all_model_dirs_exist(self, models_dir):
        """每个注册算法都有对应的模型目录"""
        for name in EXPECTED_ALGORITHMS:
            d = models_dir / name
            assert d.is_dir(), f"{name} 目录缺失: {d}"

    def test_key_files_exist(self, models_dir):
        """每个模型的关键文件都存在且非空"""
        for name, files in MODEL_KEY_FILES.items():
            for f in files:
                path = models_dir / name / f
                assert path.exists(), f"{name}/{f} 缺失"
                assert path.stat().st_size > 0, f"{name}/{f} 为空"

    def test_no_empty_dirs(self, models_dir):
        """没有空目录"""
        for d in models_dir.iterdir():
            if d.is_dir() and d.name not in ("modelscope_cache",):
                assert any(d.iterdir()), f"{d.name} 目录为空"

    def test_qwen3_safetensors_shards(self, models_dir):
        """Qwen3 的 safetensors 分片完整性"""
        shards = sorted((models_dir / "qwen3-asr").glob("model-*.safetensors"))
        assert len(shards) == 2, f"期望 2 个分片, 实际 {len(shards)}"
        idx_path = models_dir / "qwen3-asr" / "model.safetensors.index.json"
        with open(idx_path) as f:
            idx = json.load(f)
        expected_shards = set(idx["weight_map"].values())
        actual_shards = {s.name for s in shards}
        missing = expected_shards - actual_shards
        assert not missing, f"缺少分片: {missing}"

    def test_funasr_llm_zip_valid(self, models_dir):
        """funasr-llm/model.pt 是有效 ZIP"""
        import zipfile
        pt = models_dir / "funasr-llm" / "model.pt"
        with zipfile.ZipFile(pt) as z:
            bad = z.testzip()
            assert bad is None, f"ZIP 损坏: {bad}"
            assert len(z.namelist()) > 100, f"ZIP 条目过少: {len(z.namelist())}"


# ==================== 测试: 注册表 ====================

class TestASRRegistry:
    """验证 ASRRegistry 和算法描述"""

    @pytest.fixture(scope="class")
    def registry(self):
        from algorithms.asr.registry import ASRRegistry, ASR_ALGORITHM_DESCRIPTIONS
        return ASRRegistry, ASR_ALGORITHM_DESCRIPTIONS

    def test_all_algorithms_registered(self, registry):
        """所有 7 个算法已注册"""
        _, descs = registry
        registered = set(descs.keys())
        expected = set(EXPECTED_ALGORITHMS)
        assert registered == expected, f"差异: {registered ^ expected}"

    def test_each_description_has_required_fields(self, registry):
        """每个算法描述包含必填字段"""
        _, descs = registry
        required = {"display_name", "params", "cer_aishell1", "languages", "preload"}
        for name, desc in descs.items():
            missing = required - set(desc.keys())
            assert not missing, f"{name} 缺少字段: {missing}"
            assert isinstance(desc["preload"], bool), f"{name} preload 应为 bool"

    def test_preload_consistency(self, registry):
        """预加载标记与预期一致"""
        _, descs = registry
        preload = {n for n, d in descs.items() if d.get("preload")}
        assert preload == set(EXPECTED_PRELOAD), f"预加载不一致: {preload}"

    def test_list_available(self, registry):
        """list_available 返回正确结构"""
        ASRRegistry, _ = registry
        available = ASRRegistry.list_available()
        names = {a["name"] for a in available}
        assert names == set(EXPECTED_ALGORITHMS)


# ==================== 测试: 适配器初始化 (慢) ====================

class TestAdapterInitialization:
    """测试各适配器能否成功初始化（涉及模型加载，标记为 slow）"""

    @pytest.mark.slow
    @pytest.mark.parametrize("algo_name", EXPECTED_ALGORITHMS)
    def test_adapter_initialize_cpu(self, algo_name, project_root):
        """每个适配器能在 CPU 上初始化"""
        from algorithms.asr import _get_adapter_class
        from algorithms.asr.registry import ASR_ALGORITHM_DESCRIPTIONS

        cls = _get_adapter_class(algo_name)
        assert cls is not None, f"未找到适配器类: {algo_name}"

        model_dir = str(project_root / "models" / "asr" / algo_name)
        adapter = cls(device="cpu", model_dir=model_dir)
        ok = adapter.initialize()
        assert ok, f"{algo_name} CPU 初始化失败"

        assert adapter.is_initialized(), f"{algo_name} 初始化状态错误"

    @pytest.mark.slow
    def test_initialize_all_preloaded(self, project_root):
        """预加载的 3 个算法能全部成功初始化"""
        from algorithms.asr import register_all_asr_algorithms
        from algorithms.asr.registry import ASRRegistry, ASR_ALGORITHM_DESCRIPTIONS

        ASRRegistry._asrs.clear()
        ASRRegistry._instances.clear()
        register_all_asr_algorithms()
        ASRRegistry.initialize_all(model_dir=str(project_root / "models" / "asr"), device="cpu")

        initialized = ASRRegistry.list_initialized()
        for name in EXPECTED_PRELOAD:
            assert name in initialized, f"{name} 预加载失败"


# ==================== 测试: 转写 ====================

class TestTranscription:
    """测试各算法的转写功能（涉及模型推理，标记为 slow）"""

    ALGOS_FOR_TEST = ["wenet-u2pp"]  # 轻量级，速度快，适合测试

    @pytest.fixture(scope="class")
    def adapters(self, project_root):
        """按需初始化测试用适配器"""
        from algorithms.asr import _get_adapter_class

        adapters = {}
        for name in self.ALGOS_FOR_TEST:
            cls = _get_adapter_class(name)
            if cls is None:
                continue
            model_dir = str(project_root / "models" / "asr" / name)
            adapter = cls(device="cpu", model_dir=model_dir)
            assert adapter.initialize(), f"{name} 初始化失败"
            adapters[name] = adapter
        return adapters

    @pytest.mark.slow
    def test_transcribe_returns_result(self, adapters, ref_audio_numpy):
        """transcribe 返回 ASRResult 包含文本"""
        from algorithms.asr.base import ASRResult

        for name, adapter in adapters.items():
            audio, sr = ref_audio_numpy
            result = adapter.transcribe(audio, sample_rate=sr)
            assert isinstance(result, ASRResult), f"{name} 返回类型错误"
            assert isinstance(result.text, str), f"{name} text 应为 str"
            assert len(result.text) > 0, f"{name} 结果为空"

    @pytest.mark.slow
    def test_transcribe_accuracy(self, adapters, ref_audio_numpy):
        """转写结果匹配 ground truth（忽略标点差异）"""
        import re

        gt_clean = re.sub(r"\s+", "", GROUND_TRUTH)
        for name, adapter in adapters.items():
            audio, sr = ref_audio_numpy
            result = adapter.transcribe(audio, sample_rate=sr)
            text_clean = re.sub(r"\s+|[。，、！？：；""''「」]", "", result.text)
            assert text_clean == gt_clean, f"{name}: {text_clean} != {gt_clean}"

    @pytest.mark.slow
    def test_transcribe_returns_processing_time(self, adapters, ref_audio_numpy):
        """返回 processing_time >= 0"""
        for name, adapter in adapters.items():
            audio, sr = ref_audio_numpy
            result = adapter.transcribe(audio, sample_rate=sr)
            assert result.processing_time is None or result.processing_time >= 0

    @pytest.mark.slow
    @pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100])
    def test_transcribe_various_sample_rates(self, adapters, sr):
        """支持不同采样率的输入"""
        import numpy as np
        duration = 1.0
        # 生成纯音（非静音，避免模型返回空）
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        for name, adapter in adapters.items():
            result = adapter.transcribe(audio, sample_rate=sr)
            assert result.text is not None


# ==================== 测试: 底噪输入 ====================

class TestEdgeCases:
    """边界和异常输入测试"""

    @pytest.mark.slow
    def test_empty_audio_raises_error(self, project_root):
        """空音频应报错而非崩溃"""
        from algorithms.asr import _get_adapter_class
        cls = _get_adapter_class("wenet-u2pp")
        model_dir = str(project_root / "models" / "asr" / "wenet-u2pp")
        adapter = cls(device="cpu", model_dir=model_dir)
        adapter.initialize()

        with pytest.raises(Exception):
            adapter.transcribe(np.array([], dtype=np.float32), sample_rate=16000)

    def test_invalid_model_dir(self, project_root):
        """无效模型目录不应崩溃"""
        from algorithms.asr import _get_adapter_class
        cls = _get_adapter_class("wenet-u2pp")
        adapter = cls(device="cpu", model_dir="/nonexistent/path")
        # 可能初始化失败（返回 False）或尝试在线下载，但不应崩溃
        adapter.initialize()
        assert True


# ==================== 测试: ASR 指标计算 ====================

class TestASRMetrics:
    """评估指标计算"""

    def test_cer_perfect_match(self):
        """完全匹配时 CER = 0"""
        from algorithms.asr.evaluator import compute_cer
        text = "测试文本"
        cer, ins, dele, subs = compute_cer(text, text)
        assert cer == 0.0
        assert ins == 0
        assert dele == 0

    def test_cer_completely_wrong(self):
        """完全不匹配时 CER > 0"""
        from algorithms.asr.evaluator import compute_cer
        cer, ins, dele, subs = compute_cer("abc", "xyz")
        assert cer > 0

    def test_wer_calculation(self):
        """WER 计算正确"""
        from algorithms.asr.evaluator import compute_wer
        ref = "hello world foo bar"
        hyp = "hello world bar"
        wer, ins, dele, subs = compute_wer(ref, hyp)
        assert wer == 0.25  # 4 个词错 1 个

    def test_asr_metrics_integration(self):
        """综合指标计算"""
        from algorithms.asr.evaluator import evaluate_asr
        refs = ["测试文本一", "测试文本二"]
        hyps = ["测试文本一", "测试文本二"]
        metrics = evaluate_asr(refs, hyps)
        assert metrics.cer == 0.0
        assert metrics.wer == 0.0
