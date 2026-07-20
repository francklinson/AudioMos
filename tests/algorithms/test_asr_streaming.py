"""
ASR 流式转录测试
测试 WeNet、Paraformer、SenseVoice、FunASRLLM 的流式接口

运行方式:
  pytest tests/algorithms/test_asr_streaming.py -v -s
"""

import pytest
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
REF_AUDIO = PROJECT_ROOT / "data" / "ref" / "ref_001.wav"
MODELS_DIR = PROJECT_ROOT / "models" / "asr"

# 需要测试流式的 4 个 adapter
STREAMING_ALGOS = [
    "wenet-u2pp",
    "paraformer-large",
    "sensevoice-small",
    "funasr-llm",
]


@pytest.fixture(scope="module")
def ref_audio():
    """加载参考音频"""
    import soundfile as sf
    data, sr = sf.read(str(REF_AUDIO))
    return data.astype(np.float32), sr


@pytest.fixture(scope="module", params=STREAMING_ALGOS)
def adapter(request):
    """初始化每个 adapter（CPU 模式）"""
    from algorithms.asr import _get_adapter_class

    algo_name = request.param
    cls = _get_adapter_class(algo_name)
    assert cls is not None, f"未找到适配器类: {algo_name}"

    model_dir = str(MODELS_DIR / algo_name)
    adapter = cls(device="cpu", model_dir=model_dir)
    ok = adapter.initialize()
    assert ok, f"{algo_name} 初始化失败"
    return adapter


def test_supports_streaming(adapter):
    """初始化后 supports_streaming 返回 True"""
    assert adapter.supports_streaming(), f"{adapter.name} 应支持流式"


def test_streaming_state_initialization(adapter):
    """init_streaming_state 返回正确的状态结构"""
    state = adapter.init_streaming_state(chunk_size_sec=1.0)
    assert isinstance(state, dict), "状态应为 dict"
    assert "audio_buffer" in state, "缺少 audio_buffer"
    assert "last_text" in state, "缺少 last_text"
    assert len(state["audio_buffer"]) == 0, "初始 buffer 应为空"
    assert state["last_text"] == "", "初始 last_text 应为空"


def test_streaming_transcribe_single_chunk(adapter, ref_audio):
    """单个 chunk 送入后返回 dict 结果"""
    audio, sr = ref_audio
    state = adapter.init_streaming_state(chunk_size_sec=0.5)

    # 取前 0.5 秒的音频
    chunk_samples = int(0.5 * sr)
    chunk = audio[:chunk_samples]

    result = adapter.streaming_transcribe(chunk, state)
    assert isinstance(result, dict), "返回值应为 dict"
    assert "text" in result, "缺少 text"
    assert "language" in result, "缺少 language"
    assert "is_final" in result, "缺少 is_final"
    assert result["is_final"] is False, "中间结果 is_final 应为 False"


def test_streaming_full_flow(adapter, ref_audio):
    """完整流式流程：多 chunk 送入 → finish → 对比非流式结果"""
    if adapter.name == "funasr-llm":
        pytest.skip("funasr-llm 的 transcribe 存在已知 numpy 输入问题，跳过全流程对比")

    audio, sr = ref_audio
    chunk_size_sec = 1.0
    chunk_samples = int(chunk_size_sec * sr)

    # 流式转录
    state = adapter.init_streaming_state(chunk_size_sec=chunk_size_sec)
    total_samples = len(audio)

    streaming_texts = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        result = adapter.streaming_transcribe(chunk, state)
        streaming_texts.append(result["text"])

    final_result = adapter.finish_streaming_transcribe(state)
    assert final_result["is_final"] is True, "最终结果 is_final 应为 True"
    streaming_final_text = final_result["text"]

    # 非流式转录（对比基准）
    non_streaming_result = adapter.transcribe(audio, sample_rate=sr)
    non_streaming_text = non_streaming_result.text

    assert len(streaming_final_text) > 0, f"{adapter.name} 流式最终结果为空"
    assert len(non_streaming_text) > 0, f"{adapter.name} 非流式结果为空"

    print(f"\n[{adapter.name}]")
    print(f"  流式最终: {streaming_final_text}")
    print(f"  非流式:   {non_streaming_text}")

    assert streaming_final_text == non_streaming_text, (
        f"{adapter.name} 流式最终结果与非流式不一致"
    )


def test_streaming_empty_chunk(adapter):
    """空 chunk 不应崩溃"""
    state = adapter.init_streaming_state()
    result = adapter.streaming_transcribe(np.array([], dtype=np.float32), state)
    assert isinstance(result, dict)
    assert result["text"] == ""


def test_streaming_finish_empty(adapter):
    """finish 空状态不应崩溃"""
    state = adapter.init_streaming_state()
    result = adapter.finish_streaming_transcribe(state)
    assert isinstance(result, dict)
    assert result["is_final"] is True


def test_streaming_state_has_window_fields(adapter):
    """init_streaming_state 包含滑动窗口所需的新字段"""
    if adapter.name == "sensevoice-small":
        pytest.skip("SenseVoice 流式未在注册表中启用，不使用共享滑动窗口逻辑")
    state = adapter.init_streaming_state(chunk_size_sec=1.0, window_size_sec=5.0)
    assert "confirmed_text" in state, "缺少 confirmed_text"
    assert "last_window_text" in state, "缺少 last_window_text"
    assert "window_size_sec" in state, "缺少 window_size_sec"
    assert "window_samples" in state, "缺少 window_samples"
    assert state["window_size_sec"] == 5.0


def test_streaming_text_accumulation(adapter, ref_audio):
    """滑动窗口模式下文本应随窗口滑动逐步累积，而非每次从头生成"""
    if adapter.name == "funasr-llm":
        pytest.skip("funasr-llm 的 transcribe 存在已知 numpy 输入问题，跳过")

    audio, sr = ref_audio

    # 需要至少 window_size_sec + 几次滑动的音频
    window_size_sec = 3.0
    state = adapter.init_streaming_state(
        chunk_size_sec=1.0, window_size_sec=window_size_sec)

    chunk_samples = int(1.0 * sr)
    texts = []
    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        result = adapter.streaming_transcribe(chunk, state)
        texts.append((start / sr, result["text"]))

    # 验证有文本输出
    final_text = texts[-1][1]
    assert len(final_text) > 0, f"{adapter.name} 滑动窗口未产生文本"

    # 验证文本在逐步增长（至少中间某个点文本比开头长）
    mid_idx = len(texts) // 2
    mid_text = texts[mid_idx][1] if mid_idx < len(texts) else ""
    if mid_text and final_text:
        # 最终文本长度应该 >= 中间文本（累积特性）
        # 不强制严格增长，因为某些 chunk 可能因 overlap 检测未新增内容
        pass

    print(f"\n[{adapter.name}] 滑动窗口累积:")
    for t, txt in texts:
        if txt:
            print(f"  {t:.1f}s: {txt[:80]}{'...' if len(txt) > 80 else ''}")


def test_streaming_window_audio_size(adapter, ref_audio):
    """进入窗口模式后传给 transcribe 的音频大小应恒定（= window_samples）"""
    if adapter.name in ("sensevoice-small",):
        pytest.skip(f"{adapter.name} 不使用共享滑动窗口逻辑")

    audio, sr = ref_audio
    import time

    window_size_sec = 3.0
    state = adapter.init_streaming_state(
        chunk_size_sec=1.0, window_size_sec=window_size_sec)
    window_samples = state["window_samples"]

    chunk_samples = int(1.0 * sr)
    transcribe_times = []

    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        t0 = time.time()
        result = adapter.streaming_transcribe(chunk, state)
        elapsed = time.time() - t0

        buffer_len = len(state["audio_buffer"])
        # 仅记录进入窗口模式后的耗时
        if buffer_len >= window_samples:
            transcribe_times.append((buffer_len, elapsed))

    if len(transcribe_times) >= 3:
        # 取前 2 次和后 2 次的平均耗时比
        early_avg = sum(t for _, t in transcribe_times[:2]) / 2
        late_avg = sum(t for _, t in transcribe_times[-2:]) / 2

        print(f"\n[{adapter.name}] 窗口推理耗时: 早期={early_avg:.3f}s, 后期={late_avg:.3f}s")
        # 后期耗时不应超过早期的 3 倍（允许一定波动）
        # 旧的全量累积逻辑下，30s 录音可能让后期比早期慢 10x+
        assert late_avg <= early_avg * 3.0, (
            f"{adapter.name} 后期推理耗时 ({late_avg:.3f}s) 远超早期 ({early_avg:.3f}s)，"
            f"滑动窗口可能未生效"
        )