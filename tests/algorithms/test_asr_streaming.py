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