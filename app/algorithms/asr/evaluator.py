"""
ASR评测指标模块
提供CER/WER/RTF等ASR评测指标计算
"""

import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("audiomos")


@dataclass
class ASRMetrics:
    """ASR评测指标"""
    # 核心指标
    cer: Optional[float] = None       # 字错误率 (Character Error Rate)
    wer: Optional[float] = None       # 词错误率 (Word Error Rate)

    # 错误细分
    cer_del: Optional[float] = None   # 删除错误率
    cer_ins: Optional[float] = None   # 插入错误率
    cer_sub: Optional[float] = None   # 替换错误率
    wer_del: Optional[float] = None
    wer_ins: Optional[float] = None
    wer_sub: Optional[float] = None

    # 效率指标
    processing_time: float = 0.0      # 处理时间(秒)
    rtf: Optional[float] = None       # 实时因子
    audio_duration: float = 0.0       # 音频时长(秒)

    # 流式指标
    first_token_latency: Optional[float] = None  # 首字延迟(秒)

    # 统计
    num_utterances: int = 0           # 评测句子数
    num_chars: int = 0                # 总字符数
    num_words: int = 0                # 总词数

    def to_dict(self) -> dict:
        return {
            "cer": self.cer,
            "wer": self.wer,
            "cer_detail": {
                "delete": self.cer_del,
                "insert": self.cer_ins,
                "substitute": self.cer_sub,
            },
            "wer_detail": {
                "delete": self.wer_del,
                "insert": self.wer_ins,
                "substitute": self.wer_sub,
            },
            "processing_time": round(self.processing_time, 3),
            "rtf": round(self.rtf, 3) if self.rtf else None,
            "audio_duration": round(self.audio_duration, 3),
            "first_token_latency": self.first_token_latency,
            "num_utterances": self.num_utterances,
            "num_chars": self.num_chars,
        }


def _edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    计算编辑距离，返回(删除, 插入, 替换, 正确)数

    Args:
        ref: 参考序列
        hyp: 假设序列

    Returns:
        (deletions, insertions, substitutions, correct)
    """
    n = len(ref)
    m = len(hyp)

    # DP矩阵
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # 回溯获取操作类型
    i, j = n, m
    dels, ins, subs, correct = 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            correct += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    return dels, ins, subs, correct


def compute_cer(reference: str, hypothesis: str) -> Tuple[float, float, float, float]:
    """
    计算中文字错误率 (Character Error Rate)

    中文按字符级别拆分，标点符号过滤

    Args:
        reference: 参考文本
        hypothesis: 识别文本

    Returns:
        (cer, cer_del, cer_ins, cer_sub)
    """
    # 过滤标点和空白
    import re
    ref_clean = re.sub(r'[，。！？、；：\u201c\u201d\u2018\u2019\uff08\uff09\s,.!?;:\'\"()\-\u2014\u2026]', '', reference)
    hyp_clean = re.sub(r'[，。！？、；：\u201c\u201d\u2018\u2019\uff08\uff09\s,.!?;:\'\"()\-\u2014\u2026]', '', hypothesis)

    ref_chars = list(ref_clean)
    hyp_chars = list(hyp_clean)

    if len(ref_chars) == 0:
        return 0.0, 0.0, 0.0, 0.0

    dels, ins, subs, _ = _edit_distance(ref_chars, hyp_chars)
    total = len(ref_chars)
    cer = (dels + ins + subs) / total
    cer_del = dels / total
    cer_ins = ins / total
    cer_sub = subs / total

    return cer, cer_del, cer_ins, cer_sub


def compute_wer(reference: str, hypothesis: str) -> Tuple[float, float, float, float]:
    """
    计算词错误率 (Word Error Rate)

    中文使用 jieba 分词，英文按空格分词

    Args:
        reference: 参考文本
        hypothesis: 识别文本

    Returns:
        (wer, wer_del, wer_ins, wer_sub)
    """
    ref_words = _tokenize_words(reference)
    hyp_words = _tokenize_words(hypothesis)

    if len(ref_words) == 0:
        return 0.0, 0.0, 0.0, 0.0

    dels, ins, subs, _ = _edit_distance(ref_words, hyp_words)
    total = len(ref_words)
    wer = (dels + ins + subs) / total
    wer_del = dels / total
    wer_ins = ins / total
    wer_sub = subs / total

    return wer, wer_del, wer_ins, wer_sub


def _tokenize_words(text: str) -> List[str]:
    """对文本分词：中文用jieba，英文/已分好词用split"""
    # 去除标点
    clean = re.sub(r'[，。！？、；：“”‘’（）\s,.!?;:\'\"()\-—…]', '', text)
    if not clean:
        return []
    # 检测是否含中文字符
    if re.search(r'[一-鿿]', clean):
        import jieba
        return list(jieba.cut(clean))
    else:
        return clean.split()


def evaluate_asr(
    references: List[str],
    hypotheses: List[str],
    processing_times: Optional[List[float]] = None,
    audio_durations: Optional[List[float]] = None,
) -> ASRMetrics:
    """
    批量评测ASR结果

    Args:
        references: 参考文本列表
        hypotheses: 识别文本列表
        processing_times: 各条处理时间
        audio_durations: 各条音频时长

    Returns:
        ASRMetrics
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"参考和假设数量不匹配: {len(references)} vs {len(hypotheses)}")

    metrics = ASRMetrics()
    total_cer, total_cer_del, total_cer_ins, total_cer_sub = 0, 0, 0, 0
    total_wer, total_wer_del, total_wer_ins, total_wer_sub = 0, 0, 0, 0
    total_chars = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        cer, cer_del, cer_ins, cer_sub = compute_cer(ref, hyp)
        total_cer += cer
        total_cer_del += cer_del
        total_cer_ins += cer_ins
        total_cer_sub += cer_sub

        import re
        ref_clean = re.sub(r'[，。！？、；：\u201c\u201d\u2018\u2019\uff08\uff09\s,.!?;:\'\"()\-\u2014\u2026]', '', ref)
        total_chars += len(ref_clean)

        wer, wer_del, wer_ins, wer_sub = compute_wer(ref, hyp)
        total_wer += wer
        total_wer_del += wer_del
        total_wer_ins += wer_ins
        total_wer_sub += wer_sub
        total_words += len(_tokenize_words(ref))

    n = len(references)
    metrics.num_utterances = n
    metrics.num_chars = total_chars
    metrics.num_words = total_words

    metrics.cer = total_cer / n if n > 0 else 0
    metrics.cer_del = total_cer_del / n if n > 0 else 0
    metrics.cer_ins = total_cer_ins / n if n > 0 else 0
    metrics.cer_sub = total_cer_sub / n if n > 0 else 0

    metrics.wer = total_wer / n if n > 0 else 0
    metrics.wer_del = total_wer_del / n if n > 0 else 0
    metrics.wer_ins = total_wer_ins / n if n > 0 else 0
    metrics.wer_sub = total_wer_sub / n if n > 0 else 0

    if processing_times:
        metrics.processing_time = sum(processing_times)
    if audio_durations:
        metrics.audio_duration = sum(audio_durations)
        if metrics.processing_time > 0:
            metrics.rtf = metrics.processing_time / metrics.audio_duration

    return metrics
