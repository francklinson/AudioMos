"""
WeNet ASR语义匹配 + 文本DTW定位模块

利用WeNet的CTC字级时间戳和参考音频的ground truth文本，通过文本DTW实现
快速的语义级别音频段定位，替代原有的全范围声学DTW扫描。

架构位置:
  作为 OptimizedMatcher.match_with_fallback() 的 Level A（主策略）
  原全范围声学DTW降级为 Level B（回退策略）

流程:
  1. 测试音频 → WeNet ASR → 字符序列 + 字级时间戳 (40ms/帧)
  2. 参考段ground truth文本 → 文本DTW匹配到测试转录
  3. DTW对齐位置 → 映射时间戳 → 段边界（秒级精度）
  4. [可选] 窄范围声学DTW验证 ±1s（秒→帧级）
  5. [可沿用] HPSS谐波互相关精对齐（帧→样本级）
  6. [副产品] WER在匹配过程中自动获得

依赖:
  - wenet 模型（通过 wenet.cli.model.load_model 加载）
  - 参考音频目录中的 .metadata.json（含 ground_truth_text 字段）
  - dtw-python（文本DTW的后端，轻量级依赖）
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field

import numpy as np
import threading
from typing import Dict, Tuple, Optional

logger = logging.getLogger('audiomos')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class TranscriptionResult:
    """WeNet ASR转写结果（含字符级时间戳）"""
    chars: List[str]           # 字符序列 ["他", "为", "儿", ...]
    timestamps: List[float]    # 每个字符的起始时间（秒）
    raw_text: str              # 原始文本
    tokens: List[int] = field(default_factory=list)   # token ID序列
    confidence: float = 0.0    # 平均置信度


@dataclass
class SegmentMatch:
    """单个参考段的匹配结果"""
    ref_name: str        # 参考音频文件名（如 ref_001.wav）
    ref_file: str        # 参考音频完整路径
    offset: float        # 在测试音频中的起始偏移（秒）
    duration: float      # 段时长（秒）
    confidence: float    # 匹配置信度 0-1
    n_chars: int         # 匹配的字符数
    wer: float           # WER（副产品）
    wcorr: float         # 字正确率（副产品）
    method: str          # 匹配方法标识


@dataclass
class RefTextInfo:
    """参考音频的文本信息"""
    ref_name: str        # 文件名
    ref_file: str        # 完整路径
    text: str            # ground truth 文本
    chars: List[str]     # 字符序列
    head_silence: float = 0.0  # 段首静音时长（秒）
    tail_silence: float = 0.0  # 段尾静音时长（秒）
    ref_first_char_time: float = 0.0  # 参考音频自身的首字时间戳（秒），用于精确修正偏移


# ============================================================================
# 全局WER缓存（避免评分阶段重复ASR计算）
# ============================================================================

_wer_cache = {}  # {ref_name: (wer, wcorr)}
_wer_cache_lock = threading.Lock()

def set_wer_cache(wers: Dict[str, Tuple[float, float]]):
    """设置WER缓存（由预检测阶段写入）"""
    with _wer_cache_lock:
        _wer_cache.update(wers)

def get_wer_from_cache(ref_name: str) -> Optional[Tuple[float, float]]:
    """从缓存获取WER值"""
    with _wer_cache_lock:
        return _wer_cache.get(ref_name)

def clear_wer_cache():
    """清空WER缓存"""
    with _wer_cache_lock:
        _wer_cache.clear()
# ============================================================================

class TextDTWAligner:
    """
    字符级文本DTW对齐器
    在参考文本和ASR转写文本之间做DTW对齐，容忍ASR的插入/删除/替换错误

    复杂度: O(N*M), N=ref字符数(~37), M=test字符数(~150)
    相比于声学DTW的O(S*T*F): ~100 vs ~146250，快~1500倍
    """

    @staticmethod
    def _char_cost(a: str, b: str) -> float:
        """字符距离：相同为0，不同为1"""
        return 0.0 if a == b else 1.0

    @staticmethod
    def _text_to_chars(text: str) -> List[str]:
        """将文本转为字符列表（中文逐字，英文逐字母）"""
        return list(text)

    def align(self, ref_text: str, test_chars: List[str],
              max_offset: int = None) -> Optional[Dict]:
        """
        文本DTW对齐：找到参考文本在测试转写中的最佳位置

        使用子序列DTW（Subsequence DTW）算法：
        - d[0, j] = 0 允许对齐从测试序列的任意位置开始
        - d[i, 0] = INF 防止参考文本的起始字符被删除
        - 在最后一行 d[N, :] 中找全局最小值作为最佳终点
        - 从最佳终点回溯到起点

        Args:
            ref_text: 参考文本（ground truth）
            test_chars: 测试转写的字符序列
            max_offset: 最大允许偏移（字符数），None=不限

        Returns:
            {
                "start_pos": int,      # 在test_chars中的起始位置
                "end_pos": int,        # 在test_chars中的结束位置
                "distance": float,     # 归一化DTW距离
                "wer": float,          # 词错误率
                "wcorr": float,        # 字正确率
                "path": List[Tuple],   # 对齐路径 (ref_idx, test_idx)
                "n_match": int,        # 匹配字符数
                "n_sub": int,          # 替换字符数
                "n_ins": int,          # 插入字符数
                "n_del": int           # 删除字符数
            }
        """
        ref_chars = self._text_to_chars(ref_text)
        N = len(ref_chars)
        M = len(test_chars)

        if N == 0 or M == 0:
            return None

        # 测试文本太短 -> ASR漏识别严重
        if M < N * 0.5:
            return None

        INF = float('inf')

        # ---------- 子序列DTW DP ----------
        # d[i][j] = ref[:i] 对齐到 test[:j] 的最小累积距离
        # d[0][j] = 0 允许从test任意位置开始匹配（子序列DTW核心）
        # d[i][0] = INF i>0 不允许删除参考起始字符
        d = np.full((N + 1, M + 1), INF, dtype=np.float64)
        d[0, :] = 0.0  # 子序列DTW初始化：可从任意位置开始

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                cost = self._char_cost(ref_chars[i - 1], test_chars[j - 1])
                d[i, j] = min(
                    d[i - 1, j - 1] + cost,     # 匹配/替换
                    d[i, j - 1] + 0.5,           # 插入 test字符（参考无对应）
                    d[i - 1, j] + 0.5             # 删除 ref字符（测试无对应）
                )

        # ---------- 找到最佳终点 ----------
        # 从最后一行 d[N, :] 找最小值对应的测试位置
        # 只在 d[N, N-5:M+1] 范围内搜索（排除太短的匹配）
        search_start = max(N - 1, 0)
        best_end_j = int(np.argmin(d[N, search_start:])) + search_start
        best_end_dist = float(d[N, best_end_j])

        # 归一化距离
        norm_dist = best_end_dist / max(1, N)

        # 距离阈值：平均每个字符>0.8说明匹配很差
        if norm_dist > 0.8:
            logger.debug(f"[文本DTW] 距离过大: {norm_dist:.3f} > 0.8, 拒绝匹配")
            return None

        # ---------- 回溯对齐路径 ----------
        path = []
        i, j = N, best_end_j
        while i > 0 and j > 0:
            cost = self._char_cost(ref_chars[i - 1], test_chars[j - 1])
            path.append((i - 1, j - 1))

            if d[i, j] == d[i - 1, j - 1] + cost:
                i -= 1
                j -= 1
            elif d[i, j] == d[i, j - 1] + 0.5:
                j -= 1
            else:  # d[i][j] == d[i-1][j] + 0.5
                i -= 1

        path.reverse()

        # ---- 从回溯路径提取边界 ----
        # 路径的第一对点的 test_idx 是匹配的起点
        start_pos = path[0][1] if path else 0
        end_pos = path[-1][1] if path else best_end_j

        # 安全检查
        if start_pos >= M or end_pos < start_pos or end_pos >= M:
            return None

        # ---------- 从路径统计编辑操作 ----------
        n_match = n_sub = n_ins = n_del = 0
        prev_ri, prev_tj = -1, -1
        for ri, tj in path:
            if prev_ri >= 0:
                d_ri = ri - prev_ri  # ref 步进
                d_tj = tj - prev_tj  # test 步进
                if d_ri == 1 and d_tj == 1:
                    if ri < N and tj < M and ref_chars[ri] == test_chars[tj]:
                        n_match += 1
                    else:
                        n_sub += 1
                elif d_ri == 0 and d_tj == 1:
                    n_ins += 1
                elif d_ri == 1 and d_tj == 0:
                    n_del += 1
                # d_ri > 1 或 d_tj > 1: 连续处理（很少见）
            else:
                # 第一个路径点
                if ri < N and tj < M and ref_chars[ri] == test_chars[tj]:
                    n_match += 1
                else:
                    n_sub += 1
            prev_ri, prev_tj = ri, tj

        # WER = (替换+插入+删除) / 参考长度
        wer = (n_sub + n_ins + n_del) / max(1, N)
        wcorr = max(0, (N - n_sub - n_del) / max(1, N))

        # 置信度
        match_ratio = n_match / max(1, N)
        confidence = match_ratio * (1.0 - min(norm_dist / 0.8, 1.0))
        confidence = max(0.0, min(1.0, confidence))

        seg_text = "".join(test_chars[start_pos:end_pos + 1])[:20]
        logger.debug(
            f"[文本DTW] ref={ref_text[:15]} => seg={seg_text} "
            f"[{start_pos}:{end_pos}] match={n_match} sub={n_sub} "
            f"ins={n_ins} del={n_del} conf={confidence:.3f} WER={wer:.3f}"
        )

        return {
            "start_pos": start_pos,
            "end_pos": end_pos,
            "distance": float(norm_dist),
            "wer": float(wer),
            "wcorr": float(wcorr),
            "path": path,
            "n_match": n_match,
            "n_sub": n_sub,
            "n_ins": n_ins,
            "n_del": n_del,
            "confidence": float(confidence),
        }


# ============================================================================
# WeNet ASR语义匹配器
# ============================================================================

class WeNetSegmentationMatcher:
    """
    基于WeNet ASR + 文本DTW的语义匹配器

    使用方式:
        matcher = WeNetSegmentationMatcher()
        matcher.load_model(model_dir="models/wenet")
        matcher.load_ref_texts("data/ref")

        # 单文件匹配
        result = matcher.match_with_fallback(test_audio_path)

        # 批量匹配
        results = matcher.batch_match(test_file_list)

    与现有OptimizedMatcher兼容:
        match_with_fallback() 返回格式与 OptimizedMatcher.match_with_fallback() 一致
    """

    def __init__(self, ref_dir: str = None):
        self.ref_dir = ref_dir
        self._model = None
        self._tokenizer = None
        self._frame_shift = 0.04  # 40ms (subsampling 4 × 10ms hop)
        self._text_aligner = TextDTWAligner()
        self._ref_texts: Dict[str, RefTextInfo] = {}  # ref_name -> info
        self._initialized = False

    # ---- 模型管理 ----

    def load_model(self, model_dir: str = None, device: str = 'cuda') -> bool:
        """加载WeNet模型

        Args:
            model_dir: 模型目录，默认 models/wenet
            device: 运行设备 'cuda' 或 'cpu'
        """
        if self._model is not None:
            return True

        target_dir = model_dir or os.path.join(_PROJECT_ROOT, 'models', 'wenet')
        if not os.path.exists(target_dir):
            logger.error(f"[WeNet匹配] 模型目录不存在: {target_dir}")
            return False

        try:
            sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'app', 'algorithms', 'wenet'))
            from wenet.cli.model import load_model as _load_wenet

            logger.info(f"[WeNet匹配] 加载模型: {target_dir} (device={device})")
            start = time.time()
            self._model = _load_wenet(target_dir, device=device)
            self._frame_shift = 0.01 * self._model.subsampling_rate()  # 4 * 10ms
            elapsed = time.time() - start
            logger.info(f"[WeNet匹配] 模型加载完成 (subsampling={self._model.subsampling_rate()}, "
                        f"frame_shift={self._frame_shift*1000:.0f}ms, 耗时={elapsed:.2f}s)")
            return True
        except Exception as e:
            logger.error(f"[WeNet匹配] 模型加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def set_model(self, model) -> None:
        """注入已加载的WeNet模型（避免重复加载）"""
        self._model = model
        if hasattr(model, 'subsampling_rate'):
            self._frame_shift = 0.01 * model.subsampling_rate()
        # 如果参考文本已加载，补算首字时间戳
        if self._ref_texts:
            self._compute_ref_first_char_times()

    def is_model_ready(self) -> bool:
        return self._model is not None

    # ---- 参考文本管理 ----

    def load_ref_texts(self, ref_dir: str = None) -> int:
        """从参考音频目录加载ground truth文本，并检测段首/段尾静音时长

        Args:
            ref_dir: 参考音频目录

        Returns:
            成功加载的文本数
        """
        target_dir = ref_dir or self.ref_dir
        if not target_dir or not os.path.isdir(target_dir):
            logger.warning(f"[WeNet匹配] 参考目录不存在: {target_dir}")
            return 0

        # 读取 metadata.json
        metadata_file = os.path.join(target_dir, '.metadata.json')
        if not os.path.exists(metadata_file):
            logger.warning(f"[WeNet匹配] 元数据文件不存在: {metadata_file}")
            return 0

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            count = 0
            audios = metadata.get("audios", {})
            for audio_id, info in audios.items():
                filename = info.get("filename", "")
                gt_text = info.get("ground_truth_text", "")
                if not filename or not gt_text:
                    continue
                filepath = os.path.join(target_dir, filename)
                if not os.path.exists(filepath):
                    logger.warning(f"[WeNet匹配] 文件不存在（metadata中有但磁盘无）: {filepath}")
                    continue

                # 检测段首/段尾静音时长（用于修正ASR时间戳偏移）
                head_sil, tail_sil = self._detect_silence(filepath)

                self._ref_texts[filename] = RefTextInfo(
                    ref_name=filename,
                    ref_file=filepath,
                    text=gt_text,
                    chars=list(gt_text),
                    head_silence=head_sil,
                    tail_silence=tail_sil
                )
                count += 1
                logger.info(f"[WeNet匹配] 加载参考文本: {filename} ({len(gt_text)}字, "
                           f"head_sil={head_sil:.3f}s, tail_sil={tail_sil:.3f}s)")

            logger.info(f"[WeNet匹配] 共加载 {count}/{len(audios)} 个参考文本")
            self._initialized = count > 0

            # 如果模型已加载，计算参考音频自身的首字时间戳（用于精确偏移修正）
            if self._model is not None and count > 0:
                self._compute_ref_first_char_times()

            return count
        except Exception as e:
            logger.error(f"[WeNet匹配] 加载元数据失败: {e}")
            return 0

    def _compute_ref_first_char_times(self):
        """计算每个参考音频自身的ASR首字时间戳

        原理：
          ASR返回的offset是测试音频中首字的CTC峰值时间。
          要得到参考段在测试音频中的真实起始位置，需要：
            real_offset = test_first_char_time - ref_first_char_time
          其中ref_first_char_time是首字在参考音频自身中的时间戳。

          这比单纯减去段首静音更精确，因为CTC峰值偏移也被考虑在内。
        """
        if not self._model:
            return
        for name, info in self._ref_texts.items():
            try:
                ref_path = info.ref_file
                if not os.path.exists(ref_path):
                    continue
                result = self._model.transcribe(ref_path)
                raw_times = result.times if hasattr(result, 'times') else None
                if raw_times and len(raw_times) > 0:
                    first_char_time = raw_times[0] * self._frame_shift
                    info.ref_first_char_time = first_char_time
                    logger.debug(f"[参考首字时间] {name}: first_char={first_char_time:.3f}s "
                                f"(head_sil={info.head_silence:.3f}s)")
            except Exception as e:
                logger.debug(f"[参考首字时间] 计算失败 {name}: {e}")
                info.ref_first_char_time = info.head_silence  # fallback

        times = [info.ref_first_char_time for info in self._ref_texts.values()
                if info.ref_first_char_time > 0]
        if times:
            avg = np.mean(times)
            logger.info(f"[参考首字时间] 完成，平均={avg:.3f}s")

    @staticmethod
    def _detect_silence(wav_path: str, sr: int = 16000,
                         threshold: float = 0.02,
                         frame_ms: int = 10) -> Tuple[float, float]:
        """检测音频的段首和段尾静音时长

        使用短时能量检测：
        - 分帧（10ms/帧）
        - 能量低于全局最大能量的 threshold 视为静音
        - 从头部扫描到第一个非静音帧 = head_silence
        - 从尾部扫描到最后一个非静音帧 = tail_silence

        Args:
            wav_path: 音频文件路径
            sr: 目标采样率
            threshold: 静音能量阈值比例（相对于全局最大能量）
            frame_ms: 帧长（毫秒）

        Returns:
            (head_silence_seconds, tail_silence_seconds)
        """
        try:
            import librosa
            audio, _ = librosa.load(wav_path, sr=sr, mono=True)
            frame_len = int(sr * frame_ms / 1000)
            hop = frame_len // 2

            # 分帧能量
            energy = np.array([
                np.sum(audio[i:i + frame_len] ** 2)
                for i in range(0, len(audio) - frame_len, hop)
            ])
            if len(energy) == 0:
                return 0.0, 0.0

            energy_thresh = np.max(energy) * threshold

            # 头部静音
            head_frames = 0
            for e in energy:
                if e < energy_thresh:
                    head_frames += 1
                else:
                    break
            head_sil = head_frames * hop / sr

            # 尾部静音
            tail_frames = 0
            for e in reversed(energy):
                if e < energy_thresh:
                    tail_frames += 1
                else:
                    break
            tail_sil = tail_frames * hop / sr

            return head_sil, tail_sil
        except Exception as e:
            logger.debug(f"[静音检测] 失败 {wav_path}: {e}")
            return 0.0, 0.0

    def get_ref_texts(self) -> Dict[str, RefTextInfo]:
        """获取已加载的参考文本"""
        return dict(self._ref_texts)

    # ---- ASR转写 ----

    def transcribe(self, audio_path: str) -> Optional[TranscriptionResult]:
        """
        对音频做ASR转写，返回字符级时间戳

        Args:
            audio_path: 音频文件路径

        Returns:
            TranscriptionResult 或 None（转写失败时）
        """
        if not self._model:
            logger.error("[WeNet匹配] 模型未加载")
            return None

        if not os.path.exists(audio_path):
            logger.error(f"[WeNet匹配] 音频不存在: {audio_path}")
            return None

        try:
            result = self._model.transcribe(audio_path)

            # 提取字符序列
            chars = self._model.tokenizer.detokenize(result.tokens)
            if isinstance(chars, tuple):
                chars = chars[0] if chars else ""
            chars_list = list(chars) if chars else []

            # 提取时间戳
            raw_timestamps = result.times if hasattr(result, 'times') else None
            if raw_timestamps and len(raw_timestamps) == len(chars_list):
                timestamps = [t * self._frame_shift for t in raw_timestamps]
            else:
                # 无时间戳或长度不匹配
                timestamps = []

            # 提取置信度
            confidence = getattr(result, 'confidence', 0.0)

            # 获取原始文本
            raw_text = result.text if hasattr(result, 'text') else chars

            logger.debug(f"[WeNet转写] {os.path.basename(audio_path)}: "
                        f"{len(chars_list)}字, {len(timestamps)}时间戳, "
                        f"conf={confidence:.3f}")

            return TranscriptionResult(
                chars=chars_list,
                timestamps=timestamps,
                raw_text=raw_text,
                tokens=result.tokens,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[WeNet转写] ASR失败 {audio_path}: {e}")
            return None

    # ---- 文本DTW匹配 ----

    def match_segments(self, test_transcription: TranscriptionResult,
                       ref_texts: List[RefTextInfo] = None,
                       min_confidence: float = 0.15) -> List[SegmentMatch]:
        """
        用文本DTW匹配每个参考段在测试转写中的位置

        策略：
        1. 对所有参考段执行文本DTW匹配
        2. 按置信度降序排序（高置信度优先）
        3. 标记已占用的字符范围，防止后续段重叠
        4. 置信度低于阈值的段被过滤

        Args:
            test_transcription: 测试音频的ASR转写结果
            ref_texts: 参考文本列表，None则使用self._ref_texts
            min_confidence: 最低置信度阈值

        Returns:
            SegmentMatch列表，按 offset 排序
        """
        if ref_texts is None:
            ref_texts = list(self._ref_texts.values())

        if not ref_texts:
            return []

        if not test_transcription.chars or not test_transcription.timestamps:
            return []

        test_chars = test_transcription.chars
        test_times = test_transcription.timestamps
        M = len(test_chars)
        CONF_THRESHOLD = min_confidence

        # ---------- 第1轮：对所有参考段做文本DTW ----------
        raw_matches = []
        for ref in ref_texts:
            try:
                alignment = self._text_aligner.align(ref.text, test_chars)
                if alignment is None:
                    continue

                start_pos = max(0, min(alignment["start_pos"], M - 1))
                end_pos = max(start_pos, min(alignment["end_pos"], M - 1))

                start_time_raw = test_times[start_pos] if start_pos < len(test_times) else 0.0
                end_time = (test_times[end_pos] + self._frame_shift
                           if end_pos < len(test_times)
                           else test_times[-1] + self._frame_shift)

                # 修正偏移：减去参考音频首字时间戳（精确修正）
                # ASR返回测试音频中首字的CTC峰值时间，但参考段在测试音频中的
                # 真实起始位置还需要考虑参考音频自身的首字偏移（段首静音+CTC峰值偏移）。
                # real_offset = test_first_char_time - ref_first_char_time
                ref_first = ref.ref_first_char_time if hasattr(ref, 'ref_first_char_time') and ref.ref_first_char_time > 0 else 0.0
                if ref_first > 0:
                    start_time = max(0.0, start_time_raw - ref_first)
                else:
                    # fallback: 仅减去段首静音（仍残留CTC峰值偏移约0.12s）
                    head_sil = ref.head_silence if hasattr(ref, 'head_silence') else 0.0
                    start_time = max(0.0, start_time_raw - head_sil)

                text_conf = alignment["confidence"]
                # 如果文本DTW本身置信度已低于阈值，提前跳过
                if text_conf < CONF_THRESHOLD * 0.5:
                    continue

                raw_matches.append({
                    "ref": ref,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": text_conf,
                    "wer": alignment["wer"],
                    "wcorr": alignment["wcorr"],
                    "n_match": alignment["n_match"],
                    "n_sub": alignment["n_sub"],
                    "n_ins": alignment["n_ins"],
                    "n_del": alignment["n_del"],
                })
            except Exception as e:
                logger.debug(f"[WeNet匹配] 匹配异常 {ref.ref_name}: {e}")
                continue

        # ---------- 第2轮：按置信度降序，非重叠约束 ----------
        # 已占用的测试字符位置（按字符索引标记，精确范围）
        occupied = [False] * M
        final_matches = []

        # 按置信度降序排序（分数高的优先占据位置）
        raw_matches.sort(key=lambda x: x["confidence"], reverse=True)

        for rm in raw_matches:
            sp = rm["start_pos"]
            ep = rm["end_pos"]

            # 置信度过滤（提前判断，避免无效计算）
            if rm["confidence"] < CONF_THRESHOLD:
                continue

            # 重叠检测：精确匹配，允许段间精确相邻
            # 一个段占用了 sp:ep 的字符位置，相邻段从 ep+1 开始即可
            overlap = any(occupied[sp:min(M, ep + 1)])
            if overlap:
                logger.debug(f"[WeNet匹配] 跳过重叠: {rm['ref'].ref_name} "
                            f"[{sp}:{ep}] 被更高置信度段占据")
                continue

            # 标记占用
            for k in range(sp, min(M, ep + 1)):
                occupied[k] = True

            # 创建SegmentMatch
            segment = SegmentMatch(
                ref_name=rm["ref"].ref_name,
                ref_file=rm["ref"].ref_file,
                offset=rm["start_time"],
                duration=rm["end_time"] - rm["start_time"],
                confidence=rm["confidence"],
                n_chars=ep - sp + 1,
                wer=rm["wer"],
                wcorr=rm["wcorr"],
                method="asr_text_dtw"
            )
            final_matches.append(segment)

            logger.info(
                f"[WeNet匹配] ✓ {rm['ref'].ref_name}: "
                f"offset={rm['start_time']:.2f}s dur={segment.duration:.1f}s "
                f"chars=[{sp}:{ep}] conf={rm['confidence']:.3f} "
                f"WER={rm['wer']:.3f}"
            )

        final_matches.sort(key=lambda x: x.offset)
        return final_matches

    # ---- 兼容接口 ----

    def match_with_fallback(self, test_audio_path: str) -> Dict:
        """
        统一的匹配入口（与 OptimizedMatcher.match_with_fallback 兼容）

        Returns:
            {
                "matched": bool,
                "ref_file": str or None,
                "ref_name": str or None,
                "offset": float,
                "method": str,
                "confidence": float,
                "snr": float,
                "wer": float,
                "detail": {"segment_matches": [...], "transcription": ...}
            }
        """
        result = {
            "matched": False, "ref_file": None, "ref_name": None,
            "offset": 0.0, "method": "none", "confidence": 0.0,
            "snr": 0.0, "wer": 0.0, "detail": {}
        }

        # 检查模型和参考文本是否就绪
        if not self._model:
            logger.warning("[WeNet匹配] 模型未加载，跳过ASR匹配")
            return result

        if not self._ref_texts:
            logger.warning("[WeNet匹配] 无参考文本，跳过ASR匹配")
            return result

        test_basename = os.path.basename(test_audio_path)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[WeNet匹配] 开始ASR语义匹配: {test_basename}")
        logger.info(f"{'=' * 60}")

        # Step 1: ASR转写
        transcription = self.transcribe(test_audio_path)
        if transcription is None or not transcription.chars:
            logger.warning(f"[WeNet匹配] ASR转写失败: {test_basename}")
            return result

        logger.info(f"[WeNet匹配] ASR完成: {len(transcription.chars)}字, "
                    f"文本=\"{transcription.raw_text[:50]}...\"")

        # Step 2: 文本DTW匹配
        ref_texts_list = list(self._ref_texts.values())
        segment_matches = self.match_segments(transcription, ref_texts_list)

        if not segment_matches:
            logger.warning(f"[WeNet匹配] 文本DTW未匹配到任何参考段: {test_basename}")
            return result

        # Step 3: 汇总结果
        result.update({
            "matched": True,
            "ref_file": segment_matches[0].ref_file,
            "ref_name": segment_matches[0].ref_name,
            "offset": segment_matches[0].offset,
            "method": "asr_text_dtw",
            "confidence": segment_matches[0].confidence,
            "wer": segment_matches[0].wer,
            "detail": {
                "segment_matches": [
                    {
                        "ref_name": m.ref_name,
                        "ref_file": m.ref_file,
                        "offset_in_test": m.offset,
                        "duration": m.duration,
                        "confidence": m.confidence,
                        "wer": m.wer,
                        "wcorr": m.wcorr,
                        "n_chars": m.n_chars,
                        "method": m.method
                    }
                    for m in segment_matches
                ],
                "transcription": transcription.raw_text,
                "n_chars": len(transcription.chars)
            }
        })

        logger.info(f"[WeNet匹配] 完成: {len(segment_matches)}个匹配, "
                    f"best={result['ref_name']} @ {result['offset']:.2f}s, "
                    f"WER={result['wer']:.3f}")
        return result

    # ---- 批量处理 ----

    def batch_match(self, test_file_list: List[str]) -> Dict[str, Dict]:
        """
        批量匹配：对每个测试文件进行ASR转写+文本DTW匹配

        Args:
            test_file_list: 测试音频文件路径列表

        Returns:
            {test_file_path: match_result_dict, ...}
        """
        results = {}
        for fpath in test_file_list:
            try:
                match = self.match_with_fallback(fpath)
                results[fpath] = match
            except Exception as e:
                logger.error(f"[WeNet匹配] 批量异常 {fpath}: {e}")
                results[fpath] = {
                    "matched": False, "method": "error", "error": str(e)
                }
        return results

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "initialized": self._initialized,
            "model_loaded": self._model is not None,
            "ref_texts_loaded": len(self._ref_texts),
            "ref_texts": list(self._ref_texts.keys()),
            "frame_shift_ms": self._frame_shift * 1000
        }
