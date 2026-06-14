"""
参考音频内容匹配模块
基于Shazam算法的STFT音频指纹技术，用于在测试音频中快速搜索匹配的参考音频。

核心思路（两阶段匹配）：
  阶段1 - 指纹快速筛选：对所有参考音频建立指纹数据库，提取测试音频指纹并在数据库中快速查找，
           返回候选匹配（参考音频ID + 时间偏移 + 匹配hash数）。
  阶段2 - DTW精确定位：对阶段1中匹配hash数超过阈值的候选，使用DTW在限定时间范围内精确搜索，
           确定参考音频在测试音频中的精确起止位置。

技术参考：
  - Shazam算法: Wang, A. "An Industrial Strength Audio Search Algorithm", ISMIR 2003
  - 本项目参考实现: /home/zhouchenghao/PycharmProjects/ASD_for_SPK/backend/core/shazam/

使用方式：
  matcher = ReferenceMatcher(ref_dir="data/ref")
  matcher.build_database()  # 建立参考音频指纹库
  matches = matcher.match_test_audio("test.wav")  # 在测试音频中搜索匹配
  # matches: [{"ref_id": "...", "ref_file": "...", "offset_in_test": 1.5, "confidence": 0.9}, ...]
"""
import os
import sys
import hashlib
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

import numpy as np
import librosa
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure

# 添加算法路径
_ALGORITHMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algorithms')
sys.path.insert(0, _ALGORITHMS_DIR)

# 模块级日志记录器
logger = logging.getLogger('audiomos')


# ============================================================================
# 配置参数
# ============================================================================

@dataclass
class FingerprintConfig:
    """音频指纹配置参数"""
    # 采样率
    sr: int = 16000
    # STFT参数
    n_fft: int = 4096
    hop_length: int = 1024
    win_length: int = 4096
    # 峰值检测
    amp_min: float = 5.0        # 能量最小值(dB)
    neighborhood: int = 15       # 局部最大值区域范围
    # Hash生成
    near_num: int = 20           # 锚点近邻个数
    min_time_delta: int = 0      # 最小时间间隔(frames)
    max_time_delta: int = 200    # 最大时间间隔(frames)
    # 匹配
    min_hash_match: int = 10     # 最少匹配hash数（从3提高到10，确保匹配质量）
    min_confidence: float = 0.05  # 最低置信度（从0.10降低到0.05，主要依赖hash数量）


# 全局默认配置
DEFAULT_CONFIG = FingerprintConfig()


# ============================================================================
# 指纹提取和Hash生成（基于Shazam算法）
# ============================================================================

class AudioFingerprinter:
    """
    音频指纹提取器
    实现Shazam算法的核心指纹生成逻辑：
    1. STFT频谱计算
    2. 频谱峰值检测（星座图）
    3. Hash生成（频率对 + 时间差）
    """

    def __init__(self, config: FingerprintConfig = None):
        self.config = config or DEFAULT_CONFIG

    def compute_spectrogram(self, audio_path: str) -> np.ndarray:
        """
        计算音频的STFT频谱图
        Args:
            audio_path: 音频文件路径
        Returns:
            spectrogram: (频率, 时间) 的频谱幅度矩阵
        """
        y, sr = librosa.load(audio_path, sr=self.config.sr, mono=True)
        # STFT
        stft = librosa.stft(
            y,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        spectrogram = np.abs(stft)
        return spectrogram

    def compute_spectrogram_from_array(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """从numpy数组计算频谱图"""
        if sr != self.config.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sr)
        stft = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        return np.abs(stft)

    def _handle_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        处理频谱图：替换零值、转dB
        """
        # 用最小值替换0
        min_val = np.min(spectrogram[np.nonzero(spectrogram)])
        spectrogram[spectrogram == 0] = min_val
        # 转dB
        spectrogram = 10 * np.log10(spectrogram)
        # 防止负无穷
        spectrogram[spectrogram == -np.inf] = 0
        return spectrogram

    def _find_peaks(self, spectrogram: np.ndarray) -> List[Tuple[int, int]]:
        """
        在频谱图中检测局部最大值点（星座图）
        Args:
            spectrogram: dB单位的频谱图
        Returns:
            peaks: [(time_idx, freq_idx), ...] 峰值点列表
        """
        # 创建十字形结构元素
        struct = generate_binary_structure(2, 1)
        # 扩大十字架范围
        neighborhood = iterate_structure(struct, self.config.neighborhood)
        # 找局部最大值
        local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
        # 获取能量值
        amps = spectrogram[local_max].flatten()
        # 获取时间和频率索引
        j, i = np.where(local_max)  # j=freq, i=time
        # 组合为 (time, freq, amp)
        peaks_with_amp = list(zip(i, j, amps))
        # 过滤低能量峰值
        peaks_with_amp = [p for p in peaks_with_amp if p[2] > self.config.amp_min]
        # 只保留时间和频率
        peaks = [(p[0], p[1]) for p in peaks_with_amp]
        return peaks

    def _generate_hashes(self, peaks: List[Tuple[int, int]]) -> List[Tuple[str, int]]:
        """
        从峰值点生成Hash
        每个锚点与其后的near_num个近邻点生成Hash
        Hash = SHA1(freq1 | freq2 | time_delta)
        Args:
            peaks: [(time_idx, freq_idx), ...] 按时间排序的峰值点
        Returns:
            hashes: [(hash_hex, time_offset), ...]
        """
        peaks = sorted(peaks, key=lambda x: x[0])  # 按时间排序
        hashes = []

        for i in range(len(peaks)):
            for j in range(1, self.config.near_num):
                if i + j >= len(peaks):
                    break
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                t_delta = t2 - t1

                if self.config.min_time_delta <= t_delta <= self.config.max_time_delta:
                    hash_str = f"{f1}|{f2}|{t_delta}"
                    hash_hex = hashlib.sha1(hash_str.encode("utf-8")).hexdigest()
                    hashes.append((hash_hex, t1))

        return hashes

    def extract_hashes(self, audio_path: str) -> List[Tuple[str, int]]:
        """
        提取音频文件的指纹Hash
        Args:
            audio_path: 音频文件路径
        Returns:
            hashes: [(hash_hex, time_offset_in_frames), ...]
        """
        spectrogram = self.compute_spectrogram(audio_path)
        spectrogram = self._handle_spectrogram(spectrogram)
        peaks = self._find_peaks(spectrogram)
        hashes = self._generate_hashes(peaks)
        return hashes

    def frame_to_time(self, frame_idx: int) -> float:
        """将帧索引转换为时间（秒）"""
        return frame_idx * self.config.hop_length / self.config.sr

    def time_to_frame(self, time_sec: float) -> int:
        """将时间（秒）转换为帧索引"""
        return int(time_sec * self.config.sr / self.config.hop_length)


# ============================================================================
# 参考音频指纹数据库
# ============================================================================

@dataclass
class ReferenceEntry:
    """参考音频数据库条目"""
    ref_id: str           # 参考音频ID（来自metadata.json）
    ref_file: str         # 参考音频文件路径
    ref_name: str         # 参考音频文件名
    duration: float       # 音频时长（秒）
    hash_count: int       # 指纹Hash数量
    # 附加信息
    ground_truth_text: Optional[str] = None  # WER评估用的ground truth文本
    description: Optional[str] = None


class FingerprintDatabase:
    """
    参考音频指纹数据库（内存版本）
    存储所有参考音频的指纹Hash，支持快速查询。

    数据结构：
      hash_index: {hash_hex: [(ref_id, offset_in_ref_frames), ...]}
      用于快速查找哪些参考音频在哪个时间偏移处包含某个Hash。
    """

    def __init__(self, config: FingerprintConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.fingerprinter = AudioFingerprinter(self.config)
        # Hash索引: hash -> [(ref_id, offset_in_ref), ...]
        self.hash_index: Dict[str, List[Tuple[str, int]]] = {}
        # 参考音频条目: ref_id -> ReferenceEntry
        self.entries: Dict[str, ReferenceEntry] = {}
        # 统计
        self.total_hashes: int = 0
        self.build_time: float = 0.0

    def add_reference(self, ref_id: str, audio_path: str,
                      ground_truth_text: str = None,
                      description: str = None) -> int:
        """
        添加一个参考音频到数据库
        Args:
            ref_id: 参考音频ID
            audio_path: 音频文件路径
            ground_truth_text: WER评估文本
            description: 描述
        Returns:
            添加的Hash数量
        """
        logger.info(f"[指纹库] 添加参考音频: {ref_id} -> {audio_path}")

        # 提取Hash
        start = time.time()
        hashes = self.fingerprinter.extract_hashes(audio_path)
        elapsed = time.time() - start

        # 获取音频时长
        try:
            y, sr = librosa.load(audio_path, sr=self.config.sr, mono=True)
            duration = len(y) / sr
        except Exception:
            duration = 0.0

        # 存储条目
        self.entries[ref_id] = ReferenceEntry(
            ref_id=ref_id,
            ref_file=audio_path,
            ref_name=os.path.basename(audio_path),
            duration=duration,
            hash_count=len(hashes),
            ground_truth_text=ground_truth_text,
            description=description
        )

        # 更新Hash索引
        for hash_hex, offset in hashes:
            if hash_hex not in self.hash_index:
                self.hash_index[hash_hex] = []
            self.hash_index[hash_hex].append((ref_id, offset))

        self.total_hashes += len(hashes)

        logger.info(f"[指纹库] 参考音频 {ref_id} 添加完成: "
                     f"{len(hashes)} 个Hash, 时长 {duration:.1f}s, 耗时 {elapsed:.2f}s")

        return len(hashes)

    def remove_reference(self, ref_id: str):
        """从数据库中移除一个参考音频"""
        if ref_id not in self.entries:
            return

        logger.info(f"[指纹库] 移除参考音频: {ref_id}")

        removed_count = 0
        # 从Hash索引中移除
        to_delete = []
        for hash_hex, ref_list in self.hash_index.items():
            old_len = len(ref_list)
            ref_list[:] = [(rid, off) for rid, off in ref_list if rid != ref_id]
            removed_count += old_len - len(ref_list)
            if not ref_list:
                to_delete.append(hash_hex)

        for hash_hex in to_delete:
            del self.hash_index[hash_hex]

        # 移除条目并更新计数
        entry = self.entries.pop(ref_id, None)
        if entry:
            self.total_hashes -= entry.hash_count
        logger.info(f"[指纹库] 参考音频 {ref_id} 已移除 (清理 {removed_count} 条索引, 剩余 {self.total_hashes} Hash)")

    def query(self, hashes: List[Tuple[str, int]]) -> List[Tuple[str, int, int]]:
        """
        在数据库中查询匹配的Hash
        Args:
            hashes: 测试音频的Hash列表 [(hash_hex, offset_in_query), ...]
        Returns:
            match_list: [(ref_id, offset_in_ref, offset_in_query), ...]
            注意：返回列表而非集合，因为Shazam算法依赖重复计数来判断匹配强度。
            同一个(ref_id, offset_ref, offset_query)三元组出现多次意味着更强的匹配。
        """
        match_list = []

        for hash_hex, offset_query in hashes:
            if hash_hex in self.hash_index:
                for ref_id, offset_ref in self.hash_index[hash_hex]:
                    match_list.append((ref_id, offset_ref, offset_query))

        return match_list

    def get_entry(self, ref_id: str) -> Optional[ReferenceEntry]:
        """获取参考音频条目"""
        return self.entries.get(ref_id)

    def get_all_ref_ids(self) -> List[str]:
        """获取所有参考音频ID"""
        return list(self.entries.keys())

    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        return {
            "total_references": len(self.entries),
            "total_hashes": self.total_hashes,
            "unique_hashes": len(self.hash_index),
            "build_time": self.build_time,
            "references": [
                {
                    "ref_id": e.ref_id,
                    "ref_name": e.ref_name,
                    "duration": e.duration,
                    "hash_count": e.hash_count,
                    "has_ground_truth": e.ground_truth_text is not None
                }
                for e in self.entries.values()
            ]
        }

    def clear(self):
        """清空数据库"""
        self.hash_index.clear()
        self.entries.clear()
        self.total_hashes = 0
        self.build_time = 0.0


# ============================================================================
# 匹配结果和匹配器
# ============================================================================

@dataclass
class MatchResult:
    """单个匹配结果"""
    ref_id: str                     # 参考音频ID
    ref_file: str                   # 参考音频文件路径
    ref_name: str                   # 参考音频文件名
    ref_duration: float             # 参考音频时长（秒）
    offset_in_test: float           # 在测试音频中的起始时间（秒）
    offset_in_test_frames: int      # 在测试音频中的起始帧
    hash_matches: int               # 匹配的Hash数量
    confidence: float               # 置信度 (0-1)
    dtw_distance: Optional[float] = None    # DTW距离（精确定位后填充）
    ground_truth_text: Optional[str] = None  # 参考文本
    description: Optional[str] = None


class ReferenceMatcher:
    """
    参考音频内容匹配器
    提供完整的匹配管道：指纹提取 → 数据库查询 → 时间对齐 → 置信度计算
    """

    def __init__(self, ref_dir: str = None, config: FingerprintConfig = None):
        """
        Args:
            ref_dir: 参考音频目录路径
            config: 指纹配置
        """
        self.config = config or DEFAULT_CONFIG
        self.fingerprinter = AudioFingerprinter(self.config)
        self.database = FingerprintDatabase(self.config)
        self.ref_dir = ref_dir
        self._metadata_file = None
        if ref_dir:
            self._metadata_file = Path(ref_dir) / ".metadata.json"

    def build_database(self, ref_dir: str = None) -> Dict:
        """
        建立参考音频指纹数据库
        从参考音频目录加载所有音频文件并提取指纹
        Args:
            ref_dir: 参考音频目录（如果不指定则使用初始化时的目录）
        Returns:
            统计信息
        """
        target_dir = ref_dir or self.ref_dir
        if not target_dir:
            raise ValueError("未指定参考音频目录")

        ref_path = Path(target_dir)
        if not ref_path.exists():
            logger.warning(f"[参考匹配器] 参考音频目录不存在: {target_dir}")
            return self.database.get_statistics()

        logger.info(f"[参考匹配器] 开始建立指纹数据库: {target_dir}")
        start_time = time.time()

        # 清空现有数据库
        self.database.clear()

        # 加载元数据
        metadata = self._load_metadata(target_dir)
        audios = metadata.get("audios", {})

        # 遍历参考音频文件
        loaded_count = 0
        for audio_id, info in audios.items():
            filename = info.get("filename", "")
            file_path = ref_path / filename
            if file_path.exists() and file_path.suffix.lower() in ['.wav', '.mp3', '.flac']:
                try:
                    gt_text = info.get("ground_truth_text")
                    description = info.get("description")
                    self.database.add_reference(
                        ref_id=audio_id,
                        audio_path=str(file_path),
                        ground_truth_text=gt_text,
                        description=description
                    )
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"[参考匹配器] 添加参考音频失败 {filename}: {e}")

        # 如果metadata中没有记录，也尝试直接扫描目录
        if loaded_count == 0:
            logger.info(f"[参考匹配器] 元数据为空，直接扫描目录")
            for f in sorted(ref_path.iterdir()):
                if f.suffix.lower() in ['.wav', '.mp3', '.flac']:
                    try:
                        ref_id = f"direct_{f.stem}"
                        self.database.add_reference(
                            ref_id=ref_id,
                            audio_path=str(f)
                        )
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"[参考匹配器] 添加参考音频失败 {f.name}: {e}")

        self.database.build_time = time.time() - start_time
        stats = self.database.get_statistics()
        logger.info(f"[参考匹配器] 指纹数据库建立完成: "
                     f"{loaded_count} 个参考音频, "
                     f"{stats['total_hashes']} 个Hash, "
                     f"耗时 {self.database.build_time:.2f}s")

        return stats

    def match_test_audio(self, test_audio_path: str,
                         min_confidence: float = None) -> List[MatchResult]:
        """
        在测试音频中搜索匹配的参考音频
        阶段1：指纹快速筛选
        Args:
            test_audio_path: 测试音频文件路径
            min_confidence: 最低置信度阈值
        Returns:
            匹配结果列表（按置信度降序排列）
        """
        if min_confidence is None:
            min_confidence = self.config.min_confidence

        logger.info(f"[参考匹配器] 开始匹配测试音频: {test_audio_path}")

        if not self.database.entries:
            logger.warning(f"[参考匹配器] 指纹数据库为空，无法匹配")
            return []

        # 提取测试音频的指纹
        start_time = time.time()
        test_hashes = self.fingerprinter.extract_hashes(test_audio_path)
        extract_time = time.time() - start_time
        logger.info(f"[参考匹配器] 测试音频指纹提取: {len(test_hashes)} 个Hash, "
                     f"耗时 {extract_time:.2f}s")

        # 在数据库中查找匹配Hash
        start_time = time.time()
        match_hash_list = self.database.query(test_hashes)
        query_time = time.time() - start_time
        logger.info(f"[参考匹配器] 数据库查询: {len(match_hash_list)} 个匹配Hash对, "
                     f"耗时 {query_time:.2f}s")

        if not match_hash_list:
            logger.info(f"[参考匹配器] 未找到任何匹配的Hash")
            return []

        # 对齐匹配结果：按 (ref_id, offset_diff) 聚合
        # offset_diff = offset_in_ref - offset_in_query
        # 相同的offset_diff意味着参考音频和测试音频在时间上对齐
        # 注意：这里不适用set去重，Shazam算法依赖重复计数来判断匹配强度
        alignment_counts: Dict[Tuple[str, int], int] = {}
        for ref_id, offset_ref, offset_query in match_hash_list:
            offset_diff = int(offset_ref) - int(offset_query)
            key = (ref_id, offset_diff)
            alignment_counts[key] = alignment_counts.get(key, 0) + 1

        logger.info(f"[参考匹配器] 对齐分析: 共 {len(alignment_counts)} 个(ref_id, offset_diff)组合")

        # 按ref_id分组，找出每个参考音频的**所有**可能匹配位置（支持混合音频）
        ref_all_matches: Dict[str, List[Tuple[int, int]]] = {}  # ref_id -> [(offset_diff, hash_count), ...]
        for (ref_id, offset_diff), count in alignment_counts.items():
            if ref_id not in ref_all_matches:
                ref_all_matches[ref_id] = []
            ref_all_matches[ref_id].append((offset_diff, count))

        # 对每个参考音频的匹配位置按hash_count降序排序
        for ref_id in ref_all_matches:
            ref_all_matches[ref_id].sort(key=lambda x: x[1], reverse=True)

        # 生成匹配结果：从所有参考音频的所有位置中找出前N个最佳匹配
        all_candidates = []
        for ref_id, matches in ref_all_matches.items():
            entry = self.database.get_entry(ref_id)
            if entry is None:
                continue

            for offset_diff, hash_count in matches:
                # 计算置信度
                confidence = min(1.0, hash_count / max(1, entry.hash_count * 0.15))

                # 记录所有候选（包括低于阈值的，用于调试）
                # 只使用 hash_count 判断，不使用 confidence
                all_candidates.append({
                    'ref_id': ref_id,
                    'ref_entry': entry,
                    'offset_diff': offset_diff,
                    'hash_count': hash_count,
                    'confidence': confidence,
                    'passed': hash_count >= self.config.min_hash_match
                })

        # 按置信度降序排序
        all_candidates.sort(key=lambda x: x['confidence'], reverse=True)

        logger.info(f"[参考匹配器] 候选匹配分析（前10个）:")
        for i, cand in enumerate(all_candidates[:10]):
            status = "✓" if cand['passed'] else "✗"
            logger.info(f"[参考匹配器]   {status} {cand['ref_id']} @ {self.fingerprinter.frame_to_time(max(0, cand['offset_diff'])):.2f}s: "
                       f"hash={cand['hash_count']}, conf={cand['confidence']:.3f}")

        # 筛选通过阈值的候选
        # 策略：支持混合音频，只要候选通过阈值就保留
        # 但同一参考音频只保留最佳匹配（避免时间重叠的重复匹配）
        results = []
        used_ref_ids = set()
        for cand in all_candidates:
            if not cand['passed']:
                continue

            ref_id = cand['ref_id']
            # 如果同一参考音频已经匹配过，跳过（避免重复）
            # 注意：如果需要支持同一参考音频在测试音频中出现多次，需要更复杂的去重逻辑
            if ref_id in used_ref_ids:
                logger.debug(f"[参考匹配器] {ref_id}: 已存在匹配，跳过次要匹配位置")
                continue

            entry = cand['ref_entry']
            offset_diff = cand['offset_diff']
            hash_count = cand['hash_count']
            confidence = cand['confidence']

            # 计算在测试音频中的时间偏移
            offset_in_test_frames = offset_diff
            offset_in_test = self.fingerprinter.frame_to_time(max(0, offset_in_test_frames))

            result = MatchResult(
                ref_id=ref_id,
                ref_file=entry.ref_file,
                ref_name=entry.ref_name,
                ref_duration=entry.duration,
                offset_in_test=offset_in_test,
                offset_in_test_frames=offset_in_test_frames,
                hash_matches=hash_count,
                confidence=confidence,
                ground_truth_text=entry.ground_truth_text,
                description=entry.description
            )
            results.append(result)
            used_ref_ids.add(ref_id)

        # 按置信度降序排列
        results.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"[参考匹配器] 匹配完成: {len(results)} 个结果")
        for r in results:
            logger.info(f"[参考匹配器]   {r.ref_name}: offset={r.offset_in_test:.2f}s, "
                         f"confidence={r.confidence:.3f}, hash_matches={r.hash_matches}")

        return results

    def _load_metadata(self, ref_dir: str) -> dict:
        """加载参考音频元数据"""
        metadata_file = Path(ref_dir) / ".metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[参考匹配器] 加载元数据失败: {e}")
        return {"audios": {}, "default_id": None}

    # ── 增量更新方法 ──

    def add_single_reference(self, ref_id: str, audio_path: str,
                             ground_truth_text: str = None,
                             description: str = None) -> Dict:
        """
        增量添加单个参考音频到指纹数据库（无需全量重建）
        Args:
            ref_id: 参考音频ID
            audio_path: 音频文件路径
            ground_truth_text: WER评估文本
            description: 描述
        Returns:
            {"success": bool, "hash_count": int, "elapsed": float}
        """
        start = time.time()
        try:
            # 如果已存在，先移除旧的（更新场景）
            if ref_id in self.database.entries:
                self.database.remove_reference(ref_id)

            hash_count = self.database.add_reference(
                ref_id=ref_id,
                audio_path=audio_path,
                ground_truth_text=ground_truth_text,
                description=description
            )
            elapsed = time.time() - start
            logger.info(f"[参考匹配器] 增量添加参考音频 {ref_id} 完成: "
                         f"{hash_count} Hash, 耗时 {elapsed:.2f}s")
            return {"success": True, "hash_count": hash_count, "elapsed": elapsed}
        except Exception as e:
            logger.error(f"[参考匹配器] 增量添加失败 {ref_id}: {e}")
            return {"success": False, "error": str(e), "elapsed": time.time() - start}

    def remove_single_reference(self, ref_id: str) -> Dict:
        """
        增量移除单个参考音频（无需全量重建）
        Args:
            ref_id: 参考音频ID
        Returns:
            {"success": bool, "elapsed": float}
        """
        start = time.time()
        try:
            self.database.remove_reference(ref_id)
            elapsed = time.time() - start
            logger.info(f"[参考匹配器] 增量移除参考音频 {ref_id} 完成, 耗时 {elapsed:.2f}s")
            return {"success": True, "elapsed": elapsed}
        except Exception as e:
            logger.error(f"[参考匹配器] 增量移除失败 {ref_id}: {e}")
            return {"success": False, "error": str(e), "elapsed": time.time() - start}

    def get_statistics(self) -> Dict:
        """获取匹配器统计信息"""
        return {
            "config": {
                "sr": self.config.sr,
                "n_fft": self.config.n_fft,
                "hop_length": self.config.hop_length,
                "min_hash_match": self.config.min_hash_match,
                "min_confidence": self.config.min_confidence
            },
            "database": self.database.get_statistics()
        }


# ============================================================================
# 全局匹配器实例管理
# ============================================================================

_global_matcher: Optional[ReferenceMatcher] = None


def get_reference_matcher(ref_dir: str = None, force_rebuild: bool = False) -> ReferenceMatcher:
    """
    获取全局参考音频匹配器实例
    Args:
        ref_dir: 参考音频目录
        force_rebuild: 是否强制重建数据库
    Returns:
        ReferenceMatcher实例
    """
    global _global_matcher

    if _global_matcher is None:
        _global_matcher = ReferenceMatcher(ref_dir=ref_dir)
        if ref_dir and Path(ref_dir).exists():
            _global_matcher.build_database(ref_dir)
    elif force_rebuild and ref_dir:
        _global_matcher.build_database(ref_dir)

    return _global_matcher


def rebuild_matcher_database(ref_dir: str = None):
    """重建匹配器数据库（全量）"""
    global _global_matcher
    if _global_matcher is not None and ref_dir:
        _global_matcher.build_database(ref_dir)
        logger.info("[参考匹配器] 数据库已全量重建")


def add_to_matcher_database(ref_id: str, audio_path: str, ref_dir: str = None,
                            ground_truth_text: str = None,
                            description: str = None) -> Dict:
    """
    增量添加一个参考音频到全局匹配器数据库
    如果全局匹配器未初始化则先初始化
    """
    global _global_matcher
    if _global_matcher is None:
        if ref_dir:
            _global_matcher = ReferenceMatcher(ref_dir=ref_dir)
            _global_matcher.build_database(ref_dir)
        else:
            return {"success": False, "error": "全局匹配器未初始化且未提供ref_dir"}

    return _global_matcher.add_single_reference(
        ref_id=ref_id,
        audio_path=audio_path,
        ground_truth_text=ground_truth_text,
        description=description
    )


def remove_from_matcher_database(ref_id: str, ref_dir: str = None) -> Dict:
    """
    增量移除一个参考音频
    """
    global _global_matcher
    if _global_matcher is None:
        if ref_dir:
            _global_matcher = ReferenceMatcher(ref_dir=ref_dir)
            _global_matcher.build_database(ref_dir)
        return {"success": False, "error": "全局匹配器未初始化"}

    return _global_matcher.remove_single_reference(ref_id)
