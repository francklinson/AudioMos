"""
【旧版】参考音频匹配管道（指纹+DTW定位+互相关对齐）

此模块已逐步被 matching_optimizer 替代:
  - DTWLocator -> RobustDTWLocator (39维MFCC+CMVN+Delta+余弦距离)
  - ReferencePipeline.match_and_locate -> OptimizedMatcher._full_range_dtw_sweep
  - ReferencePipeline.cut_and_align -> 优化版切分自带HPSS精对齐, 无需cut_and_align

当前用途: backend/app/api/reference_audio.py 的匹配预览API仍使用此模块。
该API仅做匹配预览(不参与MOS评分), 后续可迁移。

整合的两阶段匹配流程:
  阶段1 - 指纹快速筛选：使用ReferenceMatcher在测试音频中快速搜索候选参考音频
  阶段2 - DTW精确定位：对候选参考音频使用MFCC+DTW进行精确的时间边界定位
  阶段3 - 音频切分对齐：根据定位结果切分测试音频并与参考音频对齐
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa
import soundfile as sf

# 添加路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'app', 'core'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'app', 'algorithms'))

from reference_matcher import (
    ReferenceMatcher, MatchResult, FingerprintConfig,
    get_reference_matcher, rebuild_matcher_database
)

logger = logging.getLogger('audiomos')


# ============================================================================
# DTW精确定位（复用audio_cut.py中的MFCCLocate逻辑）
# ============================================================================

class DTWLocator:
    """
    基于MFCC+DTW的音频精确定位器
    在指纹匹配确定的候选时间范围内，使用DTW精确搜索参考音频的边界。

    这是对 audio_cut.py 中 MFCCLocate 的改进版：
    - 支持限定搜索范围（由指纹匹配确定）
    - 支持批量精确定位
    - 增加置信度评估
    """

    def __init__(self, sr: int = 16000, hop_length: int = 512,
                 n_fft: int = 2048, n_mfcc: int = 13):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self._pre_enhancer = None  # 延迟初始化

        # 延迟导入dtw
        try:
            from dtw import dtw
            self._dtw = dtw
            self._dtw_available = True
        except ImportError:
            logger.warning("[DTW定位器] dtw-python未安装，DTW精确定位不可用")
            self._dtw = None
            self._dtw_available = False

    def _get_pre_enhancer(self):
        if self._pre_enhancer is None:
            try:
                from matching_optimizer import LightweightPreEnhancer
                self._pre_enhancer = LightweightPreEnhancer(method='wiener', sr=self.sr)
            except Exception:
                self._pre_enhancer = None
        return self._pre_enhancer

    def extract_mfcc(self, audio_path: str, pre_enhance: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """提取MFCC特征，可选预增强"""
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        # 可选预增强
        if pre_enhance:
            enhancer = self._get_pre_enhancer()
            if enhancer:
                y = enhancer.enhance(y, sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc,
                                     n_fft=self.n_fft, hop_length=self.hop_length)

        # 尝试CMVN归一化
        try:
            from matching_optimizer import apply_cmvn, add_delta_features
            mfcc = apply_cmvn(mfcc.T)
            mfcc = add_delta_features(mfcc)
            return mfcc, y
        except Exception:
            return mfcc.T, y

    def extract_mfcc_from_array(self, audio: np.ndarray, sr: int,
                                 pre_enhance: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """从numpy数组提取MFCC特征，可选预增强"""
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        if pre_enhance:
            enhancer = self._get_pre_enhancer()
            if enhancer:
                audio = enhancer.enhance(audio, sr)

        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
                                     n_fft=self.n_fft, hop_length=self.hop_length)

        try:
            from matching_optimizer import apply_cmvn, add_delta_features
            mfcc = apply_cmvn(mfcc.T)
            mfcc = add_delta_features(mfcc)
            return mfcc, audio
        except Exception:
            return mfcc.T, audio

    def index2time(self, index: int) -> float:
        """帧索引转时间（秒）"""
        return index * self.hop_length / self.sr

    def time2index(self, time_sec: float) -> int:
        """时间（秒）转帧索引"""
        return max(0, int(time_sec * self.sr / self.hop_length))

    def locate(self, test_audio_path: str, ref_audio_path: str,
               search_start: float = 0.0, search_end: float = None,
               jump_step: int = 5) -> Optional[Dict]:
        """
        在测试音频中精确定位参考音频的位置
        Args:
            test_audio_path: 测试音频路径
            ref_audio_path: 参考音频路径
            search_start: 搜索起始时间（秒）
            search_end: 搜索结束时间（秒），None表示到音频末尾
            jump_step: 滑动窗口步长（帧）
        Returns:
            {"start_time": float, "duration": float, "distance": float} 或 None
        """
        if not self._dtw_available:
            logger.warning("[DTW定位器] DTW不可用，返回None")
            return None

        logger.info(f"[DTW定位器] 精确定位: test={os.path.basename(test_audio_path)}, "
                     f"ref={os.path.basename(ref_audio_path)}, "
                     f"search_range=[{search_start:.1f}, {search_end or 'end'}]")

        # 先检测SNR，决定是否启用预增强
        use_pre_enhance = False
        _snr = 30.0
        try:
            from matching_optimizer import LightweightPreEnhancer
            _pre_check = LightweightPreEnhancer(method='wiener', sr=self.sr)
            _y_check, _sr_check = librosa.load(test_audio_path, sr=self.sr, mono=True)
            _snr = _pre_check.estimate_snr(_y_check)
            use_pre_enhance = _snr < 10
            if use_pre_enhance:
                logger.info(f"[DTW定位器] SNR={_snr:.1f}dB < 10dB，启用预增强+CMVN+Delta")
        except Exception:
            pass

        # 提取参考音频MFCC（参考音频是干净的，不做预增强）
        ref_mfcc, ref_y = self.extract_mfcc(ref_audio_path, pre_enhance=False)
        ref_duration = len(ref_y) / self.sr
        ref_frames = ref_mfcc.shape[0]

        # 提取测试音频MFCC（根据SNR决定是否预增强）
        test_mfcc, test_y = self.extract_mfcc(test_audio_path, pre_enhance=use_pre_enhance)
        test_duration = len(test_y) / self.sr

        # 确定搜索范围
        start_frame = self.time2index(search_start)
        if search_end is not None:
            end_frame = min(self.time2index(search_end), len(test_mfcc) - ref_frames)
        else:
            end_frame = len(test_mfcc) - ref_frames

        # 如果搜索范围太小（如自匹配场景），直接返回搜索起始位置
        if end_frame <= start_frame:
            logger.info(f"[DTW定位器] 搜索范围过小 [{start_frame}, {end_frame}]，"
                         f"可能为自匹配或短音频，直接返回搜索起始位置")
            return {
                "start_time": search_start,
                "duration": ref_duration,
                "distance": 0.0,
                "start_frame": start_frame
            }

        # 滑动窗口DTW搜索（使用余弦距离，对幅度变化更鲁棒）
        best_distance = float('inf')
        best_frame = -1

        # 在搜索范围内滑动
        for i in range(start_frame, end_frame, jump_step):
            window = test_mfcc[i:i + ref_frames]
            if window.shape[0] < ref_frames:
                break
            try:
                alignment = self._dtw(window, ref_mfcc, dist_method='cosine')
                if alignment.distance < best_distance:
                    best_distance = alignment.distance
                    best_frame = i
            except Exception as e:
                logger.debug(f"[DTW定位器] DTW计算异常: {e}")
                continue

        if best_frame < 0:
            logger.warning(f"[DTW定位器] 未找到有效匹配")
            return None

        start_time = self.index2time(best_frame)
        logger.info(f"[DTW定位器] 找到匹配: start_time={start_time:.3f}s, "
                     f"duration={ref_duration:.3f}s, distance={best_distance:.1f}, "
                     f"snr={_snr:.1f}dB, pre_enhanced={use_pre_enhance}")

        return {
            "start_time": start_time,
            "duration": ref_duration,
            "distance": best_distance,
            "start_frame": best_frame,
            "snr": _snr,
            "pre_enhanced": use_pre_enhance
        }

    def extract_mfcc_with_cmvn(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """提取CMVN归一化的MFCC（噪声鲁棒版本）"""
        try:
            from matching_optimizer import apply_cmvn, add_delta_features, librosa_load, librosa_mfcc
            y, sr = librosa_load(audio_path, sr=self.sr)
            mfcc = librosa_mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc,
                                 n_fft=self.n_fft, hop_length=self.hop_length)
            mfcc = apply_cmvn(mfcc.T)
            mfcc = add_delta_features(mfcc)
            return mfcc, y
        except Exception:
            # 回退到原始MFCC
            return self.extract_mfcc(audio_path)


# ============================================================================
# 匹配管道
# ============================================================================

class ReferencePipeline:
    """
    参考音频匹配管道
    整合指纹快速筛选 → DTW精确定位 → 音频切分对齐的完整流程
    """

    def __init__(self, ref_dir: str = None,
                 fingerprint_config: FingerprintConfig = None):
        """
        Args:
            ref_dir: 参考音频目录
            fingerprint_config: 指纹配置
        """
        self.ref_dir = ref_dir
        self.fingerprint_config = fingerprint_config or FingerprintConfig()
        self.matcher = ReferenceMatcher(ref_dir=ref_dir, config=self.fingerprint_config)
        self.dtw_locator = DTWLocator()
        self._initialized = False

    def initialize(self, ref_dir: str = None, force_rebuild: bool = False):
        """
        初始化管道：建立指纹数据库
        Args:
            ref_dir: 参考音频目录
            force_rebuild: 是否强制重建
        """
        target_dir = ref_dir or self.ref_dir
        if not target_dir:
            raise ValueError("未指定参考音频目录")

        if not self._initialized or force_rebuild:
            logger.info(f"[匹配管道] 初始化，建立指纹数据库: {target_dir}")
            self.matcher.build_database(target_dir)
            self._initialized = True

    def match_and_locate(self, test_audio_path: str,
                         min_confidence: float = 0.3,
                         use_dtw: bool = True,
                         dtw_search_margin: float = 5.0) -> List[Dict]:
        """
        完整的匹配+定位流程

        流程：
        1. 指纹快速筛选 → 候选参考音频列表
        2. DTW精确定位（对每个候选） → 精确时间边界
        3. 过滤低置信度结果

        Args:
            test_audio_path: 测试音频路径
            min_confidence: 最低置信度
            use_dtw: 是否使用DTW精确定位
            dtw_search_margin: DTW搜索范围扩展（秒），在指纹结果前后各扩展这么多秒

        Returns:
            [{ref_id, ref_file, ref_name, offset_in_test, duration, confidence,
              dtw_distance, ground_truth_text, description}, ...]
        """
        logger.info(f"[匹配管道] ====== 开始匹配: {os.path.basename(test_audio_path)} ======")

        if not self._initialized:
            logger.warning("[匹配管道] 未初始化，尝试自动初始化")
            self.initialize()

        # 阶段1：指纹快速筛选
        stage1_start = time.time()
        fingerprint_results = self.matcher.match_test_audio(
            test_audio_path, min_confidence=min_confidence
        )
        stage1_time = time.time() - stage1_start
        logger.info(f"[匹配管道] 阶段1完成: {len(fingerprint_results)} 个候选, "
                     f"耗时 {stage1_time:.2f}s")

        if not fingerprint_results:
            logger.info(f"[匹配管道] 未找到匹配的参考音频")
            return []

        # 阶段2：DTW精确定位（可选）
        final_results = []
        for fp_result in fingerprint_results:
            result = {
                "ref_id": fp_result.ref_id,
                "ref_file": fp_result.ref_file,
                "ref_name": fp_result.ref_name,
                "ref_duration": fp_result.ref_duration,
                "offset_in_test": fp_result.offset_in_test,
                "confidence": fp_result.confidence,
                "hash_matches": fp_result.hash_matches,
                "dtw_distance": None,
                "dtw_offset": None,
                "ground_truth_text": fp_result.ground_truth_text,
                "description": fp_result.description
            }

            if use_dtw and self.dtw_locator._dtw_available:
                # 在指纹结果周围扩展搜索范围
                search_start = max(0.0, fp_result.offset_in_test - dtw_search_margin)
                search_end = fp_result.offset_in_test + fp_result.ref_duration + dtw_search_margin

                dtw_result = self.dtw_locator.locate(
                    test_audio_path=test_audio_path,
                    ref_audio_path=fp_result.ref_file,
                    search_start=search_start,
                    search_end=search_end
                )

                if dtw_result:
                    result["offset_in_test"] = dtw_result["start_time"]
                    result["dtw_distance"] = dtw_result["distance"]
                    result["dtw_offset"] = dtw_result["start_time"]
                    # 归一化DTW距离: 距离/参考帧数
                    # 参考音频约12s，hop_length=512, sr=16000 → 约375帧
                    ref_frames = int(fp_result.ref_duration * 16000 / 512)
                    if ref_frames > 0:
                        normalized_dtw = dtw_result["distance"] / ref_frames
                    else:
                        normalized_dtw = dtw_result["distance"]
                    # 如果归一化DTW距离过大，降低置信度
                    # 经验阈值: <10=好匹配, 10-30=中等, >30=差匹配
                    if normalized_dtw > 30:
                        result["confidence"] *= 0.3
                        logger.warning(f"[匹配管道] 归一化DTW距离很大 ({normalized_dtw:.1f}), "
                                       f"大幅降低置信度到 {result['confidence']:.3f}")
                    elif normalized_dtw > 15:
                        result["confidence"] *= 0.7
                        logger.info(f"[匹配管道] 归一化DTW距离较大 ({normalized_dtw:.1f}), "
                                     f"适度降低置信度到 {result['confidence']:.3f}")
                else:
                    # DTW定位失败，降低置信度
                    result["confidence"] *= 0.3
                    logger.warning(f"[匹配管道] DTW定位失败 for {fp_result.ref_name}")

            final_results.append(result)

        # 过滤低置信度结果
        final_results = [r for r in final_results if r["confidence"] >= min_confidence]
        final_results.sort(key=lambda x: x["confidence"], reverse=True)

        total_time = time.time() - stage1_start
        logger.info(f"[匹配管道] 匹配完成: {len(final_results)} 个有效结果, "
                     f"总耗时 {total_time:.2f}s")
        for r in final_results:
            dtw_info = f", dtw_dist={r['dtw_distance']:.0f}" if r['dtw_distance'] else ""
            logger.info(f"[匹配管道]   {r['ref_name']}: offset={r['offset_in_test']:.2f}s, "
                         f"conf={r['confidence']:.3f}{dtw_info}")

        return final_results

    def cut_and_align(self, test_audio_path: str,
                      match_results: List[Dict],
                      output_dir: str,
                      redundancy: float = 0.5) -> List[str]:
        """
        根据匹配结果切分并对齐测试音频
        对每个匹配结果：
        1. 从测试音频中切出对应片段（含前后冗余）
        2. 与参考音频进行互相关对齐
        3. 保存对齐后的片段

        Args:
            test_audio_path: 测试音频路径
            match_results: 匹配结果列表（来自 match_and_locate）
            output_dir: 输出目录
            redundancy: 前后冗余时间（秒）

        Returns:
            对齐后的音频文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        test_basename = os.path.splitext(os.path.basename(test_audio_path))[0]
        output_files = []

        # 加载测试音频
        test_audio, test_sr = librosa.load(test_audio_path, sr=None, mono=True)
        logger.info(f"[匹配管道] 切分对齐: 测试音频 {test_basename}, "
                     f"sr={test_sr}, duration={len(test_audio)/test_sr:.2f}s")

        for i, match in enumerate(match_results):
            ref_file = match["ref_file"]
            offset = match["offset_in_test"]
            ref_duration = match["ref_duration"]

            # 切分参数
            cut_start = max(0.0, offset - redundancy)
            cut_duration = ref_duration + 2 * redundancy

            # 从测试音频中切出
            start_sample = int(cut_start * test_sr)
            end_sample = int((cut_start + cut_duration) * test_sr)
            end_sample = min(end_sample, len(test_audio))

            segment = test_audio[start_sample:end_sample]

            # 重采样到16kHz（如果需要）
            if test_sr != 16000:
                segment = librosa.resample(segment, orig_sr=test_sr, target_sr=16000)
                proc_sr = 16000
            else:
                proc_sr = test_sr

            # 加载参考音频并进行对齐
            try:
                ref_audio, ref_sr = librosa.load(ref_file, sr=16000, mono=True)

                # 互相关对齐
                from scipy import signal as scipy_signal
                # 去掉切分冗余后进行互相关
                redundancy_samples = int(redundancy * proc_sr)
                if len(segment) > redundancy_samples + len(ref_audio):
                    segment_for_corr = segment[redundancy_samples:redundancy_samples + len(ref_audio)]
                else:
                    segment_for_corr = segment[redundancy_samples:] if len(segment) > redundancy_samples else segment

                if len(segment_for_corr) > 0 and len(ref_audio) > 0:
                    # 确保参考和测试一样长
                    min_len = min(len(segment_for_corr), len(ref_audio))
                    correlation = scipy_signal.correlate(
                        ref_audio[:min_len], segment_for_corr[:min_len], mode='full'
                    )
                    lag = np.argmax(correlation) - (min_len - 1)
                    lag_seconds = lag / proc_sr

                    # 验证对齐质量：互相关峰值是否足够高
                    # 用自相关峰值作为参考
                    auto_corr = scipy_signal.correlate(
                        ref_audio[:min_len], ref_audio[:min_len], mode='same'
                    )
                    auto_peak = np.max(np.abs(auto_corr))
                    cross_peak = np.max(np.abs(correlation))
                    quality = cross_peak / auto_peak if auto_peak > 0 else 0

                    # 自适应阈值：低SNR时放宽条件
                    # 检查DTW结果中是否有SNR信息
                    dtw_snr = match.get('snr', 30) if isinstance(match.get('dtw_distance'), dict) else 30
                    if dtw_snr < 5:
                        quality_threshold = 0.10  # 极低SNR：大幅放宽
                        max_lag = int(2.0 * proc_sr)  # ±2秒
                    elif dtw_snr < 10:
                        quality_threshold = 0.15  # 低SNR：放宽
                        max_lag = int(1.5 * proc_sr)  # ±1.5秒
                    else:
                        quality_threshold = 0.25  # 正常SNR
                        max_lag = int(1.0 * proc_sr)  # ±1秒

                    # 限制 lag 在合理范围内，防止随机匹配
                    if quality >= quality_threshold and abs(lag) <= max_lag:
                        logger.info(f"[匹配管道]   对齐 {match['ref_name']}: "
                                     f"lag={lag} samples ({lag_seconds:.3f}s), quality={quality:.3f}, "
                                     f"threshold={quality_threshold}")
                        # 应用对齐
                        aligned = np.zeros_like(segment)
                        if lag > 0:
                            aligned[lag:] = segment[:-lag] if lag < len(segment) else segment
                        elif lag < 0:
                            aligned[:len(segment) + lag] = segment[-lag:]
                        else:
                            aligned = segment
                    else:
                        logger.warning(f"[匹配管道]   对齐质量不足 {match['ref_name']}: "
                                       f"lag={lag}, quality={quality:.3f}, "
                                       f"threshold={quality_threshold}, "
                                       f"跳过精细对齐，使用DTW切分结果")
                        aligned = segment
                else:
                    aligned = segment

                # 去掉前置冗余
                if len(aligned) > redundancy_samples:
                    aligned = aligned[redundancy_samples:]
                # 裁剪/填充到参考长度
                if len(aligned) > len(ref_audio):
                    aligned = aligned[:len(ref_audio)]
                elif len(aligned) < len(ref_audio):
                    aligned = np.pad(aligned, (0, len(ref_audio) - len(aligned)))

                # 保存对齐后的音频
                ref_name = os.path.splitext(match["ref_name"])[0]
                output_name = f"{test_basename}_{ref_name}_aligned.wav"
                output_path = os.path.join(output_dir, output_name)
                sf.write(output_path, aligned, proc_sr)
                output_files.append(output_path)
                logger.info(f"[匹配管道]   已保存对齐片段: {output_name}")

            except Exception as e:
                logger.error(f"[匹配管道]   对齐失败 {match['ref_name']}: {e}")
                # 即使对齐失败，也保存切分后的片段
                ref_name = os.path.splitext(match["ref_name"])[0]
                output_name = f"{test_basename}_{ref_name}_cut.wav"
                output_path = os.path.join(output_dir, output_name)
                sf.write(output_path, segment, proc_sr)
                output_files.append(output_path)

        return output_files

    def process_test_audio(self, test_audio_path: str,
                           output_dir: str,
                           min_confidence: float = 0.3,
                           use_dtw: bool = True) -> Dict:
        """
        完整的测试音频处理流程：匹配 → 定位 → 切分 → 对齐

        Args:
            test_audio_path: 测试音频路径
            output_dir: 输出目录
            min_confidence: 最低置信度
            use_dtw: 是否使用DTW精确定位

        Returns:
            {
                "test_audio": str,
                "matches": List[Dict],     # 匹配结果
                "aligned_files": List[str], # 对齐后的文件路径
                "no_match": bool,           # 是否无匹配
                "elapsed_time": float       # 总耗时
            }
        """
        total_start = time.time()

        # 确保初始化
        if not self._initialized:
            self.initialize()

        # 匹配+定位
        match_results = self.match_and_locate(
            test_audio_path,
            min_confidence=min_confidence,
            use_dtw=use_dtw
        )

        # 切分+对齐
        aligned_files = []
        if match_results:
            aligned_files = self.cut_and_align(
                test_audio_path,
                match_results,
                output_dir
            )

        elapsed = time.time() - total_start

        result = {
            "test_audio": test_audio_path,
            "matches": match_results,
            "aligned_files": aligned_files,
            "no_match": len(match_results) == 0,
            "elapsed_time": elapsed
        }

        logger.info(f"[匹配管道] 处理完成: {len(match_results)} 个匹配, "
                     f"{len(aligned_files)} 个对齐文件, 耗时 {elapsed:.2f}s")

        return result

    def get_statistics(self) -> Dict:
        """获取管道统计信息"""
        return {
            "initialized": self._initialized,
            "ref_dir": self.ref_dir,
            "matcher": self.matcher.get_statistics(),
            "dtw_available": self.dtw_locator._dtw_available
        }


# ============================================================================
# 批量处理
# ============================================================================

def process_multiple_test_files(test_audio_paths: List[str],
                                ref_dir: str,
                                output_dir: str,
                                max_workers: int = 4,
                                min_confidence: float = 0.3,
                                use_dtw: bool = True) -> List[Dict]:
    """
    批量处理多个测试音频文件
    Args:
        test_audio_paths: 测试音频路径列表
        ref_dir: 参考音频目录
        output_dir: 输出目录
        max_workers: 最大并行数
        min_confidence: 最低置信度
        use_dtw: 是否使用DTW
    Returns:
        处理结果列表
    """
    pipeline = ReferencePipeline(ref_dir=ref_dir)
    pipeline.initialize()

    results = []
    # 按顺序处理（避免并行DTW导致的内存问题）
    for test_path in test_audio_paths:
        result = pipeline.process_test_audio(
            test_path, output_dir,
            min_confidence=min_confidence,
            use_dtw=use_dtw
        )
        results.append(result)

    return results


# ============================================================================
# 全局管道实例
# ============================================================================

_global_pipeline: Optional[ReferencePipeline] = None


def get_reference_pipeline(ref_dir: str = None,
                           force_rebuild: bool = False) -> ReferencePipeline:
    """获取全局参考音频匹配管道实例"""
    global _global_pipeline

    if _global_pipeline is None:
        _global_pipeline = ReferencePipeline(ref_dir=ref_dir)
        if ref_dir:
            _global_pipeline.initialize(ref_dir, force_rebuild=force_rebuild)
    elif force_rebuild and ref_dir:
        _global_pipeline.initialize(ref_dir, force_rebuild=True)

    return _global_pipeline
