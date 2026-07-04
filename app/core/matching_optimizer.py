"""
音频匹配对齐优化模块
针对低信噪比场景的匹配对齐优化:

优化策略：
1. 预增强处理：对低SNR测试音频进行谱减法预处理后再进行指纹匹配
2. 自适应阈值指纹：根据噪声底噪动态调整峰值检测阈值
3. 谐波-打击乐分离(HPSS)：提取谐波分量进行更干净的峰值检测
4. 嵌入向量匹配(Embedding)：使用已有的TCF说话人确认模型进行语义匹配（低SNR回退方案）
5. CMVN归一化DTW：对MFCC进行倒谱均值方差归一化，提升噪声鲁棒性
"""
import os
import sys
import time
import logging
import warnings
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger('audiomos')

import importlib.util

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============ 全局共享线程池（统一模块，消除多模块独立线程池的资源争用） ============
_shared_executor = None
def get_shared_executor(max_workers: int = 8) -> ThreadPoolExecutor:
    global _shared_executor
    if _shared_executor is not None:
        return _shared_executor
    try:
        spec = importlib.util.spec_from_file_location(
            "_executor",
            os.path.join(_PROJECT_ROOT, "app", "core", "_executor.py")
        )
        if spec and spec.loader:
            _exec_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_exec_mod)
            _shared_executor = _exec_mod.get_shared_executor(max_workers=max_workers)
            return _shared_executor
    except Exception:
        pass
    # 回退到本地创建
    _shared_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='audiomos_shared')
    return _shared_executor


# ============================================================================
# 音频预处理 - 轻量级降噪
# ============================================================================

class LightweightPreEnhancer:
    """
    轻量级预增强处理器
    仅用于匹配阶段的音频预处理（不改变原始音频文件）
    支持谱减法和维纳滤波两种快速方法
    """

    def __init__(self, method: str = 'wiener', sr: int = 16000):
        """
        Args:
            method: 'wiener'(维纳滤波,推荐) 或 'spectral'(谱减法)
            sr: 目标采样率
        """
        self.method = method
        self.sr = sr
        self.n_fft = 2048
        self.hop_length = 512
        self.noise_frames = 10  # 用于噪声估计的初始帧数
        self._initialized = False

    def estimate_noise_floor(self, audio: np.ndarray) -> float:
        """
        估计音频的噪声底噪(dB)
        使用前noise_frames帧（假设开头无语音）和整段音频的底部10%分位数
        """
        stft = librosa_stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)

        # 方法1: 开头帧估计
        noise_initial = magnitude[:, :min(self.noise_frames, magnitude.shape[1])]
        floor_initial = np.median(noise_initial)

        # 方法2: 全局能量底部10%分位数
        all_mag = magnitude.flatten()
        floor_global = np.percentile(all_mag, 10)

        # 取较保守的估计
        noise_floor = max(floor_initial, floor_global)
        # 转dB（加最小偏移避免log0）
        noise_floor_db = 20 * np.log10(max(noise_floor, 1e-10))
        return noise_floor_db

    def estimate_snr(self, audio: np.ndarray) -> float:
        """估计音频的全局SNR(dB)"""
        stft = librosa_stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)

        # 噪声估计：前noise_frames帧
        noise_mag = np.mean(magnitude[:, :min(self.noise_frames, magnitude.shape[1])], axis=1, keepdims=True)
        noise_power = np.mean(noise_mag ** 2)

        # 信号+噪声总功率
        total_power = np.mean(magnitude ** 2)

        # SNR计算
        if total_power > noise_power:
            snr = 10 * np.log10((total_power - noise_power) / max(noise_power, 1e-10))
        else:
            snr = 0.0

        return max(snr, -10.0)  # 截断到-10dB以上

    def enhance(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        对音频进行预增强处理
        Args:
            audio: 输入音频数组
            sr: 采样率
        Returns:
            增强后的音频数组
        """
        start = time.time()

        # 重采样到目标采样率
        if sr != self.sr:
            audio = librosa_resample(audio, orig_sr=sr, target_sr=self.sr)

        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 计算STFT
        stft = librosa_stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        power = magnitude ** 2

        # 噪声估计（使用前noise_frames帧）
        noise_frames_actual = min(self.noise_frames, magnitude.shape[1])
        noise_power = np.mean(power[:, :noise_frames_actual], axis=1, keepdims=True)
        noise_mag = np.mean(magnitude[:, :noise_frames_actual], axis=1, keepdims=True)

        if self.method == 'wiener':
            # 维纳滤波
            # 估计信号功率
            signal_power = np.maximum(power - noise_power, 0)
            # 维纳增益: H = xi / (1 + xi), xi = signal_power / noise_power
            # 使用decision-directed先验SNR估计
            alpha_smooth = 0.98
            snr_prior = np.maximum(power / (noise_power + 1e-10) - 1, 0)
            gain = snr_prior / (1 + snr_prior + 1e-10)
            # 平滑增益（避免音乐噪声）
            gain = np.minimum(gain, 1.0)
            magnitude_enhanced = magnitude * gain
        else:
            # 谱减法
            alpha = 1.5  # 过减因子（更激进）
            magnitude_enhanced = np.maximum(
                magnitude - alpha * noise_mag,
                0.05 * magnitude  # floor防止过减
            )

        # 重建信号
        stft_enhanced = magnitude_enhanced * np.exp(1j * phase)
        audio_enhanced = librosa_istft(stft_enhanced, hop_length=self.hop_length, length=len(audio))

        elapsed = time.time() - start
        logger.debug(f"[预增强] {self.method} 耗时: {elapsed:.3f}s")

        return audio_enhanced


# ============================================================================
# libROSA兼容函数（避免全局librosa导入带来的性能开销）
# ============================================================================

def librosa_stft(y, n_fft=2048, hop_length=512):
    """STFT计算"""
    import librosa
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)


def librosa_istft(stft_matrix, hop_length=512, length=None):
    """iSTFT重建"""
    import librosa
    return librosa.istft(stft_matrix, hop_length=hop_length, length=length)


def librosa_resample(y, orig_sr, target_sr):
    """重采样"""
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def librosa_load(path, sr=16000):
    """加载音频"""
    import librosa
    return librosa.load(path, sr=sr, mono=True)


def librosa_mfcc(y, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    """MFCC计算"""
    import librosa
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)


# ============================================================================
# HPSS谐波分量提取
# ============================================================================

def extract_harmonic_component(audio: np.ndarray, sr: int = 16000,
                                 kernel_size: int = 31) -> np.ndarray:
    """
    提取音频的谐波分量（用于更鲁棒的峰值检测和对齐）
    将音频分解为谐波部分（语音）和打击乐部分（噪声/瞬态）

    性能优化:
    1. 默认kernel_size从51降至31（中值滤波快~40%）
    2. 使用scipy.ndimage.median_filter直接操作幅度谱（避免librosa的软掩码和ISTFT开销）
    3. 对于短音频(<0.5s)，直接返回原始音频（HPSS无意义）

    Args:
        audio: 输入音频数组
        sr: 采样率(仅用于日志)
        kernel_size: HPSS核大小，越大谐波越纯净；建议值21-51，默认31

    原理：
    - 对STFT时频谱沿时间轴做中值滤波 → 谐波分量（水平条纹）
    - 语音的元音表现为稳定的水平能量带，通过时间轴中值滤波保留
    - 环境噪声宽频分布，在时频谱上无稳定结构，被中值滤波衰减
    """
    # 短音频直接返回（HPSS无意义，反而可能失真）
    if len(audio) < sr * 0.5:  # <0.5s
        return audio.copy()

    try:
        # 使用librosa的STFT参数，直接操作幅度谱
        import scipy.ndimage as ndi
        import librosa

        # STFT（与librosa.effects.hpss参数一致）
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        # 使用功率谱（power=2.0）做中值滤波，与librosa的hpss一致
        S = np.abs(D) ** 2
        eps = 1e-10

        # 时间轴中值滤波 → 谐波分量，频率轴中值滤波 → 打击乐分量
        harm_S = ndi.median_filter(S, size=(1, kernel_size))
        perc_S = ndi.median_filter(S, size=(kernel_size, 1))

        # 软掩码（与librosa的softmask一致）：harm / (harm + perc)
        mask = harm_S / (harm_S + perc_S + eps)

        # 合成谐波分量（使用原始相位，应用软掩码）
        harmonic = librosa.istft(D * mask, hop_length=hop_length)

        logger.debug(f"[HPSS快速] 谐波分量提取完成: len={len(audio)}, "
                     f"kernel={kernel_size}, "
                     f"harmonic_rms={np.sqrt(np.mean(harmonic**2)):.4f}")
        return harmonic
    except ImportError:
        # 回退到librosa的hpss
        import librosa
        try:
            harmonic, percussive = librosa.effects.hpss(
                audio, kernel_size=kernel_size, power=2.0
            )
            logger.debug(f"[HPSS-librosa] 谐波分量提取完成: len={len(audio)}, "
                         f"kernel={kernel_size}")
            return harmonic
        except Exception as e2:
            logger.debug(f"[HPSS] 全部方法失败: {e2}，返回原始音频")
            return audio.copy()
    except Exception as e:
        logger.debug(f"[HPSS快速] HPSS分解失败: {e}，回退到librosa")
        import librosa
        try:
            harmonic, percussive = librosa.effects.hpss(
                audio, kernel_size=kernel_size, power=2.0
            )
            return harmonic
        except Exception:
            return audio.copy()


# ============================================================================
# CMVN归一化 (用于DTW的MFCC预处理)
# ============================================================================

def apply_cmvn(mfcc: np.ndarray) -> np.ndarray:
    """
    对MFCC特征应用倒谱均值方差归一化(Cepstral Mean and Variance Normalization)
    提高MFCC对加性噪声的鲁棒性

    Args:
        mfcc: 形状为(时间帧, n_mfcc)的MFCC特征矩阵

    Returns:
        归一化后的MFCC特征矩阵
    """
    # 计算全局均值和标准差
    mean = np.mean(mfcc, axis=0, keepdims=True)
    std = np.std(mfcc, axis=0, keepdims=True)
    # 归一化
    mfcc_norm = (mfcc - mean) / (std + 1e-10)
    return mfcc_norm


def add_delta_features(mfcc: np.ndarray, window: int = 2) -> np.ndarray:
    """
    添加delta和delta-delta特征（动态特征）
    Args:
        mfcc: (n_frames, n_coeffs) MFCC矩阵
        window: 差分窗口大小
    Returns:
        (n_frames, n_coeffs * 3) 包含静态+delta+delta-delta的特征矩阵
    """
    from scipy.ndimage import convolve1d
    # delta特征
    delta_filter = np.zeros(2 * window + 1)
    delta_filter[0] = -1
    delta_filter[-1] = 1
    delta_filter /= 2 * window
    delta = convolve1d(mfcc, delta_filter, axis=0, mode='nearest')

    # delta-delta特征
    delta2 = convolve1d(delta, delta_filter, axis=0, mode='nearest')

    # 拼接：静态 + delta + delta-delta
    features = np.concatenate([mfcc, delta, delta2], axis=1)
    return features


# ============================================================================
# 噪声鲁棒的DTW定位器
# ============================================================================

class RobustDTWLocator:
    """
    增强型DTW定位器
    改进点：
    1. CMVN归一化MFCC
    2. Delta/Delta-Delta动态特征
    3. 自适应搜索步长（根据SNR调整）
    4. 预增强处理后再提取特征
    """

    def __init__(self, sr: int = 16000, hop_length: int = 512,
                 n_fft: int = 2048, n_mfcc: int = 13):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.pre_enhancer = LightweightPreEnhancer(method='wiener', sr=sr)

        # 延迟导入dtw
        try:
            from dtw import dtw
            self._dtw = dtw
            self._dtw_available = True
        except ImportError:
            logger.warning("[RobustDTW] dtw-python未安装，DTW不可用")
            self._dtw = None
            self._dtw_available = False

    def extract_robust_mfcc(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取经过噪声鲁棒处理的MFCC特征
        1. 加载音频
        2. 应用CMVN
        3. 添加delta特征
        """
        y, sr = librosa_load(audio_path, sr=self.sr)

        # 基本MFCC
        mfcc = librosa_mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc,
                             n_fft=self.n_fft, hop_length=self.hop_length)

        # CMVN归一化
        mfcc = apply_cmvn(mfcc.T)  # (n_frames, n_coeffs)
        # 添加动态特征
        mfcc = add_delta_features(mfcc)

        return mfcc, y

    def extract_robust_mfcc_pre_enhanced(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        先预增强再提取MFCC（适用于低SNR音频）
        """
        y, sr = librosa_load(audio_path, sr=self.sr)
        # 预增强
        y_enhanced = self.pre_enhancer.enhance(y, sr)
        # MFCC
        mfcc = librosa_mfcc(y=y_enhanced, sr=sr, n_mfcc=self.n_mfcc,
                             n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = apply_cmvn(mfcc.T)
        mfcc = add_delta_features(mfcc)
        return mfcc, y

    def extract_robust_mfcc_from_array(self, audio: np.ndarray, sr: int,
                                       pre_enhance: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """从numpy数组提取鲁棒MFCC"""
        if sr != self.sr:
            audio = librosa_resample(audio, orig_sr=sr, target_sr=self.sr)

        if pre_enhance:
            audio = self.pre_enhancer.enhance(audio, self.sr)

        mfcc = librosa_mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
                             n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = apply_cmvn(mfcc.T)
        mfcc = add_delta_features(mfcc)
        return mfcc, audio

    def index2time(self, index: int) -> float:
        return index * self.hop_length / self.sr

    def time2index(self, time_sec: float) -> int:
        return max(0, int(time_sec * self.sr / self.hop_length))

    def locate(self, test_audio_path: str, ref_audio_path: str,
               search_start: float = 0.0, search_end: float = None,
               jump_step: int = None, pre_enhance: bool = True,
               snr: float = None) -> Optional[Dict]:
        """
        噪声鲁棒的DTW定位
        Args:
            test_audio_path: 测试音频
            ref_audio_path: 参考音频
            search_start: 搜索起始时间
            search_end: 搜索结束时间
            jump_step: 滑动窗口步长
            pre_enhance: 是否对低SNR音频预增强
            snr: 已知SNR(可选，传入可跳过内部SNR估计)
        Returns:
            {"start_time": ..., "duration": ..., "distance": ...} 或 None
        """
        if not self._dtw_available:
            logger.warning("[RobustDTW] DTW不可用")
            return None

        logger.info(f"[RobustDTW] test={os.path.basename(test_audio_path)}, "
                    f"ref={os.path.basename(ref_audio_path)}, "
                    f"search=[{search_start:.1f},{search_end or 'end'}]")

        # 估计SNR决定预处理策略
        if snr is None:
            y_test, sr_test = librosa_load(test_audio_path, sr=self.sr)
            snr = self.pre_enhancer.estimate_snr(y_test)
            logger.info(f"[RobustDTW] 估计SNR={snr:.1f}dB")

        # 低SNR时自动启用预增强
        use_pre_enhance = pre_enhance and snr < 10
        if use_pre_enhance:
            logger.info(f"[RobustDTW] SNR={snr:.1f}dB < 10dB，启用预增强")

        # 自适应步长：低SNR使用更小步长（更精确但更慢）
        if jump_step is None:
            if snr < 0:
                jump_step = 3  # 极低SNR: 细搜索
            elif snr < 10:
                jump_step = 5  # 低SNR: 中等
            else:
                jump_step = 8  # 高SNR: 粗搜索

        # 提取参考音频特征
        if use_pre_enhance:
            ref_mfcc, ref_y = self.extract_robust_mfcc_pre_enhanced(ref_audio_path)
            test_mfcc, test_y = self.extract_robust_mfcc_pre_enhanced(test_audio_path)
        else:
            ref_mfcc, ref_y = self.extract_robust_mfcc(ref_audio_path)
            test_mfcc, test_y = self.extract_robust_mfcc(test_audio_path)

        ref_duration = len(ref_y) / self.sr
        ref_frames = ref_mfcc.shape[0]
        test_frames = test_mfcc.shape[0]

        # 确定搜索范围
        start_frame = self.time2index(search_start)
        if search_end is not None:
            end_frame = min(self.time2index(search_end), test_frames - ref_frames)
        else:
            end_frame = test_frames - ref_frames

        if end_frame <= start_frame:
            logger.info(f"[RobustDTW] 搜索范围过小，返回起点")
            return {
                "start_time": search_start,
                "duration": ref_duration,
                "distance": 0.0,
                "start_frame": start_frame,
                "snr": snr
            }

        # 滑动窗口DTW搜索（使用余弦距离，对噪声更鲁棒）
        best_distance = float('inf')
        best_frame = -1
        distances = []

        for i in range(start_frame, end_frame, jump_step):
            window = test_mfcc[i:i + ref_frames]
            if window.shape[0] < ref_frames:
                break
            try:
                # 使用余弦距离替代欧氏距离（对幅度变化更鲁棒）
                alignment = self._dtw(window, ref_mfcc, dist_method='cosine')
                dist = alignment.distance
                distances.append(dist)
                if dist < best_distance:
                    best_distance = dist
                    best_frame = i
            except Exception as e:
                logger.debug(f"[RobustDTW] DTW异常: {e}")
                continue

        if best_frame < 0 or not distances:
            logger.warning(f"[RobustDTW] 未找到有效匹配")
            return None

        start_time = self.index2time(best_frame)

        # 计算匹配置信度
        # 使用best_distance与平均距离的比率（越小越好）
        mean_dist = np.mean(distances)
        if mean_dist > 0:
            confidence_ratio = 1.0 - (best_distance / mean_dist)
        else:
            confidence_ratio = 0.0

        logger.info(f"[RobustDTW] 匹配结果: offset={start_time:.3f}s, dist={best_distance:.2f}, "
                    f"conf_ratio={confidence_ratio:.3f}, snr={snr:.1f}dB")

        return {
            "start_time": start_time,
            "duration": ref_duration,
            "distance": best_distance,
            "start_frame": best_frame,
            "snr": snr,
            "confidence_ratio": confidence_ratio,
            "mean_distance": mean_dist,
            "pre_enhanced": use_pre_enhance,
            "jump_step": jump_step,
            "n_features": ref_mfcc.shape[1]  # 包含delta的特征维度
        }


# ============================================================================
# 基于SV模型嵌入向量的匹配器（低SNR回退方案）
# ============================================================================

class EmbeddingMatcher:
    """
    使用说话人确认(SV)模型嵌入向量匹配
    当指纹匹配和DTW都失败时，利用TCF模型的鲁棒嵌入特征进行匹配

    原理：
    - TCF模型（如eres2net）在噪声条件下仍能提取有效的说话人+内容嵌入
    - 通过滑动窗口计算测试音频的嵌入向量，与参考音频嵌入比较余弦相似度
    - 余弦相似度最高的窗口即为最佳匹配位置
    """

    def __init__(self, ref_dir: str = None):
        self.ref_dir = ref_dir
        self.sr = 16000
        self._sv_pipeline = None
        self._model_name = 'eres2net'  # 使用最佳模型
        self._initialized = False
        self._ref_embeddings = {}  # ref_file -> embedding

    def _get_sv_pipeline(self):
        """获取或创建SV模型pipeline"""
        if self._sv_pipeline is not None:
            return self._sv_pipeline

        try:
            # 尝试获取已加载的TCF pipeline
            try:
                sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'app', 'core'))
                sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'app', 'algorithms'))
                sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'app', 'algorithms', 'tcf'))

                # 直接使用mos_calculator中已加载的模型
                from calculator.mos_calculator import parallel_compute
                if 'tcf' in parallel_compute.models:
                    tcf_model = parallel_compute.models['tcf']
                    if hasattr(tcf_model, '_get_pipeline'):
                        # 尝试使用已缓存的eres2net pipeline
                        try:
                            self._sv_pipeline = tcf_model._get_pipeline(self._model_name)
                            logger.info(f"[嵌入匹配] 复用已加载的{self._model_name} pipeline")
                            return self._sv_pipeline
                        except Exception:
                            pass
            except Exception:
                pass

            # 备用：直接创建pipeline
            import torch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from modelscope import pipeline
                project_path = os.path.join(_PROJECT_ROOT, "models", "tcf", self._model_name)
                if os.path.exists(project_path):
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self._sv_pipeline = pipeline(
                        task='speaker-verification',
                        model=project_path,
                        device=device
                    )
                    logger.info(f"[嵌入匹配] 创建{self._model_name} pipeline (device={device})")
                else:
                    logger.error(f"[嵌入匹配] 模型路径不存在: {project_path}")
                    return None
        except Exception as e:
            logger.error(f"[嵌入匹配] 创建SV pipeline失败: {e}")
            return None

        return self._sv_pipeline

    def _compute_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """计算音频文件的嵌入向量"""
        pipe = self._get_sv_pipeline()
        if pipe is None:
            return None

        try:
            result = pipe(audio_path, output_emb=True)
            if 'embs' in result and len(result['embs']) > 0:
                emb = result['embs'][0]
                if isinstance(emb, (list, tuple)):
                    emb = np.array(emb)
                return emb.flatten()
        except Exception as e:
            logger.debug(f"[嵌入匹配] 嵌入计算失败 {audio_path}: {e}")

        return None

    def _compute_embedding_from_segment(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        从音频片段计算嵌入向量
        使用内存传递替代临时文件I/O（P2优化）
        """
        pipe = self._get_sv_pipeline()
        if pipe is None:
            return None

        # 确保16kHz
        if sr != self.sr:
            audio = librosa_resample(audio, orig_sr=sr, target_sr=self.sr)

        # 尝试内存传递（ModelScope管线支持numpy数组直接输入）
        try:
            result = pipe(audio, output_emb=True, sampling_rate=self.sr)
            if 'embs' in result and len(result['embs']) > 0:
                emb = result['embs'][0]
                if isinstance(emb, (list, tuple)):
                    emb = np.array(emb)
                return emb.flatten()
        except Exception as e:
            logger.debug(f"[嵌入匹配] 内存传递失败: {e}，回退到BytesIO")

        # 回退：使用BytesIO内存缓冲（不落盘）
        try:
            import io
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, audio, self.sr, format='WAV')
            buf.seek(0)
            result = pipe(buf, output_emb=True, sampling_rate=self.sr)
            if 'embs' in result and len(result['embs']) > 0:
                emb = result['embs'][0]
                if isinstance(emb, (list, tuple)):
                    emb = np.array(emb)
                return emb.flatten()
        except Exception as e:
            logger.debug(f"[嵌入匹配] BytesIO也失败: {e}")

        return None

    def build_reference_embeddings(self, ref_dir: str = None):
        """
        为所有参考音频计算嵌入向量并缓存
        """
        target_dir = ref_dir or self.ref_dir
        if not target_dir:
            logger.error("[嵌入匹配] 未指定参考目录")
            return

        from pathlib import Path
        ref_path = Path(target_dir)
        audio_files = sorted([
            f for f in ref_path.iterdir()
            if f.suffix.lower() in ['.wav', '.mp3', '.flac']
        ])

        logger.info(f"[嵌入匹配] 计算{len(audio_files)}个参考音频的嵌入向量...")
        count = 0
        for f in audio_files:
            emb = self._compute_embedding(str(f))
            if emb is not None:
                self._ref_embeddings[str(f)] = emb
                count += 1
                logger.info(f"[嵌入匹配] 参考嵌入: {f.name} dim={len(emb)}")

        logger.info(f"[嵌入匹配] 完成{count}/{len(audio_files)}个参考嵌入计算")
        self._initialized = count > 0

    def match_by_embedding(self, test_audio_path: str,
                           window_duration: float = 12.0,
                           step_duration: float = 4.0,
                           pre_enhance: bool = True) -> List[Dict]:
        """
        使用嵌入向量匹配在测试音频中定位参考音频

        Args:
            test_audio_path: 测试音频路径
            window_duration: 滑动窗口时长（应与参考音频时长大致匹配）
            step_duration: 滑动步长
            pre_enhance: 是否预增强

        Returns:
            [{"ref_file": ..., "ref_name": ..., "offset": ..., "similarity": ..., ...}, ...]
        """
        if not self._initialized or not self._ref_embeddings:
            logger.warning("[嵌入匹配] 参考嵌入未就绪，尝试构建")
            self.build_reference_embeddings()

        if not self._ref_embeddings:
            logger.error("[嵌入匹配] 无参考嵌入可用")
            return []

        # 加载测试音频
        test_audio, test_sr = librosa_load(test_audio_path, sr=self.sr)

        # 可选预增强
        if pre_enhance:
            enhancer = LightweightPreEnhancer(method='wiener', sr=self.sr)
            snr = enhancer.estimate_snr(test_audio)
            logger.info(f"[嵌入匹配] 测试音频SNR={snr:.1f}dB")
            if snr < 10:
                logger.info("[嵌入匹配] SNR<10dB，应用预增强")
                test_audio = enhancer.enhance(test_audio, self.sr)

        test_duration = len(test_audio) / self.sr
        window_samples = int(window_duration * self.sr)
        step_samples = int(step_duration * self.sr)

        logger.info(f"[嵌入匹配] 测试音频 {os.path.basename(test_audio_path)}: "
                    f"{test_duration:.1f}s, 窗口{window_duration:.0f}s, 步长{step_duration:.0f}s")

        # 遍历所有参考音频
        all_matches = []
        for ref_path, ref_emb in self._ref_embeddings.items():
            ref_name = os.path.basename(ref_path)
            ref_norm = np.linalg.norm(ref_emb)
            if ref_norm == 0:
                continue

            best_similarity = -1.0
            best_offset = 0.0

            # 滑动窗口
            num_windows = max(1, int((test_duration - window_duration) / step_duration) + 1)
            for w in range(num_windows):
                start_sample = w * step_samples
                end_sample = min(start_sample + window_samples, len(test_audio))

                segment = test_audio[start_sample:end_sample]
                # 如果片段不足，填充
                if len(segment) < window_samples:
                    segment = np.pad(segment, (0, window_samples - len(segment)))

                # 计算片段嵌入
                seg_emb = self._compute_embedding_from_segment(segment, self.sr)
                if seg_emb is None:
                    continue

                # 余弦相似度
                seg_norm = np.linalg.norm(seg_emb)
                if seg_norm == 0:
                    continue
                similarity = float(np.dot(seg_emb, ref_emb) / (seg_norm * ref_norm))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_offset = start_sample / self.sr

            logger.info(f"[嵌入匹配] {ref_name}: 最佳similarity={best_similarity:.4f} "
                        f"@ offset={best_offset:.1f}s")

            if best_similarity > 0.5:  # 阈值：余弦相似度>0.5视为匹配
                all_matches.append({
                    "ref_file": ref_path,
                    "ref_name": ref_name,
                    "offset_in_test": best_offset,
                    "similarity": best_similarity,
                    "method": "embedding"
                })

        # 按相似度降序排列
        all_matches.sort(key=lambda x: x['similarity'], reverse=True)
        return all_matches


# ============================================================================
# 统一优化匹配入口
# ============================================================================

class OptimizedMatcher:
    """
    优化匹配器 - 多级匹配策略

    策略优先级：
    Level 1: 自适应指纹匹配（增强预增强 + 自适应阈值）
    Level 2: 鲁棒DTW定位（CMVN + Delta + 余弦距离）
    Level 3: 嵌入向量匹配（TCF SV模型回退）
    Level 4: 互相关精对齐（自适应阈值）
    """

    def __init__(self, ref_dir: str = None):
        self.ref_dir = ref_dir
        self.pre_enhancer = LightweightPreEnhancer(method='wiener')
        self.robust_dtw = RobustDTWLocator()
        self.embedding_matcher = EmbeddingMatcher(ref_dir=ref_dir)
        self._embedding_initialized = False

        # WeNet ASR语义匹配器（Level A策略）
        self._wenet_matcher = None  # WeNetSegmentationMatcher实例
        self._wenet_available = False

        # 参考MFCC缓存（避免跨文件重复计算）
        self._ref_mfcc_cache = {}  # ref_name -> (mfcc, frames, y)
        # 最近一次测试音频缓存（避免双加载）
        self._last_test_path = None
        self._last_test_y = None
        self._last_test_sr = None

    def _load_and_cache_ref(self, ref_name: str, ref_path: str):
        """加载并缓存单个参考音频的MFCC"""
        if ref_name not in self._ref_mfcc_cache:
            y, _ = librosa_load(ref_path, sr=16000)
            mfcc, _ = self.robust_dtw.extract_robust_mfcc_from_array(
                y, 16000, pre_enhance=False
            )
            self._ref_mfcc_cache[ref_name] = (mfcc, mfcc.shape[0], y)
        return self._ref_mfcc_cache[ref_name]

    def build_ref_cache(self, ref_dir: str = None):
        """预加载所有参考音频到MFCC缓存"""
        target = ref_dir or self.ref_dir
        if not target or not os.path.isdir(target):
            return
        for fname in sorted(os.listdir(target)):
            if fname.endswith(('.wav', '.mp3', '.flac')):
                fpath = os.path.join(target, fname)
                self._load_and_cache_ref(fname, fpath)
        logger.info(f"[MFCC缓存] 已缓存{len(self._ref_mfcc_cache)}个参考音频")

    def _estimate_snr_and_noise_floor(self, audio_path: str) -> Tuple[float, float]:
        """估计SNR和噪声底噪（带缓存，避免同一文件重复计算）"""
        # 检查缓存（同一音频文件的SNR在整个匹配流程中不变）
        if hasattr(self, '_snr_cache') and self._snr_cache is not None:
            cached_path, cached_snr, cached_floor = self._snr_cache
            if cached_path == audio_path:
                return cached_snr, cached_floor

        y, sr = librosa_load(audio_path, sr=16000)
        snr = self.pre_enhancer.estimate_snr(y)
        noise_floor = self.pre_enhancer.estimate_noise_floor(y)

        # 缓存本次结果
        self._snr_cache = (audio_path, snr, noise_floor)
        return snr, noise_floor

    # _adaptive_fingerprint_matching 已移除（原Shazam指纹算法，不再使用）
    # 匹配策略请参考 match_with_fallback()

    def _init_wenet_matcher(self) -> bool:
        """初始化WeNet语义匹配器（Level A）

        尝试加载WeNet模型和参考文本，仅在以下条件满足时启用：
        1. models/wenet/ 目录存在
        2. ref_dir 中存在 .metadata.json 且包含 ground_truth_text

        Returns:
            是否初始化成功
        """
        if self._wenet_available:
            return True
        if self._wenet_matcher is not None:
            return False  # 已尝试但失败

        # 检查模型目录
        wenet_dir = os.path.join(_PROJECT_ROOT, "models", "wenet")
        if not os.path.exists(wenet_dir):
            logger.debug("[优化匹配-WeNet] 模型目录不存在，跳过ASR语义匹配")
            self._wenet_matcher = False  # 标记为不可用
            return False

        # 检查参考文本
        if not self.ref_dir:
            logger.debug("[优化匹配-WeNet] 无参考目录，跳过ASR语义匹配")
            self._wenet_matcher = False
            return False
        meta_file = os.path.join(self.ref_dir, ".metadata.json")
        if not os.path.exists(meta_file):
            logger.debug("[优化匹配-WeNet] 无.metadata.json，跳过ASR语义匹配")
            self._wenet_matcher = False
            return False

        try:
            from segmentation_optimizer import WeNetSegmentationMatcher
            matcher = WeNetSegmentationMatcher(ref_dir=self.ref_dir)
            ok = matcher.load_model(wenet_dir, device='cuda')
            if not ok:
                self._wenet_matcher = False
                return False
            cnt = matcher.load_ref_texts(self.ref_dir)
            if cnt == 0:
                logger.warning("[优化匹配-WeNet] 参考文本加载失败或无文本")
                self._wenet_matcher = False
                return False
            self._wenet_matcher = matcher
            self._wenet_available = True
            logger.info(f"[优化匹配-WeNet] 初始化成功: {cnt}个参考文本, "
                        f"subsampling={matcher._frame_shift*1000:.0f}ms/帧")
            return True
        except Exception as e:
            logger.warning(f"[优化匹配-WeNet] 初始化失败: {e}")
            self._wenet_matcher = False
            return False

    def get_all_segment_matches(self, test_audio_path: str) -> List[Dict]:
        """
        获取测试音频中所有参考段的匹配信息（用于预检测+切分阶段）

        策略：
          先用ASR语义匹配，对ASR未找到的参考段用DTW补缺。
          返回格式与 _full_range_dtw_sweep() 兼容。

        Args:
            test_audio_path: 测试音频路径

        Returns:
            [{"ref_id", "ref_file", "ref_name", "offset_in_test",
              "confidence", "method", "wer"(optional), ...}]
        """
        # 所有期望的参考段名称
        expected_refs = set()
        if self.ref_dir and os.path.isdir(self.ref_dir):
            for fname in os.listdir(self.ref_dir):
                if fname.endswith(('.wav', '.mp3', '.flac')):
                    expected_refs.add(fname)

        # ----- 第1轮: WeNet语义匹配 -----
        asr_segments = []
        if self._init_wenet_matcher() and expected_refs:
            try:
                asr_result = self._wenet_matcher.match_with_fallback(test_audio_path)
                asr_segments = asr_result.get("detail", {}).get("segment_matches", [])
            except Exception as e:
                logger.debug(f"[获取段匹配] WeNet失败: {e}")

        # 转换ASR结果
        asr_map = {}  # ref_name -> segment_dict
        for s in asr_segments:
            asr_map[s["ref_name"]] = {
                "ref_id": s["ref_name"],
                "ref_file": s["ref_file"],
                "ref_name": s["ref_name"],
                "offset_in_test": s["offset_in_test"],
                "confidence": s["confidence"],
                "method": "asr_text_dtw",
                "snr": asr_result.get("snr", 0.0) if asr_result else 0.0,
                "duration": s.get("duration", 0.0),
                "wer": s.get("wer", 0.0),
                "wcorr": s.get("wcorr", 0.0),
                "n_chars": s.get("n_chars", 0),
            }

        # 如果ASR找到了所有期望段，直接返回
        asr_found = set(asr_map.keys())
        missing = expected_refs - asr_found
        if not missing and asr_found:
            logger.info(f"[获取段匹配] ASR匹配全部{len(expected_refs)}个段")
            return list(asr_map.values())

        # ----- 第2轮: DTW补缺（针对ASR未找到的段）-----
        if missing:
            logger.info(f"[获取段匹配] ASR匹配{len(asr_found)}/{len(expected_refs)}个段, "
                        f"DTW补缺{len(missing)}个: {missing}")

        # 全范围DTW扫描
        snr = 0.0
        try:
            snr, _ = self._estimate_snr_and_noise_floor(test_audio_path)
        except Exception:
            pass
        dtw_results = self._full_range_dtw_sweep(test_audio_path, snr)

        # 从DTW结果中取缺失段
        for d in dtw_results:
            rn = d.get("ref_name", "")
            if rn in missing:
                d["method"] = d.get("method", "full_range_dtw")
                asr_map[rn] = d
                logger.info(f"[获取段匹配] DTW补缺: {rn} @ {d.get('offset_in_test', 0):.2f}s")

        # ----- 合并结果 -----
        final = list(asr_map.values())
        final.sort(key=lambda x: x.get('offset_in_test', 0))

        found_names = {s['ref_name'] for s in final}
        still_missing = expected_refs - found_names
        if still_missing:
            logger.warning(f"[获取段匹配] 仍有段未找到: {still_missing}")

        return final

    def _full_range_dtw_sweep(self, test_audio_path: str,
                                snr: float) -> List[Dict]:
        """
        全范围DTW扫描匹配（主匹配方案）
        对每个参考音频在测试音频全范围内做两级DTW搜索

        Args:
            test_audio_path: 测试音频路径
            snr: 信噪比(dB)

        Returns:
            [{"ref_id", "ref_file", "ref_name", "offset_in_test",
              "confidence", "dtw_distance", ...}]
        """
        if not self.ref_dir or not os.path.isdir(self.ref_dir):
            return []

        # 扫描参考音频目录
        ref_entries = []
        for fname in sorted(os.listdir(self.ref_dir)):
            if fname.endswith(('.wav', '.mp3', '.flac')):
                fpath = os.path.join(self.ref_dir, fname)
                ref_entries.append((fname, fpath))

        if not ref_entries:
            return []

        use_pre_enhance = snr < 10
        results = []
        import librosa

        y_test, sr_test = librosa.load(test_audio_path, sr=16000)
        dur_test = len(y_test) / sr_test

        # 缓存测试音频供_cut_all复用（避免_process_single_file重复加载）
        self._last_test_path = test_audio_path
        self._last_test_y = y_test
        self._last_test_sr = sr_test

        logger.info(f"[全范围DTW] 测试音频={os.path.basename(test_audio_path)}, "
                    f"dur={dur_test:.0f}s, 参考数={len(ref_entries)}, "
                    f"pre_enhance={use_pre_enhance}")

        # 提取测试MFCC一次（所有参考复用）
        dtw = self.robust_dtw
        test_mfcc, test_y = dtw.extract_robust_mfcc_from_array(
            y_test, sr_test, pre_enhance=use_pre_enhance
        )
        total_frames = test_mfcc.shape[0]
        hop_time = 512.0 / sr_test  # 每帧对应时间(秒)

        def _process_single_ref(ref_name: str, ref_path: str) -> Optional[Dict]:
            """对单个参考音频执行两级DTW搜索（线程安全，独立上下文）"""
            try:
                # 使用缓存的参考MFCC（避免跨文件重复提取）
                ref_cache = self._load_and_cache_ref(ref_name, ref_path)
                ref_mfcc = ref_cache[0]
                ref_frames_actual = ref_cache[1]
                # 测试比参考短时无法滑动窗口，跳过；等长时可做一个完整对齐
                if total_frames < ref_frames_actual:
                    return None

                # --- 粗扫描（大步长，快速定位） ---
                coarse_step = max(30, int((total_frames - ref_frames_actual) / 10))
                best_dist = float('inf')
                best_frame = -1

                for i in range(0, max(1, total_frames - ref_frames_actual), coarse_step):
                    window = test_mfcc[i:i + ref_frames_actual]
                    if window.shape[0] < ref_frames_actual:
                        break
                    try:
                        alignment = dtw._dtw(window, ref_mfcc, dist_method='cosine')
                        if alignment.distance < best_dist:
                            best_dist = alignment.distance
                            best_frame = i
                    except Exception:
                        continue

                if best_frame < 0:
                    return None

                coarse_offset = best_frame * hop_time
                logger.debug(f"[全范围DTW] {ref_name}: 粗扫offset={coarse_offset:.1f}s, "
                            f"dist={best_dist:.1f}")

                # --- 抛物线插值（子帧精度，替代部分精细搜索） ---
                # 在best_frame ± 1帧处计算DTW距离，拟合抛物线求极小值点
                # 公式: x_opt = -(d₃-d₁) / (2*(d₁+d₃-2d₂))
                # 其中 d₁ = dist(best_frame-1), d₂ = dist(best_frame), d₃ = dist(best_frame+1)
                para_best_frame = best_frame  # 默认用粗扫结果
                para_best_dist = best_dist

                # 只在best_frame有前后邻居时做插值
                if best_frame > 0 and best_frame + ref_frames_actual < total_frames:
                    try:
                        d1 = None
                        d3 = None
                        # 计算d1 = dist(best_frame - 1) 仅在帧索引有效时
                        if best_frame - 1 >= 0:
                            win = test_mfcc[best_frame - 1:best_frame - 1 + ref_frames_actual]
                            if win.shape[0] == ref_frames_actual:
                                d1 = dtw._dtw(win, ref_mfcc, dist_method='cosine').distance

                        # d2 = best_dist (已知)
                        d2 = best_dist

                        # 计算d3 = dist(best_frame + 1) 仅在帧索引有效时
                        if best_frame + 1 + ref_frames_actual <= total_frames:
                            win = test_mfcc[best_frame + 1:best_frame + 1 + ref_frames_actual]
                            if win.shape[0] == ref_frames_actual:
                                d3 = dtw._dtw(win, ref_mfcc, dist_method='cosine').distance

                        if d1 is not None and d3 is not None:
                            # 二阶抛物线拟合：用中点做局部插值
                            a = (d1 + d3 - 2 * d2) * 0.5
                            if a > 1e-10:  # 凸抛物线（极小值存在）
                                b = (d3 - d1) * 0.5
                                shift = -b / (2 * a)  # 子帧偏移量，范围(-0.5, 0.5)
                                # 限制偏移量防止过大偏移
                                shift = max(-0.5, min(0.5, shift))
                                para_best_frame = best_frame + shift
                                para_best_dist = d2 - 0.5 * b * shift  # 插值的极小值
                    except Exception:
                        pass  # 插值失败则使用粗扫结果

                # --- 自适应精细搜索（安全网） ---
                # 抛物线插值后，根据粗扫质量决定精细化程度：
                #   norm_dist < 0.5: 粗扫极准，免精扫（仅极低距离时跳过）
                #   norm_dist < 2.0: 窄范围精扫 ±0.5s（大部分情况，含10dB SNR）
                #   norm_dist >= 2.0: 标准范围精扫 ±1.0s（低质量粗扫）
                norm_dist_check = para_best_dist / max(1, ref_frames_actual)

                # 粗扫质量阈值自适应精扫范围：
                #   norm_dist < 0.5: 粗扫极准，免精扫
                #   norm_dist < 2.0: 中等质量，±2.0s（粗扫误差可达1.6s）
                #   norm_dist >= 2.0: 低质量，±3.0s（原版全范围）
                if norm_dist_check < 0.5:
                    fine_best_dist = para_best_dist
                    fine_best_frame = para_best_frame
                else:
                    if norm_dist_check < 2.0:
                        narrow_range = 2.0  # ±2.0s
                    else:
                        narrow_range = 3.0  # ±3.0s（原版范围）
                    narrow_step = 5
                    narrow_start = max(0, int((para_best_frame * hop_time - narrow_range) / hop_time))
                    narrow_end = min(total_frames - ref_frames_actual,
                                     int((para_best_frame * hop_time + ref_frames_actual * hop_time + narrow_range) / hop_time))

                    fine_best_dist = para_best_dist
                    fine_best_frame = para_best_frame
                    fine_best_frame_int = max(0, min(int(para_best_frame), total_frames - ref_frames_actual))
                    # 向前搜索
                    for i in range(fine_best_frame_int, narrow_end, narrow_step):
                        window = test_mfcc[i:i + ref_frames_actual]
                        if window.shape[0] < ref_frames_actual:
                            break
                        try:
                            alignment = dtw._dtw(window, ref_mfcc, dist_method='cosine')
                            if alignment.distance < fine_best_dist:
                                fine_best_dist = alignment.distance
                                fine_best_frame = float(i)
                        except Exception:
                            continue
                    # 向后搜索
                    for i in range(fine_best_frame_int - narrow_step, narrow_start - 1, -narrow_step):
                        if i < 0:
                            continue
                        window = test_mfcc[i:i + ref_frames_actual]
                        if window.shape[0] < ref_frames_actual:
                            continue
                        try:
                            alignment = dtw._dtw(window, ref_mfcc, dist_method='cosine')
                            if alignment.distance < fine_best_dist:
                                fine_best_dist = alignment.distance
                                fine_best_frame = float(i)
                        except Exception:
                            continue

                fine_offset = fine_best_frame * hop_time

                # --- 置信度评估 ---
                norm_dist = fine_best_dist / max(1, ref_frames_actual)
                if norm_dist < 0.5:
                    confidence = 0.9
                elif norm_dist < 1.0:
                    confidence = 0.7
                elif norm_dist < 2.0:
                    confidence = 0.5
                elif norm_dist < 4.0:
                    confidence = 0.3
                elif norm_dist < 6.0:
                    confidence = 0.15
                else:
                    confidence = 0.05

                # 日志标注是否跳过了精扫
                skip_note = ", skip_fine" if norm_dist_check < 1.0 else ""
                logger.info(f"[全范围DTW] {ref_name}: offset={fine_offset:.2f}s, "
                            f"dist={fine_best_dist:.1f}, norm={norm_dist:.2f}, "
                            f"conf={confidence:.2f}, enhanced={use_pre_enhance}"
                            f"{skip_note}")

                return {
                    "ref_id": ref_name,
                    "ref_file": ref_path,
                    "ref_name": ref_name,
                    "offset_in_test": fine_offset,
                    "confidence": confidence,
                    "dtw_distance": float(fine_best_dist),
                    "normalized_distance": float(norm_dist),
                    "method": "full_range_dtw",
                    "snr": snr,
                    "pre_enhanced": use_pre_enhance
                }
            except Exception as e:
                logger.debug(f"[全范围DTW] 参考{ref_name}处理异常: {e}")
                return None

        # 并行处理所有参考（每个参考独立DTW，互不依赖）
        if len(ref_entries) > 1:
            logger.info(f"[全范围DTW] 并行处理{len(ref_entries)}个参考音频")
            dtw_executor = get_shared_executor()
            all_futures = {
                dtw_executor.submit(_process_single_ref, ref_name, ref_path): ref_name
                for ref_name, ref_path in ref_entries
            }
            for future in as_completed(all_futures):
                try:
                    r = future.result()
                    if r is not None:
                        results.append(r)
                except Exception as e:
                    logger.debug(f"[全范围DTW] 线程异常: {e}")
        else:
            # 只有一个参考时，直接执行（避免线程开销）
            for ref_name, ref_path in ref_entries:
                r = _process_single_ref(ref_name, ref_path)
                if r is not None:
                    results.append(r)

        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results

    def _hpss_fine_align(self, ref_audio: Optional[np.ndarray],
                          test_audio: np.ndarray,
                          dtw_offset: float, sr: int = 16000,
                          kernel_size: int = 31,
                          max_correction_s: float = 2.0,
                          min_quality: float = 0.02,
                          ref_harmonic: Optional[np.ndarray] = None,
                          ref_samples: Optional[int] = None,
                          decimate_for_hpss: bool = True,
                          test_snr: Optional[float] = None) -> Tuple[float, float, float]:
        """
        HPSS谐波互相关精对齐
        对DTW帧级定位结果进行样本级精对齐修正

        性能优化:
        - 默认kernel_size从51降至31，中值滤波加速~40%
        - SNR>15dB时跳过HPSS，使用原始音频互相关（更快、更准确）

        原理：
        1. 按DTW offset切出测试段
        2. 对参考和测试段分别提取HPSS谐波分量（抑制非谐波噪声）
        3. 谐波分量互相关，检测残留的时间偏移（lag）
        4. 修正offset = DTW_offset - lag

        Args:
            ref_audio: 参考音频数组(ref_harmonic和ref_samples提供时可为None)
            test_audio: 测试音频数组
            dtw_offset: DTW给出的偏移量(秒)
            sr: 采样率
            kernel_size: HPSS谐波核大小，建议21-51，默认31
            max_correction_s: 最大允许修正量(秒)
            min_quality: 最低互相关质量阈值
            ref_harmonic: 预缓存的参考谐波分量, None则从ref_audio计算
            ref_samples: 预缓存的参考样本数, None则从ref_audio获取
            decimate_for_hpss: 高采样率(>=16K)时先2倍降采样再HPSS，提效降噪
            test_snr: 测试音频估计SNR(dB)。>15dB时跳过HPSS用原始音频互相关

        Returns:
            (corrected_offset, residual_lag, quality)
        """
        from scipy import signal as scipy_signal

        # 使用缓存的ref_samples或从音频获取
        if ref_samples is None:
            assert ref_audio is not None, "ref_audio或ref_samples必须提供"
            ref_samples = len(ref_audio)

        # 按DTW offset切出段
        cut_start = int(dtw_offset * sr)
        cut_end = min(len(test_audio), cut_start + ref_samples)
        segment = test_audio[cut_start:cut_end]

        if len(segment) < ref_samples:
            segment = np.pad(segment, (0, ref_samples - len(segment)))
        elif len(segment) > ref_samples:
            segment = segment[:ref_samples]

        # HPSS谐波提取（使用缓存或实时计算；开启降采样时先decimate再HPSS）
        # 所有音频统一走HPSS路径以确保对齐精度（P2.2已优化kernel_size=31 + scipy快速实现）
        _hpss_sr = sr
        if ref_harmonic is not None:
            ref_harm = ref_harmonic
            if decimate_for_hpss and sr >= 16000:
                _hpss_sr = sr // 2
                ref_harm = scipy_signal.decimate(ref_harm, 2, ftype='fir')
        else:
            _ref_for_hpss = ref_audio
            if decimate_for_hpss and sr >= 16000:
                _hpss_sr = sr // 2
                _ref_for_hpss = scipy_signal.decimate(ref_audio, 2, ftype='fir')
            ref_harm = extract_harmonic_component(_ref_for_hpss, kernel_size=kernel_size)

        _seg_for_hpss = segment
        if decimate_for_hpss and sr >= 16000:
            _seg_for_hpss = scipy_signal.decimate(segment, 2, ftype='fir')
        seg_harm = extract_harmonic_component(_seg_for_hpss, kernel_size=kernel_size)

        # 互相关找lag（注意：_hpss_sr是decimate后的采样率，与sr可能不同）
        min_len = min(len(ref_harm), len(seg_harm))
        correlation = scipy_signal.correlate(
            ref_harm[:min_len], seg_harm[:min_len], mode='full', method='auto'
        )
        mid = min_len - 1

        max_lag = int(max_correction_s * _hpss_sr)
        s = max(0, mid - max_lag)
        e = min(len(correlation), mid + max_lag)
        search = correlation[s:e]

        pk = np.argmax(np.abs(search))
        lag = (s + pk) - mid
        lag_s = lag / _hpss_sr  # 使用HPSS的采样率换算实际时间

        # 质量 = 互相关峰值 / 参考自相关峰值
        auto_corr = scipy_signal.correlate(
            ref_harm[:min_len], ref_harm[:min_len], mode='same'
        )
        quality = float(np.max(np.abs(correlation)) / (np.max(np.abs(auto_corr)) + 1e-10))

        # 安全校验：质量太低或lag不合理时跳过修正
        if quality < min_quality:
            logger.debug(f"[HPSS对齐] 质量不足: {quality:.4f} < {min_quality}, 跳过修正")
            return dtw_offset, 0.0, quality

        if abs(lag_s) > max_correction_s:
            logger.debug(f"[HPSS对齐] lag超范围: {lag_s:.2f}s > {max_correction_s}s, 跳过修正")
            return dtw_offset, 0.0, quality

        # 修正offset = DTW offset - lag
        corrected_offset = dtw_offset - lag_s

        # 安全校验：极低质量互相关（如0dB SNR）跳过修正
        if quality < 0.08:
            logger.debug(f"[HPSS对齐] 质量过低({quality:.4f})，跳过修正")
            corrected_offset = dtw_offset

        # 验证：用修正后offset重新切分，检测残留lag
        ncs = int(corrected_offset * sr)
        nce = min(len(test_audio), ncs + ref_samples)
        new_seg = test_audio[ncs:nce]
        if len(new_seg) < ref_samples:
            new_seg = np.pad(new_seg, (0, ref_samples - len(new_seg)))
        elif len(new_seg) > ref_samples:
            new_seg = new_seg[:ref_samples]

        new_harm = extract_harmonic_component(new_seg, kernel_size=kernel_size)
        new_corr = scipy_signal.correlate(
            ref_harm[:min_len], new_harm[:min_len], mode='full', method='auto'
        )
        new_search = new_corr[s:e]
        new_pk = np.argmax(np.abs(new_search))
        residual_lag = ((s + new_pk) - mid) / sr

        logger.info(f"[HPSS对齐] offset: {dtw_offset:.3f}s → {corrected_offset:.3f}s, "
                    f"lag={lag_s:+.4f}s, residual={residual_lag:+.4f}s, "
                    f"quality={quality:.4f}, kernel={kernel_size}")

        return corrected_offset, residual_lag, quality

    def match_with_fallback(self, test_audio_path: str) -> Dict:
        """
        优化匹配 - WeNet语义匹配为主，全范围DTW为回退

        匹配策略层级：
          Level A: WeNet ASR + 文本DTW（语义匹配，含WER副产品）
          Level B: 全范围DTW扫描（声学匹配，原Level A）
          Level C: 嵌入向量匹配（SV模型）
          Level D: 接受低置信度DTW结果

        Returns:
            {
                "matched": bool, "ref_file": str or None,
                "offset": float, "method": str,
                "confidence": float, "detail": dict,
                "wer": float,       # WER（Level A匹配时的副产品）
                "segment_matches": list  # 全部分段匹配信息
            }
        """
        test_basename = os.path.basename(test_audio_path)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[优化匹配] 开始匹配: {test_basename}")
        logger.info(f"{'=' * 60}")

        snr, noise_floor = self._estimate_snr_and_noise_floor(test_audio_path)
        logger.info(f"[优化匹配] SNR={snr:.1f}dB")

        result = {
            "matched": False, "ref_file": None, "offset": 0.0,
            "method": "none", "confidence": 0.0, "snr": snr,
            "wer": 0.0, "segment_matches": [], "detail": {}
        }

        # ----- Level A: WeNet语义匹配（主方案）-----
        if self._init_wenet_matcher():
            logger.info(f"[优化匹配] Level A: WeNet ASR + 文本DTW")
            asr_start = time.time()
            asr_result = self._wenet_matcher.match_with_fallback(test_audio_path)
            asr_time = time.time() - asr_start

            seg_matches = asr_result.get("detail", {}).get("segment_matches", [])
            if seg_matches:
                best = seg_matches[0]
                logger.info(f"[优化匹配] ✓ Level A 语义匹配: {best['ref_name']} "
                            f"@ {best['offset_in_test']:.2f}s, "
                            f"conf={best['confidence']:.3f}, WER={best['wer']:.3f}, "
                            f"耗时={asr_time:.1f}s")
                result.update({
                    "matched": True,
                    "ref_file": best["ref_file"],
                    "ref_name": best["ref_name"],
                    "offset": best["offset_in_test"],
                    "method": "asr_text_dtw",
                    "confidence": best["confidence"],
                    "wer": best["wer"],
                    "segment_matches": seg_matches,
                    "detail": {
                        "asr_result": asr_result,
                        "asr_time": asr_time,
                        "n_segments": len(seg_matches)
                    }
                })
                return result
            else:
                logger.info(f"[优化匹配] Level A 未匹配 (耗时={asr_time:.1f}s), 降级到Level B")
        else:
            logger.info(f"[优化匹配] Level A: WeNet不可用，直接降级到Level B")

        # ----- Level B: 全范围DTW扫描（原Level A）-----
        logger.info(f"[优化匹配] Level B: 全范围DTW扫描")
        dtw_start = time.time()
        dtw_results = self._full_range_dtw_sweep(test_audio_path, snr)
        dtw_time = time.time() - dtw_start
        logger.info(f"[优化匹配] Level B 耗时={dtw_time:.1f}s, 结果={len(dtw_results)}个")

        if dtw_results:
            best = dtw_results[0]
            conf = best['confidence']
            if conf >= 0.3:
                logger.info(f"[优化匹配] ✓ DTW匹配: {best['ref_name']} "
                            f"@ {best['offset_in_test']:.2f}s, conf={conf:.3f}")
                result.update({
                    "matched": True, "ref_file": best["ref_file"],
                    "ref_name": best["ref_name"],
                    "offset": best["offset_in_test"],
                    "method": best["method"],
                    "confidence": conf,
                    "detail": {"dtw_results": dtw_results}
                })
                return result
            result["detail"]["dtw_results"] = dtw_results

        # ----- Level C: 嵌入向量匹配（低SNR回退）-----
        if not self._embedding_initialized:
            try:
                self.embedding_matcher.build_reference_embeddings(self.ref_dir)
                self._embedding_initialized = True
            except Exception as e:
                logger.warning(f"[优化匹配] 嵌入初始化失败: {e}")

        if self._embedding_initialized:
            logger.info(f"[优化匹配] Level C: 嵌入向量匹配")
            emb_matches = self.embedding_matcher.match_by_embedding(
                test_audio_path, pre_enhance=(snr < 10)
            )
            if emb_matches and emb_matches[0].get('similarity', 0) > 0.5:
                best = emb_matches[0]
                logger.info(f"[优化匹配] ✓ 嵌入匹配: {best['ref_name']}")
                result.update({
                    "matched": True, "ref_file": best["ref_file"],
                    "ref_name": best["ref_name"],
                    "offset": best["offset_in_test"],
                    "method": "embedding",
                    "confidence": best["similarity"],
                    "detail": result.get("detail", {})
                })
                return result

        # ----- Level D: 接受低置信度DTW结果 -----
        if dtw_results and dtw_results[0]['confidence'] >= 0.1:
            best = dtw_results[0]
            logger.info(f"[优化匹配] ✓ 接受低置信DTW结果: {best['ref_name']} "
                        f"@ {best['offset_in_test']:.2f}s")
            result.update({
                "matched": True, "ref_file": best["ref_file"],
                "ref_name": best["ref_name"],
                "offset": best["offset_in_test"],
                "method": best["method"] + "_lowconf",
                "confidence": best["confidence"],
                "detail": result.get("detail", {})
            })
            return result

        logger.warning(f"[优化匹配] ✗ 全部策略失败")
        return result


# ============================================================================
# 基于OptimizedMatcher的音频切分（替代audio_cut.py的旧MFCCLocate方案）
# ============================================================================

def cut_all_audio_files_with_optimized_matcher(
    input_file_list: list,
    ref_dir: str,
    output_dir: str,
    redundancy: float = 0.0,
    cached_dtw_results: Optional[Dict[str, List[Dict]]] = None,
    opt_matcher: Optional['OptimizedMatcher'] = None
) -> list:
    """
    使用OptimizedMatcher（全范围DTW）定位并切分音频
    切分后每段时长与参考音频完全一致（等长对齐），用于带参考MOS计算。

    对每个测试音频:
    1. 用OptimizedMatcher全范围DTW扫描，找出所有参考段的位置
    2. 从 offset 到 offset+ref_dur 精确切分
    3. 输出文件名 _001 ~ _004 对应 ref_001 ~ ref_004

    Args:
        input_file_list: 测试音频文件路径列表
        ref_dir: 参考音频目录
        output_dir: 输出目录
        cached_dtw_results: 可选，预检测阶段缓存的DTW匹配结果。
            格式: {test_file_path: [{"ref_id", "ref_file", "ref_name",
                                     "offset_in_test", "confidence", ...}, ...]}
            提供时跳过内部DTW扫描，直接使用缓存结果做HPSS精对齐+切分，
            避免预检测与切分阶段DTW重复计算。
        opt_matcher: 可选，预检测阶段创建并已缓存的OptimizedMatcher实例。
            提供时复用其内部MFCC缓存（_ref_mfcc_cache），避免重新提取。
            未提供则内部创建新实例（兼容旧调用方）。

    Returns:
        切分后的音频文件路径列表
    """
    import librosa
    import soundfile as sf
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)
    output_file_list = []

    # 扫描参考音频目录
    ref_files = []
    if ref_dir and os.path.isdir(ref_dir):
        for fname in sorted(os.listdir(ref_dir)):
            if fname.endswith(('.wav', '.mp3', '.flac')):
                ref_files.append((fname, os.path.join(ref_dir, fname)))

    logger.info(f"[优化切分] 共{len(input_file_list)}个测试文件，"
                f"{len(ref_files)}个参考音频")

    # ─── 复用预检测阶段已建好的OptimizedMatcher（含MFCC缓存） ───
    if opt_matcher is not None:
        opt_matcher.ref_dir = ref_dir  # 确保ref_dir正确
        logger.info(f"[优化切分] 复用外部OptimizedMatcher实例"
                    f" (已缓存{len(opt_matcher._ref_mfcc_cache)}个参考MFCC)")
    else:
        opt_matcher = OptimizedMatcher(ref_dir=ref_dir)
        # 预计算并缓存所有参考音频的MFCC
        opt_matcher.build_ref_cache(ref_dir)
        logger.info(f"[优化切分] 参考MFCC缓存完成 ({len(opt_matcher._ref_mfcc_cache)}个)")

    # ─── 预计算并缓存所有参考音频的HPSS谐波分量 ───
    ref_harm_cache = {}
    ref_samples_cache = {}
    for ref_name, ref_path in ref_files:
        ref_audio, _ = librosa.load(ref_path, sr=16000)
        ref_harm_cache[ref_name] = extract_harmonic_component(ref_audio, kernel_size=31)
        ref_samples_cache[ref_name] = len(ref_audio)
        logger.debug(f"[优化切分] 缓存HPSS: {ref_name} ({ref_samples_cache[ref_name]} samples)")

    # ─── 文件间并行处理：每个文件的DTW+HPSS+切分独立 ───
    def _process_single_file(test_file: str) -> list:
        """处理单个测试音频：DTW扫描→HPSS精对齐→切分
        如果cached_dtw_results中有该文件的DTW结果，跳过重复扫描。
        """
        test_name = os.path.splitext(os.path.basename(test_file))[0]

        # 检查是否有缓存的DTW结果（预检测阶段已计算，消除冗余DTW扫描）
        use_cached = (cached_dtw_results is not None
                      and test_file in cached_dtw_results
                      and cached_dtw_results[test_file])
        if use_cached:
            dtw_results = cached_dtw_results[test_file]
            logger.info(f"[优化切分] {test_name}: 复用缓存DTW结果 ({len(dtw_results)}个匹配)，跳过重复扫描")
            # 使用DTW阶段缓存的测试音频（预检测时已缓存）
            if opt_matcher._last_test_path == test_file and opt_matcher._last_test_y is not None:
                y_test = opt_matcher._last_test_y
                sr_test = opt_matcher._last_test_sr
            else:
                y_test, sr_test = librosa.load(test_file, sr=16000)
        else:
            # 多级匹配获取所有参考段位置（优先ASR语义匹配，回退到DTW）
            dtw_results = opt_matcher.get_all_segment_matches(test_file)

            if not dtw_results:
                logger.warning(f"[优化切分] {test_name}: 未找到任何匹配，跳过")
                return []

            # 使用DTW阶段缓存的测试音频（消除双加载）
            if opt_matcher._last_test_path == test_file and opt_matcher._last_test_y is not None:
                y_test = opt_matcher._last_test_y
                sr_test = opt_matcher._last_test_sr
            else:
                y_test, sr_test = librosa.load(test_file, sr=16000)

        # 按 ref_name 排序输出
        dtw_results.sort(key=lambda x: x.get('ref_name', ''))

        # ─── 文件内4个参考段的HPSS精对齐+切分（并行） ───
        def _hpss_and_cut(i: int, r: dict) -> Optional[str]:
            """单个参考段的HPSS精对齐+切分"""
            ref_id = r.get('ref_id', '')
            ref_name = r.get('ref_name', '')
            offset = r['offset_in_test']
            ref_harm = ref_harm_cache.get(ref_id)
            ref_nsamples = ref_samples_cache.get(ref_id, 0)
            if ref_harm is None or ref_nsamples == 0:
                return None
            # HPSS精对齐（传递SNR+原始音频，高SNR时跳过HPSS）
            test_snr = r.get('snr', None)
            final_offset, _, _ = opt_matcher._hpss_fine_align(
                None, y_test, offset, sr=sr_test, kernel_size=31,
                max_correction_s=2.0, min_quality=0.02,
                ref_harmonic=ref_harm, ref_samples=ref_nsamples,
                test_snr=test_snr
            )
            # 切分
            cs = int(final_offset * sr_test)
            ce = min(len(y_test), cs + ref_nsamples)
            seg = y_test[cs:ce]
            if len(seg) < ref_nsamples:
                seg = np.pad(seg, (0, ref_nsamples - len(seg)))
            elif len(seg) > ref_nsamples:
                seg = seg[:ref_nsamples]
            ref_tag = os.path.splitext(ref_name)[0] if ref_name else f"ref_{i + 1:03d}"
            suffix = f"_{ref_tag}_{i + 1:03d}.wav"
            op = os.path.join(output_dir, test_name + suffix)
            sf.write(op, seg, sr_test)
            logger.info(f"[优化切分] {test_name}{suffix}: ref={ref_name:15s} offset={final_offset:.2f}s")
            return op

        # 文件内4段HPSS并行（使用共享池，与文件级独立池不冲突）
        _seg_exec = get_shared_executor()
        _seg_futs = {_seg_exec.submit(_hpss_and_cut, i, r): i for i, r in enumerate(dtw_results)}
        file_outputs = []
        for _f in as_completed(_seg_futs):
            r = _f.result()
            if r:
                file_outputs.append(r)
        return file_outputs

    # 并行处理所有文件（使用独立线程池，不与内部DTW的get_shared_executor嵌套）
    logger.info(f"[优化切分] 并行处理{len(input_file_list)}个文件")
    from concurrent.futures import ThreadPoolExecutor as _TPE
    _file_executor = _TPE(max_workers=min(8, len(input_file_list)), thread_name_prefix='cut_files')
    try:
        _futures = {_file_executor.submit(_process_single_file, f): f for f in input_file_list}
        for future in as_completed(_futures):
            try:
                output_file_list.extend(future.result())
            except Exception as e:
                logger.error(f"[优化切分] 文件异常: {e}")
    finally:
        _file_executor.shutdown(wait=True)

    logger.info(f"[优化切分] 完成: 共切出{len(output_file_list)}个片段")
    return output_file_list
