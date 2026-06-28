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

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============ 全局共享线程池 ============
_SHARED_EXECUTOR = None
_SHARED_EXECUTOR_LOCK = threading.Lock()


def get_shared_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """获取全局共享线程池（延迟初始化，避免每次调用创建销毁）"""
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is None:
        with _SHARED_EXECUTOR_LOCK:
            if _SHARED_EXECUTOR is None:
                _SHARED_EXECUTOR = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix='matching_shared'
                )
    return _SHARED_EXECUTOR


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
                                 kernel_size: int = 51) -> np.ndarray:
    """
    提取音频的谐波分量（用于更鲁棒的峰值检测和对齐）
    将音频分解为谐波部分（语音）和打击乐部分（噪声/瞬态）

    Args:
        audio: 输入音频数组
        sr: 采样率(仅用于日志)
        kernel_size: HPSS核大小，越大谐波越纯净；建议值31-81，默认51

    原理：
    - 对STFT时频谱沿频率轴做中值滤波 → 打击乐分量（垂直条纹）
    - 沿时间轴做中值滤波 → 谐波分量（水平条纹）
    - 语音的元音表现为稳定的水平能量带，通过时间轴中值滤波保留
    - 环境噪声宽频分布，在时频谱上无稳定结构，被中值滤波衰减
    """
    import librosa
    try:
        harmonic, percussive = librosa.effects.hpss(
            audio,
            kernel_size=kernel_size,
            power=2.0
        )
        logger.debug(f"[HPSS] 谐波分量提取完成: len={len(audio)}, "
                     f"kernel={kernel_size}, "
                     f"harmonic_rms={np.sqrt(np.mean(harmonic**2)):.4f}")
        return harmonic
    except Exception as e:
        logger.debug(f"[HPSS] HPSS分解失败: {e}，返回原始音频")
        return audio


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
        需要保存为临时文件后调用pipeline
        """
        import tempfile
        import soundfile as sf

        pipe = self._get_sv_pipeline()
        if pipe is None:
            return None

        # 确保16kHz
        if sr != self.sr:
            audio = librosa_resample(audio, orig_sr=sr, target_sr=self.sr)

        # 保存到临时文件
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            sf.write(tmp_file.name, audio, self.sr)
            result = pipe(tmp_file.name, output_emb=True)
            if 'embs' in result and len(result['embs']) > 0:
                emb = result['embs'][0]
                if isinstance(emb, (list, tuple)):
                    emb = np.array(emb)
                return emb.flatten()
        except Exception as e:
            logger.debug(f"[嵌入匹配] 片段嵌入计算失败: {e}")
        finally:
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass

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

        # 指纹匹配器（复用原有逻辑，但配置自适应）
        from reference_matcher import ReferenceMatcher, FingerprintConfig, DEFAULT_CONFIG
        self._default_config = DEFAULT_CONFIG
        self._fingerprint_matcher = None

    def _get_fingerprint_matcher(self):
        """获取指纹匹配器实例"""
        if self._fingerprint_matcher is None and self.ref_dir:
            from reference_matcher import get_reference_matcher
            self._fingerprint_matcher = get_reference_matcher(ref_dir=self.ref_dir)
        return self._fingerprint_matcher

    def _estimate_snr_and_noise_floor(self, audio_path: str) -> Tuple[float, float]:
        """估计SNR和噪声底噪"""
        y, sr = librosa_load(audio_path, sr=16000)
        snr = self.pre_enhancer.estimate_snr(y)
        noise_floor = self.pre_enhancer.estimate_noise_floor(y)
        return snr, noise_floor

    def _adaptive_fingerprint_matching(self, test_audio_path: str,
                                        snr: float) -> List:
        """
        自适应指纹匹配
        根据SNR调整指纹参数
        """
        from reference_matcher import FingerprintConfig

        # 根据SNR自适应调整参数
        if snr < 0:
            # 极低SNR: 大幅放宽条件
            amp_min = 2.0
            min_hash = 3
            neighborhood = 10
            near_num = 15
        elif snr < 10:
            # 低SNR: 适度放宽
            amp_min = 3.0
            min_hash = 4
            neighborhood = 12
            near_num = 18
        else:
            # 正常SNR: 使用默认参数
            amp_min = 5.0
            min_hash = 5
            neighborhood = 15
            near_num = 20

        # 对测试音频进行预增强
        if snr < 15:
            logger.info(f"[优化匹配] SNR={snr:.1f}dB < 15dB，预增强后提取指纹")
            try:
                y, sr = librosa_load(test_audio_path, sr=16000)
                y_enhanced = self.pre_enhancer.enhance(y, sr)

                # 保存增强后的音频到临时文件（用于指纹提取）
                import tempfile
                import soundfile as sf
                tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav')
                os.close(tmp_fd)
                sf.write(tmp_path, y_enhanced, 16000)

                # 使用增强音频提取指纹
                from reference_matcher import FingerprintConfig, AudioFingerprinter
                config = FingerprintConfig(
                    amp_min=amp_min,
                    min_hash_match=min_hash,
                    neighborhood=neighborhood,
                    near_num=near_num
                )
                fingerprinter = AudioFingerprinter(config)
                test_hashes = fingerprinter.extract_hashes(tmp_path)

                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                # 在数据库中查询
                matcher = self._get_fingerprint_matcher()
                if matcher is None:
                    return []

                match_hash_list = matcher.database.query(test_hashes)
                logger.info(f"[优化匹配] 自适应指纹匹配: snr={snr:.1f}dB, "
                            f"amp_min={amp_min}, min_hash={min_hash}, "
                            f"hashes={len(test_hashes)}, matches={len(match_hash_list)}")

                if not match_hash_list:
                    return []

                # 对齐分析
                alignment_counts = {}
                for ref_id, offset_ref, offset_query in match_hash_list:
                    offset_diff = int(offset_ref) - int(offset_query)
                    key = (ref_id, offset_diff)
                    alignment_counts[key] = alignment_counts.get(key, 0) + 1

                # 生成结果
                results = []
                for (ref_id, offset_diff), count in alignment_counts.items():
                    if count >= min_hash:
                        entry = matcher.database.get_entry(ref_id)
                        if entry:
                            offset_sec = offset_diff * config.hop_length / config.sr
                            confidence = min(1.0, count / max(1, entry.hash_count * 0.15))
                            results.append({
                                "ref_id": ref_id,
                                "ref_file": entry.ref_file,
                                "ref_name": entry.ref_name,
                                "offset_in_test": max(0, offset_sec),
                                "confidence": confidence,
                                "hash_matches": count,
                                "method": "adaptive_fingerprint",
                                "snr": snr
                            })

                results.sort(key=lambda x: x['confidence'], reverse=True)
                return results
            except Exception as e:
                logger.warning(f"[优化匹配] 自适应指纹匹配异常: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return []

        # 高SNR: 使用原始匹配器，转换为统一dict格式
        matcher = self._get_fingerprint_matcher()
        if matcher is None:
            return []

        raw_results = matcher.match_test_audio(test_audio_path)
        dict_results = []
        for r in raw_results:
            dict_results.append({
                "ref_id": r.ref_id,
                "ref_file": r.ref_file,
                "ref_name": r.ref_name,
                "offset_in_test": r.offset_in_test,
                "confidence": r.confidence,
                "hash_matches": r.hash_matches,
                "method": "standard_fingerprint",
                "snr": snr
            })
        return dict_results

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
        matcher = self._get_fingerprint_matcher()
        if matcher is None or not matcher.database.entries:
            return []

        use_pre_enhance = snr < 10
        results = []
        import librosa

        y_test, sr_test = librosa.load(test_audio_path, sr=16000)
        dur_test = len(y_test) / sr_test

        logger.info(f"[全范围DTW] 测试音频={os.path.basename(test_audio_path)}, "
                    f"dur={dur_test:.0f}s, 参考数={len(matcher.database.entries)}, "
                    f"pre_enhance={use_pre_enhance}")

        # 提取测试MFCC一次（所有参考复用）
        dtw = self.robust_dtw
        test_mfcc, test_y = dtw.extract_robust_mfcc_from_array(
            y_test, sr_test, pre_enhance=use_pre_enhance
        )
        total_frames = test_mfcc.shape[0]
        hop_time = 512.0 / sr_test  # 每帧对应时间(秒)

        def _process_single_ref(ref_id: str, entry) -> Optional[Dict]:
            """对单个参考音频执行两级DTW搜索（线程安全，独立上下文）"""
            try:
                ref_dur = entry.duration
                ref_frames = int(ref_dur * sr_test / 512)
                if total_frames <= ref_frames:
                    return None

                # 提取参考MFCC
                ref_y, _ = librosa.load(entry.ref_file, sr=16000)
                ref_mfcc, _ = dtw.extract_robust_mfcc_from_array(
                    ref_y, 16000, pre_enhance=False
                )
                ref_frames_actual = ref_mfcc.shape[0]

                # --- 粗扫描（大步长，快速定位） ---
                coarse_step = max(20, int((total_frames - ref_frames_actual) / 15))
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
                logger.debug(f"[全范围DTW] {entry.ref_name}: 粗扫offset={coarse_offset:.1f}s, "
                            f"dist={best_dist:.1f}")

                # --- 精细搜索（小步长精确定位） ---
                fine_range_sec = 3.0
                fine_start_frame = max(0, int((coarse_offset - fine_range_sec) / hop_time))
                fine_end_frame = min(total_frames - ref_frames_actual,
                                     int((coarse_offset + ref_dur + fine_range_sec) / hop_time))

                fine_best_dist = best_dist
                fine_best_frame = best_frame
                for i in range(fine_start_frame, fine_end_frame, 3):
                    window = test_mfcc[i:i + ref_frames_actual]
                    if window.shape[0] < ref_frames_actual:
                        break
                    try:
                        alignment = dtw._dtw(window, ref_mfcc, dist_method='cosine')
                        if alignment.distance < fine_best_dist:
                            fine_best_dist = alignment.distance
                            fine_best_frame = i
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

                logger.info(f"[全范围DTW] {entry.ref_name}: offset={fine_offset:.2f}s, "
                            f"dist={fine_best_dist:.1f}, norm={norm_dist:.2f}, "
                            f"conf={confidence:.2f}, enhanced={use_pre_enhance}")

                return {
                    "ref_id": ref_id,
                    "ref_file": entry.ref_file,
                    "ref_name": entry.ref_name,
                    "offset_in_test": fine_offset,
                    "confidence": confidence,
                    "dtw_distance": float(fine_best_dist),
                    "normalized_distance": float(norm_dist),
                    "method": "full_range_dtw",
                    "snr": snr,
                    "pre_enhanced": use_pre_enhance
                }
            except Exception as e:
                logger.debug(f"[全范围DTW] 参考{entry.ref_name}处理异常: {e}")
                return None

        # 并行处理所有参考（每个参考独立DTW，互不依赖）
        entries = list(matcher.database.entries.items())
        if len(entries) > 1:
            logger.info(f"[全范围DTW] 并行处理{len(entries)}个参考音频")
            dtw_executor = get_shared_executor()
            all_futures = {
                dtw_executor.submit(_process_single_ref, ref_id, entry): ref_id
                for ref_id, entry in entries
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
            for ref_id, entry in entries:
                r = _process_single_ref(ref_id, entry)
                if r is not None:
                    results.append(r)

        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results

    def _hpss_fine_align(self, ref_audio: Optional[np.ndarray],
                          test_audio: np.ndarray,
                          dtw_offset: float, sr: int = 16000,
                          kernel_size: int = 51,
                          max_correction_s: float = 2.0,
                          min_quality: float = 0.02,
                          ref_harmonic: Optional[np.ndarray] = None,
                          ref_samples: Optional[int] = None) -> Tuple[float, float, float]:
        """
        HPSS谐波互相关精对齐
        对DTW帧级定位结果进行样本级精对齐修正

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
            kernel_size: HPSS谐波核大小，建议51
            max_correction_s: 最大允许修正量(秒)
            min_quality: 最低互相关质量阈值
            ref_harmonic: 预缓存的参考谐波分量, None则从ref_audio计算
            ref_samples: 预缓存的参考样本数, None则从ref_audio获取

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

        # HPSS谐波提取（使用缓存或实时计算）
        if ref_harmonic is not None:
            ref_harm = ref_harmonic
        else:
            ref_harm = extract_harmonic_component(ref_audio, kernel_size=kernel_size)

        seg_harm = extract_harmonic_component(segment, kernel_size=kernel_size)

        # 互相关找lag
        min_len = min(len(ref_harm), len(seg_harm))
        correlation = scipy_signal.correlate(
            ref_harm[:min_len], seg_harm[:min_len], mode='full', method='auto'
        )
        mid = min_len - 1

        max_lag = int(max_correction_s * sr)
        s = max(0, mid - max_lag)
        e = min(len(correlation), mid + max_lag)
        search = correlation[s:e]

        pk = np.argmax(np.abs(search))
        lag = (s + pk) - mid
        lag_s = lag / sr

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
        优化匹配 - 全范围DTW为主，嵌入向量为回退

        Returns:
            {
                "matched": bool, "ref_file": str or None,
                "offset": float, "method": str,
                "confidence": float, "detail": dict
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
            "method": "none", "confidence": 0.0, "snr": snr, "detail": {}
        }

        # ----- Level A: 全范围DTW扫描（主方案）-----
        logger.info(f"[优化匹配] Level A: 全范围DTW扫描")
        dtw_start = time.time()
        dtw_results = self._full_range_dtw_sweep(test_audio_path, snr)
        dtw_time = time.time() - dtw_start
        logger.info(f"[优化匹配] Level A 耗时={dtw_time:.1f}s, 结果={len(dtw_results)}个")

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

        # ----- Level B: 嵌入向量匹配（低SNR回退）-----
        if not self._embedding_initialized:
            try:
                self.embedding_matcher.build_reference_embeddings(self.ref_dir)
                self._embedding_initialized = True
            except Exception as e:
                logger.warning(f"[优化匹配] 嵌入初始化失败: {e}")

        if self._embedding_initialized:
            logger.info(f"[优化匹配] Level B: 嵌入向量匹配")
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

        # ----- Level C: 接受低置信度DTW结果 -----
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
    redundancy: float = 0.0
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

    Returns:
        切分后的音频文件路径列表
    """
    import librosa
    import soundfile as sf
    from pathlib import Path

    # 确保参考库已加载
    from reference_matcher import get_reference_matcher
    matcher = get_reference_matcher(ref_dir=ref_dir)
    if not matcher.database.entries:
        matcher.build_database(ref_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_file_list = []

    logger.info(f"[优化切分] 共{len(input_file_list)}个测试文件，"
                f"{len(matcher.database.entries)}个参考音频")

    opt_matcher = OptimizedMatcher(ref_dir=ref_dir)

    # ─── 预计算并缓存所有参考音频的HPSS谐波分量 ───
    ref_harm_cache = {}
    ref_samples_cache = {}
    for ref_id, entry in matcher.database.entries.items():
        ref_audio, _ = librosa.load(entry.ref_file, sr=16000)
        ref_harm_cache[ref_id] = extract_harmonic_component(ref_audio, kernel_size=51)
        ref_samples_cache[ref_id] = len(ref_audio)
        logger.debug(f"[优化切分] 缓存HPSS: {entry.ref_name} ({ref_samples_cache[ref_id]} samples)")

    # ─── 文件间并行处理：每个文件的DTW+HPSS+切分独立 ───
    def _process_single_file(test_file: str) -> list:
        """处理单个测试音频：DTW扫描→HPSS精对齐→切分"""
        test_name = os.path.splitext(os.path.basename(test_file))[0]

        # 全范围DTW扫描获取所有参考段位置
        snr, _ = opt_matcher._estimate_snr_and_noise_floor(test_file)
        dtw_results = opt_matcher._full_range_dtw_sweep(test_file, snr)

        if not dtw_results:
            logger.warning(f"[优化切分] {test_name}: 未找到任何匹配，跳过")
            return []

        # 加载测试音频
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
            # HPSS精对齐
            final_offset, _, _ = opt_matcher._hpss_fine_align(
                None, y_test, offset, sr=sr_test, kernel_size=51,
                max_correction_s=2.0, min_quality=0.02,
                ref_harmonic=ref_harm, ref_samples=ref_nsamples
            )
            # 切分
            cs = int(final_offset * sr_test)
            ce = min(len(y_test), cs + ref_nsamples)
            seg = y_test[cs:ce]
            if len(seg) < ref_nsamples:
                seg = np.pad(seg, (0, ref_nsamples - len(seg)))
            elif len(seg) > ref_nsamples:
                seg = seg[:ref_nsamples]
            suffix = f"_{i + 1:03d}.wav"
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

    # 并行处理所有文件（⚠️ 使用独立线程池，不与内部DTW的get_shared_executor嵌套）
    # 文件数≤4时直接在当前线程执行，避免线程开销
    logger.info(f"[优化切分] 并行处理{len(input_file_list)}个文件")
    if len(input_file_list) <= 4:
        for f in input_file_list:
            output_file_list.extend(_process_single_file(f))
    else:
        # 用独立池避免与DTW内部共享池嵌套死锁
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _file_executor = _TPE(max_workers=4, thread_name_prefix='cut_files')
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
