"""
@file: audio_mos_optimized.py
@time: 2026/5/14
@desc: 优化版MOS分计算模块
基于原audio_mos.py进行性能优化:
1. 并行计算多个模型
2. 音频文件缓存
3. 批处理优化
4. 性能监控
"""
import os
import sys
import time
import asyncio
import threading
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# 添加本地包路径 - 从 algorithms 目录导入
# 当前文件位置: app/core/calculator/mos_calculator.py
# 需要到达: app/algorithms/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ALGORITHMS_DIR = os.path.join(_PROJECT_ROOT, 'app', 'algorithms')
sys.path.insert(0, _PROJECT_ROOT)  # 用于 `from app.core._executor` 统一线程池导入
sys.path.insert(0, _ALGORITHMS_DIR)
sys.path.insert(0, os.path.join(_ALGORITHMS_DIR, 'speechmetrics'))
# nisqa的predict.py在algorithms/nisqa/目录下，需要添加algorithms目录到路径
sys.path.insert(0, _ALGORITHMS_DIR)
sys.path.insert(0, os.path.join(_ALGORITHMS_DIR, 'wenet'))
sys.path.insert(0, os.path.join(_ALGORITHMS_DIR, 'scoreq'))
sys.path.insert(0, os.path.join(_ALGORITHMS_DIR, 'utmos'))

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import warnings
import logging
import torch
import torch.cuda as cuda

# 模块级日志记录器 - 使用 audiomos 名称以匹配项目日志配置
logger = logging.getLogger('audiomos')

warnings.filterwarnings("ignore")

# ============ CUDA延迟初始化（不在模块加载时设置环境变量）============
_CUDA_INITIALIZED = False


def init_cuda_from_config():
    """
    从config.yaml读取CUDA配置并初始化。
    延迟到首次需要时调用，避免模块import时产生全局副作用。
    """
    global _CUDA_INITIALIZED
    if _CUDA_INITIALIZED:
        return
    _CUDA_INITIALIZED = True

    # CUDA设备配置
    # 优先级: 环境变量 CUDA_VISIBLE_DEVICES > config.yaml cuda.device_id > 默认不设置
    _cuda_device_id = None
    _config_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "config", "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "config.yaml"),
    ]
    for _cfg_path in _config_paths:
        if os.path.exists(_cfg_path):
            try:
                import yaml
                with open(_cfg_path, 'r') as _f:
                    _config = yaml.safe_load(_f)
                _cuda_cfg = _config.get('cuda', {})
                if _cuda_cfg.get('enabled', True):
                    _cuda_device_id = _cuda_cfg.get('device_id', None)
                break
            except Exception:
                pass

    if _cuda_device_id is not None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_cuda_device_id)

    # 启用cuDNN
    torch.backends.cudnn.enabled = True

    # 初始化CUDA上下文
    try:
        if torch.cuda.is_available():
            _ = torch.zeros(1).cuda()
            _device_count = torch.cuda.device_count()
            _device_name = torch.cuda.get_device_name(0)
            _cudnn_version = torch.backends.cudnn.version()
            _cudnn_status = "启用" if torch.backends.cudnn.enabled else "禁用"
            print(f"✓ CUDA初始化完成: {_device_name} (共{_device_count}个GPU, cuDNN{_cudnn_status}, 版本{_cudnn_version})")
    except Exception as e:
        print(f"⚠️ CUDA初始化警告: {e}")

# 尝试导入各模块
try:
    import speechmetrics.speechmetrics as speechmetrics
except ImportError:
    try:
        import speechmetrics
    except ImportError:
        speechmetrics = None
        print("警告: speechmetrics未安装，STOI/SISDR评分将不可用")

try:
    import scoreq
except ImportError:
    scoreq = None
    print("警告: scoreq未安装，Scoreq评分将不可用")

# 导入UTMOS - 路径已在顶部添加
try:
    from utmos_score import UTMOSCore, UTMOS_AVAILABLE
except ImportError as e:
    UTMOS_AVAILABLE = False
    print(f"警告: utmos_score未导入，UTMOS评分将不可用 - {e}")

try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("警告: pesq模块未安装，PESQ评分将不可用")

try:
    from nisqa.predict import nisqa_predict
    NISQA_AVAILABLE = True
except ImportError as e:
    NISQA_AVAILABLE = False
    print(f"警告: nisqa模块未安装，NISQA评分将不可用 - {e}")

# ModelScope兼容补丁（解决datasets版本兼容性问题）
try:
    import datasets
    if not hasattr(datasets, 'LargeList'):
        class _LargeListStub(list):
            pass
        datasets.LargeList = _LargeListStub
        print("已应用ModelScope兼容补丁 (datasets.LargeList)")
except ImportError:
    pass

# 尝试导入modelscope
try:
    from modelscope import pipeline
    MODELSCOPE_AVAILABLE = True
    print("✓ ModelScope可用")
except ImportError as e:
    MODELSCOPE_AVAILABLE = False
    print(f"警告: modelscope模块未安装或导入失败，TCF评分将不可用 - {e}")

WENET_AVAILABLE = False
try:
    # 路径已在顶部添加
    import wenet
    if hasattr(wenet, 'load_model'):
        # 添加 wenet_local 路径
        sys.path.insert(0, os.path.join(_ALGORITHMS_DIR, 'wenet', 'wenet_local'))
        from wer import wer
        WENET_AVAILABLE = True
    else:
        print("警告: wenet库不完整，WER评分将不可用")
except ImportError as e:
    print(f"警告: wenet模块未安装，WER评分将不可用 - {e}")

# 注意：兼容性补丁已在文件顶部应用，不需要再次导入 app.compat
# 如果之前已成功导入 modelscope，保持 MODELSCOPE_AVAILABLE = True
# 如果之前导入失败，MODELSCOPE_AVAILABLE 已经为 False
if not MODELSCOPE_AVAILABLE:
    print("警告: modelscope模块未安装，音色还原度评分将不可用")


# ============ 全局共享线程池（统一模块，消除多模块独立线程池的资源争用） ============
# 使用importlib加载统一线程池模块，避免多环境sys.path不一致导致的导入失败
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


# ============ 性能监控 ============
class PerformanceTimer:
    """简单的性能计时器"""
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def record(self, stage: str, elapsed: float):
        """记录耗时"""
        with self._lock:
            if stage not in self.timings:
                self.timings[stage] = []
            self.timings[stage].append(elapsed)
    
    def get_report(self) -> Dict:
        """获取性能报告"""
        report = {}
        with self._lock:
            for stage, times in self.timings.items():
                report[stage] = {
                    "total_sec": round(sum(times), 3),
                    "avg_sec": round(sum(times) / len(times), 3) if times else 0,
                    "count": len(times)
                }
        return report
    
    def reset(self):
        """重置"""
        with self._lock:
            self.timings.clear()


# 全局计时器
perf_timer = PerformanceTimer()


def timed_execution(stage_name: str):
    """执行时间装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            perf_timer.record(stage_name, elapsed)
            return result
        return wrapper
    return decorator


# ============ 基于内容匹配的参考文件查找 ============

def get_ref_file_by_content(input_wav_file, ref_dir):
    """
    使用DTW+嵌入匹配查找对应的参考文件。

    Args:
        input_wav_file: 测试音频文件路径
        ref_dir: 参考音频目录

    Returns:
        (ref_file_path, match_info) 或 (None, None)
        match_info: {"ref_id": str, "confidence": float, "offset": float, ...}
    """
    input_basename = os.path.basename(input_wav_file)
    input_stem = os.path.splitext(input_basename)[0]

    # 从切分文件名中提取 ref_tag（如 dut_ref_001_001.wav → ref_001.wav）
    # 这是 DTW 匹配结果的持久化映射，非文件名匹配
    import re
    ref_matches = re.findall(r'ref_\d+', input_stem)
    if ref_matches:
        # 取最后一个 ref_tag（文件名格式 {test}_{ref_tag}_{idx}.wav）
        ref_tag = ref_matches[-1]
        ref_candidate = os.path.join(ref_dir, ref_tag + '.wav')
        if os.path.exists(ref_candidate):
            logger.info(f"[参考匹配-切分映射] {input_basename} -> {ref_tag}.wav")
            return ref_candidate, {"method": "split_mapping", "ref_name": ref_tag}

    # 优化版多级回退匹配
    try:
        from matching_optimizer import OptimizedMatcher

        opt_matcher = OptimizedMatcher(ref_dir=ref_dir)
        match_result = opt_matcher.match_with_fallback(input_wav_file)

        if match_result["matched"] and match_result["ref_file"]:
            ref_file = match_result["ref_file"]
            ref_name = match_result.get("ref_name", os.path.basename(ref_file))
            method = match_result["method"]
            confidence = match_result["confidence"]
            offset = match_result["offset"]
            snr = match_result.get("snr", 0)

            logger.info(f"[参考匹配-优化] ✓ 优化匹配成功: {input_basename} -> {ref_name} "
                        f"(method={method}, confidence={confidence:.3f}, "
                        f"offset={offset:.2f}s, snr={snr:.1f}dB)")
            return ref_file, {
                "method": f"optimized_{method}",
                "ref_name": ref_name,
                "confidence": confidence,
                "offset": offset,
                "snr": snr,
                "detail": match_result.get("detail", {})
            }
        else:
            logger.warning(f"[参考匹配-优化] ✗ 优化匹配也失败: {input_basename}")
    except ImportError as e:
        logger.debug(f"[参考匹配-优化] 优化匹配模块不可用: {e}")
    except Exception as e:
        logger.error(f"[参考匹配-优化] 优化匹配异常: {e}")

    return None, None


def get_ref_ground_truth_text(ref_dir: str, ref_filename: str) -> Optional[str]:
    """
    从参考音频元数据中获取ground truth文本（用于WER计算）

    Args:
        ref_dir: 参考音频目录
        ref_filename: 参考音频文件名

    Returns:
        ground truth文本或None
    """
    import json
    metadata_file = os.path.join(ref_dir, ".metadata.json")
    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        audios = metadata.get("audios", {})
        for audio_id, info in audios.items():
            if info.get("filename") == ref_filename:
                gt_text = info.get("ground_truth_text")
                if gt_text:
                    logger.debug(f"[WER文本] 从元数据获取ground truth: {ref_filename} -> {gt_text[:30]}...")
                    return gt_text
    except Exception as e:
        logger.warning(f"[WER文本] 读取元数据失败: {e}")

    # 回退到硬编码的默认文本（兼容旧模式）
    fallback_texts = {
        'ref_001.wav': '他为儿子买了一整根甘蔗市区的停车收费将大幅提高他醒来后发现自己脸上有黑眼圈',
        'ref_002.wav': '大风刮倒了一处在建厂房姚大爷觉得车夫的想法蛮有道理汹涌的河水顺利而下流的很快',
        'ref_003.wav': '坚持终于让他有所收获据说这是当地最古老的小区你就是那个爱打篮球的人',
        'ref_004.wav': '总理对任何事情都要刨根问底渐渐的他还真就睡着了这身衣服就像被大雨淋过似的'
    }
    return fallback_texts.get(ref_filename)


# ============ 音频缓存 ============
class AudioCache:
    """音频文件缓存"""
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
    
    def get(self, file_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """获取缓存的音频"""
        with self._lock:
            if file_path in self._cache:
                # 更新访问顺序
                self._access_order.remove(file_path)
                self._access_order.append(file_path)
                return self._cache[file_path]
            return None
    
    def put(self, file_path: str, audio: np.ndarray, sr: int):
        """缓存音频"""
        with self._lock:
            if file_path in self._cache:
                self._access_order.remove(file_path)
            elif len(self._cache) >= self.max_size:
                # LRU淘汰
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
            
            self._cache[file_path] = (audio.copy(), sr)
            self._access_order.append(file_path)
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


# 全局音频缓存
audio_cache = AudioCache(max_size=50)


def preload_audio_cache(file_paths: List[str], sr: int = 16000):
    """
    预热音频缓存：在评分开始前将所有音频加载到缓存
    避免各评分模块重复读盘+重采样
    """
    count = 0
    for fp in file_paths:
        try:
            if os.path.exists(fp):
                load_audio_cached(fp, sr=sr)
                count += 1
        except Exception:
            pass
    if count > 0:
        logger.debug(f"[音频缓存] 预热完成: {count}/{len(file_paths)} 个文件")


def clear_audio_cache():
    """清空音频缓存（新任务开始时调用）"""
    audio_cache.clear()


def load_audio_cached(file_path: str, sr: int = None) -> Tuple[np.ndarray, int]:
    """带缓存的音频加载"""
    cache_key = f"{file_path}_{sr}"
    cached = audio_cache.get(cache_key)
    if cached is not None:
        return cached
    
    audio, orig_sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # 取第一个声道
    
    if sr is not None and orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        orig_sr = sr
    
    audio_cache.put(cache_key, audio, orig_sr)
    return audio, orig_sr


# ============ 优化的MOS计算类 ============

class OptimizedDNSMOScore:
    """优化的DNSMOS评分"""

    def __init__(self) -> None:
        import time
        print("\n[DNSMOS] 初始化DNSMOS模型...")
        start_time = time.time()
        
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # 优先检查 models/dnsmos/，然后检查 app/algorithms/dnsmos/
        p808_model_path = os.path.join(project_root, 'models', 'dnsmos', 'DNSMOS', 'model_v8.onnx')
        primary_model_path = os.path.join(project_root, 'models', 'dnsmos', 'pDNSMOS', 'sig_bak_ovr.onnx')
        
        print(f"[DNSMOS] 检查模型路径...")
        print(f"  项目模型路径: {os.path.join(project_root, 'models', 'dnsmos')}")
        
        if not os.path.exists(p808_model_path):
            print(f"  ⚠ 项目路径不存在，尝试备用路径...")
            p808_model_path = os.path.join(project_root, 'app', 'algorithms', 'dnsmos', 'DNSMOS', 'model_v8.onnx')
            primary_model_path = os.path.join(project_root, 'app', 'algorithms', 'dnsmos', 'pDNSMOS', 'sig_bak_ovr.onnx')
            print(f"  备用路径: {os.path.join(project_root, 'app', 'algorithms', 'dnsmos')}")
        
        # 检查模型文件是否存在
        if not os.path.exists(primary_model_path):
            raise FileNotFoundError(f"DNSMOS Primary模型未找到: {primary_model_path}")
        if not os.path.exists(p808_model_path):
            raise FileNotFoundError(f"DNSMOS P808模型未找到: {p808_model_path}")
        
        print(f"  ✓ Primary模型: {primary_model_path}")
        print(f"  ✓ P808模型: {p808_model_path}")
        
        # 使用CUDA加速
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda.is_available() else ['CPUExecutionProvider']
        print(f"[DNSMOS] 使用ONNX Runtime providers: {providers}")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        print(f"[DNSMOS] 加载ONNX模型...")
        load_start = time.time()
        self.onnx_sess = ort.InferenceSession(primary_model_path, sess_options=sess_options, providers=providers)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, sess_options=sess_options, providers=providers)
        load_time = time.time() - load_start
        
        self.INPUT_LENGTH = 9.01
        self.SAMPLING_RATE = 16000
        
        init_time = time.time() - start_time
        print(f"[DNSMOS] 模型初始化完成 (耗时: {init_time:.2f}s)")
        print(f"  ✓ DNSMOS评分器就绪")
    
    @staticmethod
    def __audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, 
            hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T
    
    @staticmethod
    def __get_polyfit_val(sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        
        return p_sig(sig), p_bak(bak), p_ovr(ovr)
    
    def _load_and_prepare_audio(self, fpath):
        """加载并准备音频（使用缓存，避免重复I/O）"""
        # 使用缓存加载，sr=16000 直接返回重采样后的数据
        audio, fs = load_audio_cached(fpath, sr=self.SAMPLING_RATE)

        len_samples = int(self.INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
        return audio, len_samples, num_hops

    def _run_segment_sequential(self, audio, len_samples, num_hops,
                                is_personalized_MOS=None):
        """逐段串行推理（回退模式）"""
        hop_len_samples = int(self.SAMPLING_RATE)
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + self.INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(self.__audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]

            p808_mos = self.p808_onnx_sess.run(None, {'input_1': p808_input_features})[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, {'input_1': input_features})[0][0]
            mos_sig, mos_bak, mos_ovr = self.__get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)

            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        return predicted_mos_sig_seg, predicted_mos_bak_seg, predicted_mos_ovr_seg, predicted_p808_mos

    def _run_segment_batch(self, audio, len_samples, num_hops,
                           is_personalized_MOS=None):
        """
        批处理模式: 收集所有段→一次ONNX推理
        消除Python循环中N次ONNX Runtime调用的框架开销
        """
        hop_len_samples = int(self.SAMPLING_RATE)

        # 收集所有段的输入张量
        all_audio_segs = []
        all_p808_segs = []
        valid_indices = []
        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + self.INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue
            all_audio_segs.append(np.array(audio_seg).astype('float32'))
            all_p808_segs.append(
                np.array(self.__audio_melspec(audio=audio_seg[:-160])).astype('float32')
            )
            valid_indices.append(idx)

        if not valid_indices:
            return [], [], [], []

        try:
            # 堆叠为batch: [N, L] 和 [N, T, n_mels]
            batch_input = np.stack(all_audio_segs)       # [N, 144160]
            batch_p808 = np.stack(all_p808_segs)          # [N, T, 120]

            # 一次ONNX推理全部段
            p808_out = self.p808_onnx_sess.run(None, {'input_1': batch_p808})[0]
            primary_out = self.onnx_sess.run(None, {'input_1': batch_input})[0]

            # 解析输出
            N = len(valid_indices)
            # p808_out shape: [N, 1, 1] 或 [N, 1]; 提取标量
            if p808_out.ndim == 3:
                p808_mos_all = p808_out[:, 0, 0]
            elif p808_out.ndim == 2:
                p808_mos_all = p808_out[:, 0]
            else:
                p808_mos_all = p808_out.flatten()

            # primary_out shape: [N, 3] (sig, bak, ovr)
            mos_sig_raw_all = primary_out[:, 0]
            mos_bak_raw_all = primary_out[:, 1]
            mos_ovr_raw_all = primary_out[:, 2]

            # 逐段应用polyfit（矢量化的poly1d操作）
            predicted_mos_sig_seg = []
            predicted_mos_bak_seg = []
            predicted_mos_ovr_seg = []
            predicted_p808_mos = []
            for i in range(N):
                mos_sig, mos_bak, mos_ovr = self.__get_polyfit_val(
                    mos_sig_raw_all[i], mos_bak_raw_all[i], mos_ovr_raw_all[i],
                    is_personalized_MOS
                )
                predicted_mos_sig_seg.append(mos_sig)
                predicted_mos_bak_seg.append(mos_bak)
                predicted_mos_ovr_seg.append(mos_ovr)
                predicted_p808_mos.append(float(p808_mos_all[i]))

            logger.debug(f"[DNSMOS批处理] 批大小={N}, 各段均已批处理推理")
            return predicted_mos_sig_seg, predicted_mos_bak_seg, predicted_mos_ovr_seg, predicted_p808_mos

        except Exception as batch_err:
            # batch推理失败（如模型固定batch dim=1），回退到逐段串行
            logger.debug(f"[DNSMOS批处理] 批处理失败({batch_err})，回退到串行模式")
            return self._run_segment_sequential(audio, len_samples, num_hops, is_personalized_MOS)

    @timed_execution("dnsmos")
    def get_mos(self, file_list):
        """计算dnsmos - 批处理+自动回退"""
        file_num = len(file_list)
        ovrl = [0.0 for _ in range(file_num)]
        sig = [0.0 for _ in range(file_num)]
        bak = [0.0 for _ in range(file_num)]
        p808mos = [0.0 for _ in range(file_num)]

        for file_index, clip in enumerate(file_list):
            try:
                audio, len_samples, num_hops = self._load_and_prepare_audio(clip)
                # 优先批处理，自动回退串行
                sig_seg, bak_seg, ovr_seg, p808_seg = self._run_segment_batch(
                    audio, len_samples, num_hops
                )
                ovrl[file_index] = float(np.mean(ovr_seg)) if ovr_seg else 0.0
                sig[file_index] = float(np.mean(sig_seg)) if sig_seg else 0.0
                bak[file_index] = float(np.mean(bak_seg)) if bak_seg else 0.0
                p808mos[file_index] = float(np.mean(p808_seg)) if p808_seg else 0.0
            except Exception as e:
                print(f"DNSMOS计算失败 {clip}: {e}")

        return {'OVRL': ovrl, 'SIG': sig, 'BAK': bak, 'P808_MOS': p808mos}


class OptimizedNisqaMosScore:
    """优化的NISQA评分"""

    def __init__(self):
        import time
        print("\n[NISQA] 初始化NISQA模型...")
        start_time = time.time()
        
        if not NISQA_AVAILABLE:
            raise ImportError("nisqa未安装")
        
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # 检查模型路径
        model_path_project = os.path.join(project_root, 'models', 'nisqa', 'weights', 'nisqa_3000.tar')
        model_path_algo = os.path.join(project_root, 'app', 'algorithms', 'nisqa', 'weights', 'nisqa_3000.tar')
        
        print(f"[NISQA] 检查模型路径...")
        if os.path.exists(model_path_project):
            self.nisqa_model = model_path_project
            print(f"  ✓ 使用项目路径模型: {model_path_project}")
        elif os.path.exists(model_path_algo):
            self.nisqa_model = model_path_algo
            print(f"  ✓ 使用算法路径模型: {model_path_algo}")
        else:
            # 使用默认文件名，让nisqa自己查找
            self.nisqa_model = 'nisqa_3000.tar'
            print(f"  ⚠ 本地模型未找到，使用默认配置: {self.nisqa_model}")
        
        self.nisqa_mode = "predict_list"
        
        init_time = time.time() - start_time
        print(f"[NISQA] 模型初始化完成 (耗时: {init_time:.2f}s)")
        print(f"  ✓ NISQA评分器就绪")
    
    @timed_execution("nisqa")
    def get_mos(self, file_dir_list) -> dict:
        """计算nisqa分 - 使用列表模式批量处理"""
        file_num = len(file_dir_list)
        
        # 使用predict_list模式一次性处理所有文件
        nisqa_prediction = nisqa_predict(
            mode=self.nisqa_mode,
            deg_list=file_dir_list,
            model=self.nisqa_model,
            bs=10,  # 增加batch size
            num_workers=0  # 使用主进程加载（多线程环境下num_workers>0会死锁）
        )
        
        ret = nisqa_prediction.to_dict(orient='list')
        ret.pop("deg", None)
        
        # 验证输出长度
        for k, v in ret.items():
            if len(v) != file_num:
                print(f"Length of {k}: {len(v)} does not match input length {file_num}")
                if len(v) < file_num:
                    v.extend([0.0] * (file_num - len(v)))
                else:
                    v = v[:file_num]
                ret[k] = v
        
        return ret


class OptimizedScoreqScore:
    """优化的Scoreq评分"""
    
    def __init__(self, data_domain='natural', mode='nr'):
        if scoreq is None:
            raise ImportError("scoreq未安装")
        self.pred_mos_ins = scoreq.Scoreq(data_domain=data_domain, mode=mode)
    
    @timed_execution("scoreq")
    def get_mos(self, file_dir_list):
        """计算scoreq分 - 使用共享线程池并行"""
        score_list = [0.0 for _ in range(len(file_dir_list))]

        def process_single(file_path, idx):
            try:
                return idx, self.pred_mos_ins.predict(file_path)
            except Exception as e:
                print(f"Scoreq计算失败 {file_path}: {e}")
                return idx, 0.0

        # 使用全局共享线程池（复用，避免每次创建销毁）
        executor = get_shared_executor()
        futures = [executor.submit(process_single, f, i) for i, f in enumerate(file_dir_list)]
        for future in futures:
            idx, score = future.result()
            score_list[idx] = score

        return {"scoreq": score_list}


class OptimizedRefScore:
    """优化的带参考评分"""
    
    def __init__(self):
        if speechmetrics is None:
            raise ImportError("speechmetrics未安装")
        self.metrics = speechmetrics.load(["stoi", "sisdr"])
    
    @staticmethod
    def get_ref_file(input_wav_file, ref_dir):
        """获取参考文件 - 支持多种匹配方式（含内容匹配）"""
        # 先尝试内容增强匹配（包含文件名匹配和指纹匹配）
        ref_file, match_info = get_ref_file_by_content(input_wav_file, ref_dir)
        if ref_file is not None:
            logger.info(f"[RefScore] 匹配参考文件(方法={match_info.get('method')}): "
                         f"{os.path.basename(input_wav_file)} -> {os.path.basename(ref_file)}")
            return ref_file
        return None

    @timed_execution("ref_score")
    def get_mos(self, file_list, ref_dir):
        """计算stoi、sisdr、pesq分 — 多文件并行"""
        file_num = len(file_list)
        STOI = [0.0 for _ in range(file_num)]
        SISDR = [0.0 for _ in range(file_num)]
        PESQ = [0.0 for _ in range(file_num)]

        logger.info(f"\n[RefScore] 开始计算参考相关指标，文件数: {file_num}")
        logger.info(f"[RefScore] 参考音频目录: {ref_dir}")

        if not os.path.exists(ref_dir):
            logger.warning(f"[RefScore] ❌ 参考目录不存在: {ref_dir}")
            return {'STOI': STOI, 'SISDR': SISDR, 'pesq': PESQ}

        ref_files = [f for f in os.listdir(ref_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        logger.info(f"[RefScore] 参考目录中的音频文件: {ref_files}")

        def _process_one_file(file_index: int, file: str) -> tuple:
            """处理单个文件的STOI/SISDR/PESQ"""
            try:
                path_to_reference = self.get_ref_file(file, ref_dir)
                if path_to_reference is None:
                    return file_index, 0.0, 0.0, 0.0

                # stoi + sisdr（speechmetrics内部已优化）
                scores = self.metrics(file, path_to_reference)
                stoi_val = float(scores['stoi'].mean())
                sisdr_val = float(scores['sisdr'].mean())

                # pesq（CPU计算密集，释放GIL，适合线程并行）
                pesq_val = 0.0
                if PESQ_AVAILABLE:
                    est, _ = load_audio_cached(file, sr=16000)
                    ref, _ = load_audio_cached(path_to_reference, sr=16000)
                    rms_ref = np.sqrt(np.mean(ref**2))
                    rms_est = np.sqrt(np.mean(est**2))
                    if rms_est > 0:
                        gain = rms_ref / rms_est
                        est = est * gain
                    pesq_val = float(pesq.pesq(fs=16000, ref=ref, deg=est, mode='wb'))

                return file_index, stoi_val, sisdr_val, pesq_val
            except Exception as e:
                logger.error(f"[RefScore] ❌ {os.path.basename(file)} 计算失败: {e}")
                return file_index, 0.0, 0.0, 0.0

        # 使用共享线程池并行处理所有文件（PESQ是CPU密集但释放GIL，多线程有效）
        ref_executor = get_shared_executor()
        futures = {
            ref_executor.submit(_process_one_file, i, f): i
            for i, f in enumerate(file_list)
        }
        matched_count = 0
        for future in as_completed(futures):
            idx, stoi, sisdr, pesq_val = future.result()
            STOI[idx] = stoi
            SISDR[idx] = sisdr
            PESQ[idx] = pesq_val
            if stoi > 0 or pesq_val > 0:
                matched_count += 1

        logger.info(f"\n[RefScore] 计算完成: {matched_count}/{file_num} 个文件匹配到参考, "
                     f"STOI非零: {sum(1 for s in STOI if s != 0)}, "
                     f"SISDR非零: {sum(1 for s in SISDR if s != 0)}, "
                     f"PESQ非零: {sum(1 for p in PESQ if p != 0)}")
        return {'STOI': STOI, 'SISDR': SISDR, 'pesq': PESQ}


class OptimizedWerScore:
    """优化的WER评分"""
    
    def __init__(self):
        import time
        print("\n[WeNet] 初始化WeNet模型...")
        start_time = time.time()
        
        if not WENET_AVAILABLE:
            raise ImportError("wenet未安装")
        
        # 项目路径下的模型目录(优先)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        project_model_path = os.path.join(project_root, "models", "wenet")
        
        # 本地缓存路径(备选)
        local_model_path = os.path.expanduser("~/.wenet/wenetspeech")
        
        print(f"[WeNet] 检查模型路径...")
        print(f"  项目模型路径: {project_model_path}")
        print(f"  本地缓存路径: {local_model_path}")
        
        load_start = time.time()
        # 优先使用项目路径下的模型
        if os.path.exists(project_model_path) and os.path.exists(os.path.join(project_model_path, "final.pt")):
            print(f"  ✓ 找到项目路径模型")
            print(f"[WeNet] 加载模型...")
            self.model = wenet.load_model(project_model_path)
            load_time = time.time() - load_start
            print(f"  ✓ 模型加载完成 (耗时: {load_time:.2f}s)")
        elif os.path.exists(local_model_path):
            print(f"  ✓ 找到本地缓存模型")
            print(f"[WeNet] 加载模型...")
            self.model = wenet.load_model(local_model_path)
            load_time = time.time() - load_start
            print(f"  ✓ 模型加载完成 (耗时: {load_time:.2f}s)")
        else:
            print(f"  ⚠ 本地模型未找到，将从网络下载...")
            print(f"[WeNet] 下载并加载模型...")
            self.model = wenet.load_model("wenetspeech")
            load_time = time.time() - load_start
            print(f"  ✓ 模型下载并加载完成 (耗时: {load_time:.2f}s)")
        
        init_time = time.time() - start_time
        print(f"[WeNet] 模型初始化完成 (总耗时: {init_time:.2f}s)")
        print(f"  ✓ WeNet语音识别器就绪")
    
    def __get_ref_gt_text(self, input_wav_file, ref_dir=None):
        """获取参考文本 - 优先从元数据获取，回退到硬编码默认值"""
        # 先尝试从参考音频匹配获取文本
        if ref_dir:
            ref_file, match_info = get_ref_file_by_content(input_wav_file, ref_dir)
            if ref_file:
                ref_filename = os.path.basename(ref_file)
                gt_text = get_ref_ground_truth_text(ref_dir, ref_filename)
                if gt_text:
                    logger.info(f"[WER] 获取ground truth文本: {ref_filename} -> {gt_text[:30]}...")
                    return gt_text

        # 回退到硬编码默认文本
        fallback_texts = {
            '001': '他为儿子买了一整根甘蔗市区的停车收费将大幅提高他醒来后发现自己脸上有黑眼圈',
            '002': '大风刮倒了一处在建厂房姚大爷觉得车夫的想法蛮有道理汹涌的河水顺利而下流的很快',
            '003': '坚持终于让他有所收获据说这是当地最古老的小区你就是那个爱打篮球的人',
            '004': '总理对任何事情都要刨根问底渐渐的他还真就睡着了这身衣服就像被大雨淋过似的'
        }
        input_file_name = os.path.basename(input_wav_file).removesuffix('.wav')
        suffix = input_file_name[-3:]
        text = fallback_texts.get(suffix)
        if text:
            logger.debug(f"[WER] 使用硬编码ground truth: suffix={suffix}")
        else:
            logger.warning(f"[WER] 未找到ground truth文本 for {input_wav_file}，WER将返回0")
        return text
    
    @timed_execution("wer")
    def get_wer(self, file_dir_list, ref_dir=None):
        """计算wer - 多文件并行转写"""
        file_num = len(file_dir_list)
        wer_data = [0.0 for _ in range(file_num)]
        wcorr = [0.0 for _ in range(file_num)]

        def _process_one_file(file_index: int, file: str) -> tuple:
            """单个文件的ASR转写+WER计算"""
            try:
                result = self.model.transcribe(file)
                ref = self.__get_ref_gt_text(file, ref_dir=ref_dir)
                if ref is None:
                    return file_index, 0.0, 0.0

                if hasattr(result, 'text'):
                    text = result.text
                elif isinstance(result, dict):
                    text = result['text']
                else:
                    text = str(result)

                from wer import wer
                tmp_wer, tmp_wcorr = wer(ref, text)
                return file_index, tmp_wer, tmp_wcorr
            except Exception as e:
                logger.error(f"WER计算失败 {file}: {e}")
                return file_index, 0.0, 0.0

        # 使用共享线程池并行转写（WeNet模型GPU推理为主，但多文件可交错CPU/GPU）
        wer_executor = get_shared_executor()
        futures = {
            wer_executor.submit(_process_one_file, i, f): i
            for i, f in enumerate(file_dir_list)
        }
        for future in as_completed(futures):
            idx, w, c = future.result()
            wer_data[idx] = w
            wcorr[idx] = c

        return {'wer': wer_data, 'wcorr': wcorr}


class OptimizedToneColorFidelityScore:
    """优化的音色还原度评分 - 支持多模型加权评估"""

    # GPU显存预算比例：超过此阈值时将新模型加载到CPU而非GPU
    # 避免6个TCF管线同时占用GPU导致OOM
    _GPU_MEM_BUDGET_RATIO = 0.80  # 80%阈值，可通过类变量覆盖
    # TCF模型最大并发数：限制同时推理的模型数量以控制峰值显存
    # 2路并发约需6-8GB显存，6路全开可能超24GB（尤其与大模型共存时）
    _TCF_MAX_CONCURRENT = 2

    def __init__(self):
        import time
        logger.info("[TCF] 初始化音色还原度评分模型...")
        start_time = time.time()

        if not MODELSCOPE_AVAILABLE:
            raise ImportError("modelscope未安装")

        # 多模型配置，权重根据ERR值得到，ERR越大错误率越高权重越低
        # 项目路径下的模型目录(优先)
        # 计算项目根目录: app/algorithms/tcf/ -> 项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        logger.info(f"[TCF] 项目根目录: {project_root}")
        logger.info(f"[TCF] 配置多模型加权评估...")

        # 多模型配置，权重根据ERR值得到: weight = 10 - ERR
        # ERR越小(性能越好)，权重越高
        self.sv_model_dict = {
            "eres2net": {
                "model_id": "damo/speech_eres2net_sv_zh-cn_16k-common",
                "project_path": os.path.join(project_root, "models", "tcf", "eres2net"),
                "cache_path": os.path.expanduser("~/.cache/modelscope/hub/damo/speech_eres2net_sv_zh-cn_16k-common"),
                "weight": 7.21,  # 10 - 2.79
                "revision": "v1.0.0"
            },
            "eres2netv2": {
                "model_id": "damo/speech_eres2netv2_sv_zh-cn_16k-common",
                "project_path": os.path.join(project_root, "models", "tcf", "eres2netv2"),
                "cache_path": os.path.expanduser("~/.cache/modelscope/hub/damo/speech_eres2netv2_sv_zh-cn_16k-common"),
                "weight": 6.19,  # 10 - 3.81
                "revision": "v1.0.0"
            },
            "campplus": {
                "model_id": "damo/speech_campplus_sv_zh-cn_16k-common",
                "project_path": os.path.join(project_root, "models", "tcf", "campplus"),
                "cache_path": os.path.expanduser("~/.cache/modelscope/hub/damo/speech_campplus_sv_zh-cn_16k-common"),
                "weight": 5.0,   # 根据实际ERR调整
                "revision": "v1.0.0"
            },
            "ecapa-tdnn": {
                "model_id": "damo/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k",
                "project_path": os.path.join(project_root, "models", "tcf", "ecapa-tdnn"),
                "cache_path": os.path.expanduser("~/.cache/modelscope/hub/damo/speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k"),
                "weight": 4.5,
                "revision": "v1.0.0"
            },
            "res2net": {
                "model_id": "damo/speech_res2net_sv_zh-cn_3dspeaker_16k",
                "project_path": os.path.join(project_root, "models", "tcf", "res2net"),
                "cache_path": os.path.expanduser("~/.cache/modelscope/hub/damo/speech_res2net_sv_zh-cn_3dspeaker_16k"),
                "weight": 5.0,   # 10 - 5
                "revision": "v1.0.0"
            },
            "resnet34": {
                "model_id": "damo/speech_resnet34_sv_zh-cn_3dspeaker_16k",
                "project_path": os.path.join(project_root, "models", "tcf", "resnet34"),
                "cache_path": os.path.expanduser("~/.cache/modelscope/hub/damo/speech_resnet34_sv_zh-cn_3dspeaker_16k"),
                "weight": 3.03,  # 10 - 6.97
                "revision": "v1.0.0"
            }
        }
        
        self._pipeline_cache = {}
        self._init_error = None
    
    def _check_and_fix_tcf_config(self, model_path: str) -> None:
        """检查并修复 eres2netv2 的 configuration.json（ModelScope 官方配置缺少必需字段）"""
        config_file = os.path.join(model_path, "configuration.json")
        if not os.path.exists(config_file):
            return
        
        try:
            import json
            with open(config_file, 'r') as f:
                cfg = json.load(f)
            
            model_cfg = cfg.get('model', {})
            mc = model_cfg.get('model_config', {})
            
            # eres2netv2 需要这些字段，但 ModelScope Hub 官方配置缺失
            required_fields = {
                'embed_dim': 192,
                'baseWidth': 26,
                'scale': 2,
                'expansion': 2
            }
            
            missing = {k: v for k, v in required_fields.items() if k not in mc}
            if missing:
                logger.info(f"[TCF] 修复 eres2netv2 configuration.json，补充缺失字段: {list(missing.keys())}")
                mc.update(missing)
                with open(config_file, 'w') as f:
                    json.dump(cfg, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[TCF] 检查/修复 configuration.json 失败: {e}")

    def _check_model_exists(self, model_config: dict) -> tuple:
        """检查模型是否存在，返回(是否存在, 实际路径, 是否项目路径)"""
        # 优先检查项目路径
        if os.path.exists(model_config["project_path"]):
            config_file = os.path.join(model_config["project_path"], "configuration.json")
            if os.path.exists(config_file):
                return True, model_config["project_path"], True
        
        # 其次检查本地缓存
        if os.path.exists(model_config["cache_path"]):
            config_file = os.path.join(model_config["cache_path"], "configuration.json")
            if os.path.exists(config_file):
                return True, model_config["cache_path"], False
        
        return False, None, False
    
    def _get_pipeline(self, alg: str):
        """获取或创建指定算法的pipeline"""
        if alg not in self._pipeline_cache:
            model_config = self.sv_model_dict[alg]
            exists, model_path, is_project = self._check_model_exists(model_config)

            if not exists:
                raise FileNotFoundError(
                    f"TCF模型 [{alg}] 不存在。\n"
                    f"  项目路径: {model_config['project_path']}\n"
                    f"  缓存路径: {model_config['cache_path']}\n"
                    f"  离线部署前请将模型放入以上任一目录。\n"
                    f"  下载方式: python download_tcf_models.py"
                )

            location = "项目路径" if is_project else "本地缓存"
            logger.info(f"[TCF] 使用{location}模型 [{alg}]: {model_path}")

            # 修复 eres2netv2 的 configuration.json（ModelScope 官方配置缺少必需字段）
            if alg == 'eres2netv2':
                self._check_and_fix_tcf_config(model_path)

            # 加载前清理CUDA缓存，释放之前模型占用的显存
            device = 'cuda' if cuda.is_available() else 'cpu'
            mem_free_gb = mem_total_gb = None
            if cuda.is_available():
                cuda.empty_cache()
                try:
                    mem_free, mem_total = cuda.mem_get_info()
                    mem_free_gb = mem_free / (1024 ** 3)
                    mem_total_gb = mem_total / (1024 ** 3)
                    mem_used_ratio = 1.0 - (mem_free / mem_total)
                    # GPU显存预算管理：使用率超过80%时自动回退到CPU
                    if mem_used_ratio > self._GPU_MEM_BUDGET_RATIO:
                        device = 'cpu'
                        logger.warning(
                            f"[TCF] GPU显存使用率{mem_used_ratio:.0%}>"
                            f"{self._GPU_MEM_BUDGET_RATIO:.0%}, "
                            f"模型[{alg}]将在CPU上加载"
                        )
                    logger.info(f"[TCF] GPU显存: {mem_free_gb:.1f}GB 可用 / "
                                f"{mem_total_gb:.1f}GB 总量 "
                                f"(使用率: {mem_used_ratio:.0%}, "
                                f"预算: {self._GPU_MEM_BUDGET_RATIO:.0%})")
                except Exception:
                    dev_info = f"device={device}"
                    if mem_free_gb is not None:
                        dev_info += f", mem_free={mem_free_gb:.1f}GB"
                    logger.info(f"[TCF] GPU显存检查: {dev_info}")

            try:
                # 使用catch_warnings抑制废弃API警告（防止 pynvml 等 FutureWarning 在
                # PYTHONWARNINGS=error 环境下变成致命异常，导致 pipeline 创建失败）
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    self._pipeline_cache[alg] = pipeline(
                        task='speaker-verification',
                        model=model_path,
                        device=device
                    )
                logger.info(f"[TCF] ✓ [{alg}] pipeline 创建成功 (device={device})")
            except Exception as e:
                import traceback
                err_detail = traceback.format_exc()
                if cuda.is_available() and mem_free_gb is not None:
                    logger.error(
                        f"[TCF] ✗ 模型 [{alg}] 初始化失败\n"
                        f"  设备: {device}\n"
                        f"  GPU显存: {mem_free_gb:.1f}GB 可用 / {mem_total_gb:.1f}GB 总量\n"
                        f"  错误类型: {type(e).__name__}\n"
                        f"  错误信息: {e}\n"
                        f"  详细堆栈:\n{err_detail}"
                    )
                else:
                    logger.error(
                        f"[TCF] ✗ 模型 [{alg}] 初始化失败\n"
                        f"  设备: {device}\n"
                        f"  错误类型: {type(e).__name__}\n"
                        f"  错误信息: {e}\n"
                        f"  详细堆栈:\n{err_detail}"
                    )
                raise

        return self._pipeline_cache[alg]
    
    @staticmethod
    def _compare_speakers(features1, features2):
        """计算余弦相似度"""
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )
        return similarity
    
    @staticmethod
    def get_ref_file(input_wav_file, ref_dir):
        """获取参考文件 - 支持多种匹配方式（含内容匹配，与RefScore保持一致）"""
        # 使用增强版匹配（文件名 + 内容匹配）
        ref_file, match_info = get_ref_file_by_content(input_wav_file, ref_dir)
        if ref_file is not None:
            logger.info(f"[TCF] 匹配参考文件(方法={match_info.get('method')}): "
                         f"{os.path.basename(input_wav_file)} -> {os.path.basename(ref_file)}")
            return ref_file
        return None
    
    @timed_execution("tcf")
    def get_mos(self, input_test_file_list, ref_dir):
        """计算音色还原度 - 多模型加权评估"""
        test_file_list = input_test_file_list.copy()
        file_num = len(test_file_list)

        # {file: {algorithm: {"embedding": embedding, "score": score}}}
        file_embedding_score_dict = {}
        total_score_list = [0.0 for _ in range(file_num)]

        # 给待测音频注册
        for test_file in test_file_list:
            file_embedding_score_dict[test_file] = {}

        # 给参考音频注册并加入待分析列表
        ref_files_in_dir = [f for f in os.listdir(ref_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        logger.info(f"[TCF] 参考目录: {ref_dir}, 参考音频文件数: {len(ref_files_in_dir)}")
        if ref_files_in_dir:
            logger.info(f"[TCF] 参考文件列表: {ref_files_in_dir}")
        else:
            logger.warning(f"[TCF] 参考目录中没有音频文件，TCF评分将全部为0")

        for ref_file in ref_files_in_dir:
            ref_file_full_path = os.path.join(ref_dir, ref_file)
            file_embedding_score_dict[ref_file_full_path] = {}
            test_file_list.append(ref_file_full_path)

        # 预先检查每个测试文件是否能找到参考文件（结果缓存，供评分循环复用，避免二次匹配）
        ref_file_cache = {}  # {test_file: ref_file_path}
        no_ref_files = []
        for file_index in range(file_num):
            file = input_test_file_list[file_index]
            ref_file = self.get_ref_file(file, ref_dir)
            if ref_file is None or ref_file == file or ref_file not in file_embedding_score_dict:
                no_ref_files.append(os.path.basename(file))
            else:
                ref_file_cache[file] = ref_file
        if no_ref_files:
            logger.warning(f"[TCF] 以下文件未找到匹配的参考音频，TCF评分将为0: {no_ref_files}")
            logger.warning(f"[TCF] 文件命名需匹配 ref_ 模式，或包含 _XXX 后缀")

        # ============ 并行计算所有SV模型嵌入向量 ============
        available_algs = []
        failed_algs = []
        file_embedding_lock = threading.Lock()

        def _run_single_model(alg: str) -> tuple:
            """
            运行单个SV模型，返回(alg, embeddings_dict, error_or_None)
            每个模型在独立线程中运行，thread-safe使用 _pipeline_cache
            """
            try:
                logger.info(f"[TCF] [{alg}] 开始并行计算，共{len(test_file_list)}个音频...")
                sv_pipeline = self._get_pipeline(alg)
                result = sv_pipeline(test_file_list, output_emb=True)

                # 构建 {file_path: embedding} 映射
                all_embs = result['embs']
                embeddings = {}
                for i in range(len(all_embs)):
                    embeddings[test_file_list[i]] = all_embs[i]

                # 生产环境(4090 24GB显存)足够容纳所有模型，保留pipeline缓存以提高性能
                # 如需清理缓存（显存不足时），可设置环境变量 TCF_CLEAR_CACHE=1
                if os.environ.get('TCF_CLEAR_CACHE', '0') == '1':
                    if alg in self._pipeline_cache:
                        del self._pipeline_cache[alg]
                        logger.info(f"[TCF] [{alg}] pipeline缓存已清理 (TCF_CLEAR_CACHE=1)")

                logger.info(f"[TCF] ✓ [{alg}] 并行计算完成")
                return alg, embeddings, None
            except Exception as e:
                logger.warning(f"[TCF] ✗ [{alg}] 模型计算失败: {e}")
                return alg, None, e

        # TCF模型并行执行（限制并发数防OOM）
        # _TCF_MAX_CONCURRENT 控制GPU峰值显存，2路并发约需6-8GB
        _max_concurrent = getattr(self, '_TCF_MAX_CONCURRENT', 2)
        _tcf_executor = ThreadPoolExecutor(
            max_workers=_max_concurrent,
            thread_name_prefix='tcf_parallel'
        )
        try:
            tcf_futures = {
                _tcf_executor.submit(_run_single_model, alg): alg
                for alg in self.sv_model_dict.keys()
            }

            for future in as_completed(tcf_futures):
                alg, embeddings, error = future.result()
                if error is None:
                    available_algs.append(alg)
                    with file_embedding_lock:
                        for fpath, emb in embeddings.items():
                            file_embedding_score_dict[fpath][alg] = {"embedding": emb}
                else:
                    failed_algs.append(alg)
        finally:
            _tcf_executor.shutdown(wait=True)

        if not available_algs:
            logger.error(f"[TCF] 所有TCF模型都不可用！失败的模型: {failed_algs}")
            logger.error(f"[TCF] TCF评分全部为0，请检查ModelScope和模型文件配置")
            return {"tcf": total_score_list}

        if failed_algs:
            logger.warning(f"[TCF] 部分模型失败: {failed_algs}, 可用模型: {available_algs}")

        logger.info(f"[TCF] 可用算法模型: {available_algs}")
        logger.info(f"[TCF] 各算法权重: {[(alg, self.sv_model_dict[alg]['weight']) for alg in available_algs]}")

        # 计算加权得分（使用预检阶段缓存的ref_file_cache，避免二次匹配）
        matched_count = 0
        for file_index in range(file_num):
            file = input_test_file_list[file_index]
            ref_file = ref_file_cache.get(file)

            if ref_file is not None and ref_file != file and ref_file in file_embedding_score_dict:
                file_total_score = 0.0
                total_weight = 0.0
                alg_scores = []

                for alg in available_algs:
                    if alg in file_embedding_score_dict[file] and alg in file_embedding_score_dict[ref_file]:
                        # 计算相似度
                        similarity = self._compare_speakers(
                            file_embedding_score_dict[file][alg]["embedding"],
                            file_embedding_score_dict[ref_file][alg]["embedding"]
                        )
                        file_embedding_score_dict[file][alg]["score"] = similarity

                        # 加权累加
                        weight = self.sv_model_dict[alg]["weight"]
                        file_total_score += similarity * weight
                        total_weight += weight
                        alg_scores.append((alg, similarity, weight, similarity * weight))

                # 归一化并存储结果
                if total_weight > 0:
                    final_score = float(file_total_score / total_weight)
                    total_score_list[file_index] = final_score
                    matched_count += 1

                    # 使用debug级别记录详细结果
                    logger.debug(f"[TCF] 文件: {os.path.basename(file)}, 参考: {os.path.basename(ref_file)}, TCF={final_score:.4f}")
                    for alg, sim, w, weighted in alg_scores:
                        logger.debug(f"[TCF]   {alg}: sim={sim:.4f}, w={w}, weighted={weighted:.4f}")
            else:
                logger.warning(f"[TCF] 未找到参考音频文件: {os.path.basename(file)}, TCF=0.0")

        # 汇总统计
        if matched_count == 0:
            logger.error(f"[TCF] 所有测试文件均未匹配到参考音频！TCF评分全部为0")
            logger.error(f"[TCF] 请检查文件命名格式和参考目录配置")
        else:
            non_zero = [s for s in total_score_list if s > 0]
            avg_tcf = np.mean(non_zero) if non_zero else 0.0
            logger.info(f"[TCF] TCF计算完成: {matched_count}/{file_num} 个文件匹配成功, 平均TCF={avg_tcf:.4f}")

        return {"tcf": total_score_list}


# ============ 并行计算控制器 ============

class ParallelMOSCompute:
    """并行MOS计算控制器"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.models = {}
        self._init_start_time = None
        logger.info("[ParallelMOSCompute] 实例创建完成，等待模型初始化...")

    def init_models(self):
        """初始化所有模型"""
        # 确保CUDA按配置初始化（延迟加载，避免import时全局副作用）
        init_cuda_from_config()
        self._init_start_time = time.time()
        logger.info("=" * 60)
        logger.info("[MOS模型初始化] 开始初始化所有MOS计算模型")
        logger.info(f"[MOS模型初始化] 当前已加载模型: {list(self.models.keys())}")
        logger.info("=" * 60)

        init_stats = {'success': [], 'failed': [], 'skipped': []}

        # 无参考模型
        logger.info("[MOS模型初始化] === 无参考模型初始化 ===")

        try:
            logger.info("[MOS模型初始化] [1/7] 正在加载 DNSMOS 模型...")
            model_start = time.time()
            self.models['dnsmos'] = OptimizedDNSMOScore()
            model_time = time.time() - model_start
            init_stats['success'].append(('DNSMOS', model_time))
            logger.info(f"[MOS模型初始化] ✓ DNSMOS 模型加载成功 (耗时: {model_time:.2f}s)")
        except Exception as e:
            init_stats['failed'].append(('DNSMOS', str(e)))
            logger.error(f"[MOS模型初始化] ✗ DNSMOS 模型加载失败: {e}")

        try:
            logger.info("[MOS模型初始化] [2/7] 正在加载 NISQA 模型...")
            model_start = time.time()
            self.models['nisqa'] = OptimizedNisqaMosScore()
            model_time = time.time() - model_start
            init_stats['success'].append(('NISQA', model_time))
            logger.info(f"[MOS模型初始化] ✓ NISQA 模型加载成功 (耗时: {model_time:.2f}s)")
        except Exception as e:
            init_stats['failed'].append(('NISQA', str(e)))
            logger.error(f"[MOS模型初始化] ✗ NISQA 模型加载失败: {e}")

        try:
            logger.info("[MOS模型初始化] [3/7] 正在加载 Scoreq 模型...")
            model_start = time.time()
            self.models['scoreq'] = OptimizedScoreqScore()
            model_time = time.time() - model_start
            init_stats['success'].append(('Scoreq', model_time))
            logger.info(f"[MOS模型初始化] ✓ Scoreq 模型加载成功 (耗时: {model_time:.2f}s)")
        except Exception as e:
            init_stats['failed'].append(('Scoreq', str(e)))
            logger.error(f"[MOS模型初始化] ✗ Scoreq 模型加载失败: {e}")

        # 有参考模型
        logger.info("[MOS模型初始化] === 有参考模型初始化 ===")

        try:
            logger.info("[MOS模型初始化] [4/7] 正在加载 RefScore 模型 (PESQ/STOI/SISDR)...")
            model_start = time.time()
            self.models['ref_score'] = OptimizedRefScore()
            model_time = time.time() - model_start
            init_stats['success'].append(('RefScore', model_time))
            logger.info(f"[MOS模型初始化] ✓ RefScore 模型加载成功 (耗时: {model_time:.2f}s)")
        except Exception as e:
            init_stats['failed'].append(('RefScore', str(e)))
            logger.error(f"[MOS模型初始化] ✗ RefScore 模型加载失败: {e}")

        try:
            logger.info("[MOS模型初始化] [5/7] 正在加载 WER 模型 (WeNet)...")
            model_start = time.time()
            self.models['wer'] = OptimizedWerScore()
            model_time = time.time() - model_start
            init_stats['success'].append(('WER', model_time))
            logger.info(f"[MOS模型初始化] ✓ WER 模型加载成功 (耗时: {model_time:.2f}s)")
        except Exception as e:
            init_stats['failed'].append(('WER', str(e)))
            logger.error(f"[MOS模型初始化] ✗ WER 模型加载失败: {e}")

        try:
            logger.info("[MOS模型初始化] [6/7] 正在加载 TCF 模型 (音色还原度，6个子模型)...")
            model_start = time.time()
            self.models['tcf'] = OptimizedToneColorFidelityScore()
            model_time = time.time() - model_start
            init_stats['success'].append(('TCF', model_time))
            logger.info(f"[MOS模型初始化] ✓ TCF 模型加载成功 (耗时: {model_time:.2f}s)")
        except Exception as e:
            init_stats['failed'].append(('TCF', str(e)))
            logger.error(f"[MOS模型初始化] ✗ TCF 模型加载失败: {e}")
            import traceback
            logger.error(f"[MOS模型初始化] TCF错误详情: {traceback.format_exc()}")

        # UTMOS模型
        logger.info("[MOS模型初始化] === UTMOS模型初始化 ===")
        try:
            if UTMOS_AVAILABLE:
                logger.info("[MOS模型初始化] [7/7] 正在加载 UTMOS 模型...")
                model_start = time.time()
                self.models['utmos'] = UTMOSCore()
                model_time = time.time() - model_start
                init_stats['success'].append(('UTMOS', model_time))
                logger.info(f"[MOS模型初始化] ✓ UTMOS 模型加载成功 (耗时: {model_time:.2f}s)")
            else:
                init_stats['skipped'].append(('UTMOS', '模块未安装'))
                logger.warning("[MOS模型初始化] ⚠ UTMOS模块未安装，跳过加载")
        except Exception as e:
            init_stats['failed'].append(('UTMOS', str(e)))
            logger.error(f"[MOS模型初始化] ✗ UTMOS 模型加载失败: {e}")
            import traceback
            logger.error(f"[MOS模型初始化] UTMOS错误详情: {traceback.format_exc()}")

        # 汇总统计
        total_time = time.time() - self._init_start_time
        logger.info("=" * 60)
        logger.info("[MOS模型初始化] 初始化完成统计")
        logger.info(f"[MOS模型初始化] 总耗时: {total_time:.2f}s")
        logger.info(f"[MOS模型初始化] 成功: {len(init_stats['success'])}个 - {[m[0] for m in init_stats['success']]}")
        if init_stats['failed']:
            logger.warning(f"[MOS模型初始化] 失败: {len(init_stats['failed'])}个 - {[m[0] for m in init_stats['failed']]}")
        if init_stats['skipped']:
            logger.info(f"[MOS模型初始化] 跳过: {len(init_stats['skipped'])}个 - {[m[0] for m in init_stats['skipped']]}")
        logger.info(f"[MOS模型初始化] 当前已加载模型: {list(self.models.keys())}")
        logger.info("=" * 60)

        return init_stats
    
    def compute_all_no_ref(self, audio_files: List[str], selected_metrics: Optional[List[str]] = None) -> Dict:
        """
        并行计算所有无参考指标（NISQA / DNSMOS / Scoreq / UTMOS 四路并行）

        优化策略:
        1. NISQA/DNSMOS/Scoreq/UTMOS全部同时并行
        2. NISQA内部批处理已由自身优化
        3. DNSMOS内部批处理（段级batch）已由自身优化
        """
        file_num = len(audio_files)

        if selected_metrics is None:
            selected_metrics = ['dnsmos', 'nisqa', 'scoreq', 'utmos']

        metric_defaults = {
            'nisqa': lambda: {'mos_pred': [0.0]*file_num, 'noi_pred': [0.0]*file_num,
                              'dis_pred': [0.0]*file_num, 'col_pred': [0.0]*file_num,
                              'loud_pred': [0.0]*file_num},
            'dnsmos': lambda: {'OVRL': [0.0]*file_num, 'SIG': [0.0]*file_num,
                               'BAK': [0.0]*file_num, 'P808_MOS': [0.0]*file_num},
            'scoreq': lambda: {'scoreq': [0.0]*file_num},
            'utmos':  lambda: {'utmos': [0.0]*file_num},
        }

        # 定义各个指标的计算闭包
        def compute_nisqa():
            if 'nisqa' not in selected_metrics or 'nisqa' not in self.models:
                return metric_defaults['nisqa']()
            try:
                return self.models['nisqa'].get_mos(audio_files)
            except Exception as e:
                print(f"NISQA计算失败: {e}")
                return metric_defaults['nisqa']()

        def compute_dnsmos():
            if 'dnsmos' not in selected_metrics or 'dnsmos' not in self.models:
                return metric_defaults['dnsmos']()
            try:
                return self.models['dnsmos'].get_mos(audio_files)
            except Exception as e:
                print(f"DNSMOS计算失败: {e}")
                return metric_defaults['dnsmos']()

        def compute_scoreq():
            if 'scoreq' not in selected_metrics or 'scoreq' not in self.models:
                return metric_defaults['scoreq']()
            try:
                return self.models['scoreq'].get_mos(audio_files)
            except Exception as e:
                print(f"Scoreq计算失败: {e}")
                return metric_defaults['scoreq']()

        def compute_utmos():
            if 'utmos' not in selected_metrics or 'utmos' not in self.models:
                return metric_defaults['utmos']()
            try:
                return self.models['utmos'].predict_files(audio_files)
            except Exception as e:
                print(f"UTMOS计算失败: {e}")
                return metric_defaults['utmos']()

        # 四路并行提交到共享线程池
        logger.info(f"[并行无参考] 开始并行计算: {selected_metrics}")
        futures = {}
        if 'nisqa' in selected_metrics and 'nisqa' in self.models:
            futures['nisqa'] = self.executor.submit(compute_nisqa)
        if 'dnsmos' in selected_metrics and 'dnsmos' in self.models:
            futures['dnsmos'] = self.executor.submit(compute_dnsmos)
        if 'scoreq' in selected_metrics and 'scoreq' in self.models:
            futures['scoreq'] = self.executor.submit(compute_scoreq)
        if 'utmos' in selected_metrics and 'utmos' in self.models:
            futures['utmos'] = self.executor.submit(compute_utmos)

        # 收集结果
        results = {}
        for name, future in futures.items():
            try:
                results.update(future.result())
            except Exception as e:
                logger.error(f"{name}计算异常: {e}")
                results.update(metric_defaults[name]())

        # 确保未提交的指标有默认值
        for m in selected_metrics:
            defaults = metric_defaults[m]()
            for k, v in defaults.items():
                if k not in results:
                    results[k] = v

        return results
    
    def compute_all_with_ref(self, audio_files: List[str], ref_dir: str, selected_metrics: Optional[List[str]] = None) -> Dict:
        """
        并行计算所有有参考指标（RefScore / WER / TCF 三路并行）
        """
        results = {}
        file_num = len(audio_files)
        defaults = {}

        # 如果没有指定计算项目，使用默认全部
        if selected_metrics is None:
            selected_metrics = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf']

        metrics = selected_metrics.copy()
        logger.info(f"[compute_all_with_ref] 并行计算有参考指标: {metrics}")

        need_ref = any(m in metrics for m in ['pesq', 'stoi', 'sisdr'])
        need_wer = 'wer' in metrics
        need_tcf = 'tcf' in metrics

        # 任务闭包
        def _compute_ref():
            if not need_ref or 'ref_score' not in self.models:
                return {}
            try:
                ref_scores = self.models['ref_score'].get_mos(audio_files, ref_dir)
                if 'pesq' not in metrics and 'pesq' in ref_scores:
                    ref_scores.pop('pesq', None)
                if 'stoi' not in metrics and 'STOI' in ref_scores:
                    ref_scores.pop('STOI', None)
                if 'sisdr' not in metrics and 'SISDR' in ref_scores:
                    ref_scores.pop('SISDR', None)
                return ref_scores
            except Exception as e:
                logger.error(f"RefScore计算失败: {e}")
                return {}

        def _compute_wer():
            if not need_wer or 'wer' not in self.models:
                return {}
            try:
                return self.models['wer'].get_wer(audio_files, ref_dir=ref_dir)
            except Exception as e:
                logger.error(f"WER计算失败: {e}")
                return {}

        def _compute_tcf():
            if not need_tcf or 'tcf' not in self.models:
                return {}
            try:
                return self.models['tcf'].get_mos(audio_files, ref_dir)
            except Exception as e:
                logger.error(f"TCF计算失败: {e}")
                return {}

        # 三路并行提交到共享线程池
        ref_future = self.executor.submit(_compute_ref)
        wer_future = self.executor.submit(_compute_wer) if need_wer else None
        tcf_future = self.executor.submit(_compute_tcf) if need_tcf else None

        # 收集结果（ref总在最前面，但实际计算是并行的）
        ref_result = ref_future.result()
        results.update(ref_result)
        if need_ref and 'STOI' not in results:
            defaults.update({'STOI': [0.0]*file_num, 'SISDR': [0.0]*file_num, 'pesq': [0.0]*file_num})

        if wer_future:
            wer_result = wer_future.result()
            results.update(wer_result)
            if 'wer' not in results:
                defaults.update({'wer': [0.0]*file_num, 'wcorr': [0.0]*file_num})

        if tcf_future:
            tcf_result = tcf_future.result()
            results.update(tcf_result)
            if 'tcf' not in results:
                defaults.update({'tcf': [0.0]*file_num})

        results.update({k: v for k, v in defaults.items() if k not in results})

        print(f"[compute_all_with_ref] 最终结果(填充前): {results}")
        
        # 确保有默认值 - 只填充缺失的键，不覆盖已有值
        if any(m in metrics for m in ['pesq', 'stoi', 'sisdr']):
            if 'STOI' not in results:
                results['STOI'] = [0.0]*file_num
            if 'SISDR' not in results:
                results['SISDR'] = [0.0]*file_num
            if 'pesq' not in results:
                results['pesq'] = [0.0]*file_num
        if 'wer' in metrics:
            if 'wer' not in results:
                results['wer'] = [0.0]*file_num
            if 'wcorr' not in results:
                results['wcorr'] = [0.0]*file_num
        if 'tcf' in metrics and 'tcf' not in results:
            results['tcf'] = [0.0]*file_num

        print(f"[compute_all_with_ref] 返回结果: {results}")
        return results
    
    def compute_final_scores(self, results: Dict, audio_files: List[str], has_reference: bool, selected_metrics: Optional[List[str]] = None) -> List[float]:
        """计算最终得分 - 使用字典键名访问，确保正确的指标映射，支持动态计算项目

        关键规则：
        - 有参考指标（pesq/stoi/sisdr/wer/tcf）中，如果某个指标值为0且未找到参考，则不参与final score计算
        - 无参考指标始终参与计算
        """
        file_num = len(audio_files)
        final_scores = []

        # 如果没有指定计算项目，使用默认全部
        if selected_metrics is None:
            selected_metrics = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf', 'dnsmos', 'nisqa', 'scoreq', 'utmos']

        # 安全的获取指标值
        def get_value(key, idx, default=0.0):
            if key in results and idx < len(results[key]):
                val = results[key][idx]
                return float(val) if isinstance(val, (int, float)) else default
            return default

        for i in range(file_num):
            scores = []

            # NISQA指标 (0-5)
            if 'nisqa' in selected_metrics:
                mos_pred = get_value('mos_pred', i)
                noi_pred = get_value('noi_pred', i)
                dis_pred = get_value('dis_pred', i)
                col_pred = get_value('col_pred', i)
                loud_pred = get_value('loud_pred', i)
                scores.extend([mos_pred, noi_pred, dis_pred, col_pred, loud_pred])

            # DNSMOS指标 (0-5)
            if 'dnsmos' in selected_metrics:
                ovrl = get_value('OVRL', i)
                sig = get_value('SIG', i)
                bak = get_value('BAK', i)
                p808 = get_value('P808_MOS', i)
                scores.extend([ovrl, sig, bak, p808])

            # ScoreQ (0-5)
            if 'scoreq' in selected_metrics:
                scoreq_val = get_value('scoreq', i)
                scores.append(scoreq_val)

            # UTMOS (0-5)
            if 'utmos' in selected_metrics:
                utmos_val = get_value('utmos', i)
                scores.append(utmos_val)

            # 有参考指标 — 只有has_reference=True且值非0时参与计算
            if has_reference:
                # STOI: -1~1 映射到 0-5（值>0才参与）
                if 'stoi' in selected_metrics:
                    stoi = get_value('STOI', i)
                    if stoi != 0.0:
                        scores.append((stoi + 1) * 2.5)

                # SISDR: 使用sigmoid归一化到0-5（值>0才参与）
                if 'sisdr' in selected_metrics:
                    sisdr = get_value('SISDR', i)
                    if sisdr != 0.0:
                        scores.append((1 / (1 + np.exp(-sisdr/10))) * 5)

                # PESQ: 0-4.5 映射到 0-5（值>0才参与）
                if 'pesq' in selected_metrics:
                    pesq = get_value('pesq', i)
                    if pesq != 0.0:
                        scores.append(pesq * (5/4.5))

                # WER: 0-1 映射到 0-5 (越低越好)（值>0才参与）
                if 'wer' in selected_metrics:
                    wer = get_value('wer', i)
                    if wer != 0.0:
                        scores.append((1 - wer) * 5)

                    # WCORR: 0-1 映射到 0-5
                    wcorr = get_value('wcorr', i)
                    if wcorr != 0.0:
                        scores.append(wcorr * 5)

                # TCF: 0-1 映射到 0-5（值>0才参与）
                if 'tcf' in selected_metrics:
                    tcf = get_value('tcf', i)
                    if tcf != 0.0:
                        scores.append(tcf * 5)

            # 计算最终得分
            if scores:
                tmp = np.mean(scores)
            else:
                tmp = 0.0

            # 打印调试信息
            logger.debug(f"FinalScore计算 - 文件{i}: has_ref={has_reference}, 选择项目={selected_metrics}")
            logger.debug(f"  所有有效分数({len(scores)}个): {[f'{s:.2f}' for s in scores]}")
            logger.debug(f"  最终得分: {tmp:.2f}")

            final_scores.append(tmp)

        return final_scores
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        return perf_timer.get_report()
    
    def reset_performance(self):
        """重置性能统计"""
        perf_timer.reset()


# 全局并行计算实例
parallel_compute = ParallelMOSCompute(max_workers=4)


def compute_mos_scores_optimized(
    audio_files: List[str],
    ref_dir: str,
    has_reference: bool = True,
    selected_metrics: Optional[List[str]] = None
) -> Dict:
    """
    优化版MOS分数计算入口

    Args:
        audio_files: 音频文件路径列表
        ref_dir: 参考音频目录
        has_reference: 是否有参考音频
        selected_metrics: 用户选择的计算项目列表

    Returns:
        评分结果字典
    """
    # 确保模型已初始化
    print(f"[compute_mos_scores_optimized] 检查模型状态，当前models: {list(parallel_compute.models.keys())}")
    if not parallel_compute.models:
        print("[compute_mos_scores_optimized] 模型未初始化，调用init_models...")
        parallel_compute.init_models()
        print(f"[compute_mos_scores_optimized] init_models完成，当前models: {list(parallel_compute.models.keys())}")
    else:
        print("[compute_mos_scores_optimized] 模型已初始化，跳过init_models")

    # 如果没有指定计算项目，使用默认全部
    if selected_metrics is None:
        selected_metrics = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf', 'dnsmos', 'nisqa', 'scoreq', 'utmos']

    # 预热音频缓存：将所有测试音频+参考音频一次性加载到缓存
    logger.info(f"[音频缓存] 预热缓存：{len(audio_files)}个测试文件")
    all_audio_paths = list(audio_files)
    if has_reference and os.path.isdir(ref_dir):
        ref_audio_files = [
            os.path.join(ref_dir, f) for f in os.listdir(ref_dir)
            if f.endswith(('.wav', '.mp3', '.flac'))
        ]
        all_audio_paths.extend(ref_audio_files)
    preload_audio_cache(all_audio_paths, sr=16000)

    results = {}
    file_num = len(audio_files)

    # 计算无参考指标(并行)
    no_ref_metrics = [m for m in selected_metrics if m in ['dnsmos', 'nisqa', 'scoreq', 'utmos']]
    if no_ref_metrics:
        no_ref_results = parallel_compute.compute_all_no_ref(audio_files, no_ref_metrics)
        results.update(no_ref_results)
    else:
        # 填充0值
        results.update({
            'OVRL': [0.0]*file_num, 'SIG': [0.0]*file_num, 'BAK': [0.0]*file_num, 'P808_MOS': [0.0]*file_num,
            'mos_pred': [0.0]*file_num, 'noi_pred': [0.0]*file_num, 'dis_pred': [0.0]*file_num,
            'col_pred': [0.0]*file_num, 'loud_pred': [0.0]*file_num,
            'scoreq': [0.0]*file_num, 'utmos': [0.0]*file_num
        })

    # 计算有参考指标(并行)
    print(f"[compute_mos_scores_optimized] has_reference={has_reference}, selected_metrics={selected_metrics}")
    if has_reference:
        ref_metrics = [m for m in selected_metrics if m in ['pesq', 'stoi', 'sisdr', 'wer', 'tcf']]
        print(f"[compute_mos_scores_optimized] ref_metrics={ref_metrics}")
        if ref_metrics:
            logger.info(f"[MOS计算] 开始计算有参考指标: {ref_metrics}")
            ref_results = parallel_compute.compute_all_with_ref(audio_files, ref_dir, ref_metrics)
            logger.info(f"[MOS计算] 有参考指标计算完成，共 {len(ref_results)} 个指标")
            results.update(ref_results)
        else:
            print(f"[compute_mos_scores_optimized] 无参考指标需要计算，填充0值")
            # 填充0值
            results.update({
                'STOI': [0.0]*file_num, 'SISDR': [0.0]*file_num, 'pesq': [0.0]*file_num,
                'wer': [0.0]*file_num, 'wcorr': [0.0]*file_num, 'tcf': [0.0]*file_num
            })
    else:
        print(f"[compute_mos_scores_optimized] 无参考音频，填充0值")
        # 填充0值
        results.update({
            'STOI': [0.0]*file_num, 'SISDR': [0.0]*file_num, 'pesq': [0.0]*file_num,
            'wer': [0.0]*file_num, 'wcorr': [0.0]*file_num, 'tcf': [0.0]*file_num
        })

    # 计算最终得分
    final_scores = parallel_compute.compute_final_scores(results, audio_files, has_reference, selected_metrics)
    results['final_scores'] = final_scores

    return results


def get_performance_report() -> Dict:
    """获取性能报告"""
    return parallel_compute.get_performance_report()


def reset_performance_tracking():
    """重置性能跟踪"""
    parallel_compute.reset_performance()
