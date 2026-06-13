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
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# 添加本地包路径 - 从 algorithms 目录导入
# 当前文件位置: app/core/calculator/mos_calculator.py
# 需要到达: app/algorithms/
_ALGORITHMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'algorithms')
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

# 设置CUDA环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 启用cuDNN - 已安装cuDNN 9.8.0，与PyTorch 2.8.0+cu128兼容
torch.backends.cudnn.enabled = True

# 初始化CUDA上下文
try:
    if torch.cuda.is_available():
        # 不手动调用torch.cuda.init()，让PyTorch自动初始化
        # 简单预热
        _ = torch.zeros(1).cuda()
        cudnn_version = torch.backends.cudnn.version()
        cudnn_status = "启用" if torch.backends.cudnn.enabled else "禁用"
        print(f"✓ CUDA初始化完成: {torch.cuda.get_device_name(0)} (cuDNN{cudnn_status}, 版本{cudnn_version})")
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
    使用音频内容匹配（指纹+DTW）查找对应的参考文件。
    这是增强版的 get_ref_file，支持任意自定义参考音频。

    匹配策略（按优先级）：
    1. 基于文件名的传统匹配（快速，兼容旧模式）
    2. 基于音频指纹的内容匹配（支持任意参考音频）

    Args:
        input_wav_file: 测试音频文件路径
        ref_dir: 参考音频目录

    Returns:
        (ref_file_path, match_info) 或 (None, None)
        match_info: {"ref_id": str, "confidence": float, "offset": float, ...}
    """
    import json

    # 策略1: 传统文件名匹配（快速路径）
    input_basename = os.path.basename(input_wav_file)
    input_name = input_basename.removesuffix('.wav')

    # 加载参考音频元数据
    metadata_file = os.path.join(ref_dir, ".metadata.json")
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception:
            pass

    audios = metadata.get("audios", {})

    # 方式1: 直接匹配相同文件名
    ref_file = os.path.join(ref_dir, input_basename)
    if os.path.exists(ref_file):
        logger.info(f"[参考匹配-文件名] 直接匹配: {input_basename} -> {input_basename}")
        return ref_file, {"method": "filename_exact", "ref_name": input_basename}

    # 方式2: 匹配 ref_前缀 + 完整文件名
    ref_file_name = "ref_" + input_basename
    ref_file = os.path.join(ref_dir, ref_file_name)
    if os.path.exists(ref_file):
        logger.info(f"[参考匹配-文件名] ref_前缀匹配: {input_basename} -> {ref_file_name}")
        return ref_file, {"method": "filename_ref_prefix", "ref_name": ref_file_name}

    # 方式3: 从文件名提取ID匹配 (如 test_001.wav -> ref_001.wav)
    parts = input_name.split('_')
    if len(parts) >= 2:
        file_id = parts[-1]
        ref_file_name = f"ref_{file_id}.wav"
        ref_file = os.path.join(ref_dir, ref_file_name)
        if os.path.exists(ref_file):
            logger.info(f"[参考匹配-文件名] ID匹配: {input_basename} -> {ref_file_name}")
            return ref_file, {"method": "filename_id", "ref_name": ref_file_name}

    # 方式4: 尝试匹配不带后缀的文件名 (如 test.wav -> ref_test.wav)
    ref_file_name = f"ref_{input_name}.wav"
    ref_file = os.path.join(ref_dir, ref_file_name)
    if os.path.exists(ref_file):
        logger.info(f"[参考匹配-文件名] 全名匹配: {input_basename} -> {ref_file_name}")
        return ref_file, {"method": "filename_full", "ref_name": ref_file_name}

    # 策略2: 音频内容指纹匹配（较慢但支持任意参考音频）
    logger.info(f"[参考匹配-内容] 文件名匹配均失败，尝试内容指纹匹配: {input_basename}")
    try:
        from reference_matcher import get_reference_matcher

        matcher = get_reference_matcher(ref_dir=ref_dir)
        # 确保数据库已建立
        if not matcher.database.entries:
            matcher.build_database(ref_dir)

        match_results = matcher.match_test_audio(input_wav_file, min_confidence=0.2)

        if match_results:
            best_match = match_results[0]
            ref_file = best_match.ref_file
            ref_name = best_match.ref_name
            logger.info(f"[参考匹配-内容] ✓ 指纹匹配成功: {input_basename} -> {ref_name} "
                         f"(confidence={best_match.confidence:.3f}, "
                         f"offset={best_match.offset_in_test:.2f}s)")
            return ref_file, {
                "method": "content_fingerprint",
                "ref_id": best_match.ref_id,
                "ref_name": ref_name,
                "confidence": best_match.confidence,
                "offset": best_match.offset_in_test,
                "hash_matches": best_match.hash_matches
            }
        else:
            logger.warning(f"[参考匹配-内容] ✗ 未找到匹配的参考音频: {input_basename}")
    except ImportError as e:
        logger.warning(f"[参考匹配-内容] 指纹匹配模块不可用: {e}")
    except Exception as e:
        logger.error(f"[参考匹配-内容] 匹配失败: {e}")

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
    
    def __get_score(self, fpath, is_personalized_MOS=None):
        aud, input_fs = sf.read(fpath)
        if len(aud.shape) > 1:
            aud = aud[:, 0]
        
        fs = self.SAMPLING_RATE
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        
        len_samples = int(self.INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
        hop_len_samples = fs
        
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
        
        return {
            'OVRL': np.mean(predicted_mos_ovr_seg),
            'SIG': np.mean(predicted_mos_sig_seg),
            'BAK': np.mean(predicted_mos_bak_seg),
            'P808_MOS': np.mean(predicted_p808_mos)
        }
    
    @timed_execution("dnsmos")
    def get_mos(self, file_list):
        """计算dnsmos - 使用批处理优化"""
        file_num = len(file_list)
        ovrl = [0.0 for _ in range(file_num)]
        sig = [0.0 for _ in range(file_num)]
        bak = [0.0 for _ in range(file_num)]
        p808mos = [0.0 for _ in range(file_num)]
        
        for file_index, clip in enumerate(file_list):
            try:
                data = self.__get_score(clip)
                ovrl[file_index] = float(data['OVRL'])
                sig[file_index] = float(data['SIG'])
                bak[file_index] = float(data['BAK'])
                p808mos[file_index] = float(data['P808_MOS'])
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
            num_workers=4  # 使用多线程数据加载
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
        """计算scoreq分 - 使用线程池并行"""
        score_list = [0.0 for _ in range(len(file_dir_list))]
        
        def process_single(file_path, idx):
            try:
                return idx, self.pred_mos_ins.predict(file_path)
            except Exception as e:
                print(f"Scoreq计算失败 {file_path}: {e}")
                return idx, 0.0
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=4) as executor:
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
        """计算stoi、sisdr、pesq分"""
        file_num = len(file_list)
        STOI = [0.0 for _ in range(file_num)]
        SISDR = [0.0 for _ in range(file_num)]
        PESQ = [0.0 for _ in range(file_num)]

        logger.info(f"\n[RefScore] 开始计算参考相关指标，文件数: {file_num}")
        logger.info(f"[RefScore] 参考音频目录: {ref_dir}")

        # 检查参考目录是否存在
        if not os.path.exists(ref_dir):
            logger.warning(f"[RefScore] ❌ 参考目录不存在: {ref_dir}")
            return {'STOI': STOI, 'SISDR': SISDR, 'pesq': PESQ}

        # 列出参考目录中的文件
        ref_files = [f for f in os.listdir(ref_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        logger.info(f"[RefScore] 参考目录中的音频文件: {ref_files}")

        matched_count = 0
        for file_index, file in enumerate(file_list):
            file_basename = os.path.basename(file)
            logger.info(f"\n[RefScore] 处理文件 {file_index+1}/{file_num}: {file_basename}")

            path_to_reference = self.get_ref_file(file, ref_dir)

            if path_to_reference is None:
                logger.warning(f"[RefScore] ⚠️ 未找到参考文件 for {file_basename}，"
                               f"参考指标将返回0")
                continue

            logger.info(f"[RefScore] ✓ 找到参考文件: {os.path.basename(path_to_reference)}")

            try:
                # 检查文件是否存在
                if not os.path.exists(file):
                    logger.error(f"[RefScore] ❌ 测试文件不存在: {file}")
                    continue
                if not os.path.exists(path_to_reference):
                    logger.error(f"[RefScore] ❌ 参考文件不存在: {path_to_reference}")
                    continue

                # stoi 和 sisdr
                scores = self.metrics(file, path_to_reference)
                stoi_val = float(scores['stoi'].mean())
                sisdr_val = float(scores['sisdr'].mean())
                STOI[file_index] = stoi_val
                SISDR[file_index] = sisdr_val
                logger.info(f"[RefScore] ✓ STOI={stoi_val:.4f}, SISDR={sisdr_val:.4f}")

                # pesq
                if PESQ_AVAILABLE:
                    ref, sr_ref = sf.read(path_to_reference)
                    est, sr_est = sf.read(file)

                    if sr_est != 16000:
                        est = librosa.resample(est, orig_sr=sr_est, target_sr=16000)
                    if sr_ref != 16000:
                        ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=16000)

                    # 电平平滑
                    rms_ref = np.sqrt(np.mean(ref**2))
                    rms_est = np.sqrt(np.mean(est**2))
                    if rms_est > 0:
                        gain = rms_ref / rms_est
                        est = est * gain

                    pesq_val = float(pesq.pesq(fs=16000, ref=ref, deg=est, mode='wb'))
                    PESQ[file_index] = pesq_val
                    logger.info(f"[RefScore] ✓ PESQ={pesq_val:.4f}")
                else:
                    logger.warning(f"[RefScore] ⚠️ PESQ不可用")

                matched_count += 1
            except Exception as e:
                logger.error(f"[RefScore] ❌ 计算失败 {file_basename}: {e}")
                import traceback
                logger.error(f"[RefScore] 错误详情: {traceback.format_exc()}")

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
        """计算wer - 支持动态ground truth文本"""
        file_num = len(file_dir_list)
        wer_data = [0.0 for _ in range(file_num)]
        wcorr = [0.0 for _ in range(file_num)]

        for file_index, file in enumerate(file_dir_list):
            try:
                result = self.model.transcribe(file)
                ref = self.__get_ref_gt_text(file, ref_dir=ref_dir)
                if ref is None:
                    logger.warning(f"[WER] 无ground truth文本 for {os.path.basename(file)}，WER=0")
                    continue

                if hasattr(result, 'text'):
                    text = result.text
                elif isinstance(result, dict):
                    text = result['text']
                else:
                    text = str(result)

                from wer import wer
                tmp_wer, tmp_wcorr = wer(ref, text)
                wer_data[file_index] = tmp_wer
                wcorr[file_index] = tmp_wcorr
            except Exception as e:
                logger.error(f"WER计算失败 {file}: {e}")

        return {'wer': wer_data, 'wcorr': wcorr}


class OptimizedToneColorFidelityScore:
    """优化的音色还原度评分 - 支持多模型加权评估"""

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

            try:
                if exists:
                    location = "项目路径" if is_project else "本地缓存"
                    logger.info(f"[TCF] 使用{location}模型 [{alg}]: {model_path}")
                    # 使用CUDA
                    self._pipeline_cache[alg] = pipeline(
                        task='speaker-verification',
                        model=model_path,
                        device='cuda' if cuda.is_available() else 'cpu'
                    )
                else:
                    logger.info(f"[TCF] 下载模型 [{alg}]: {model_config['model_id']}")
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    os.environ['CURL_CA_BUNDLE'] = ''
                    os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'

                    self._pipeline_cache[alg] = pipeline(
                        task='speaker-verification',
                        model=model_config["model_id"],
                        model_revision=model_config["revision"],
                        device='cuda' if cuda.is_available() else 'cpu'
                    )
            except Exception as e:
                logger.error(f"[TCF] 模型 [{alg}] 初始化失败: {e}")
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

        # 预先检查每个测试文件是否能找到参考文件
        no_ref_files = []
        for file_index in range(file_num):
            file = input_test_file_list[file_index]
            ref_file = self.get_ref_file(file, ref_dir)
            if ref_file is None or ref_file == file or ref_file not in file_embedding_score_dict:
                no_ref_files.append(os.path.basename(file))
        if no_ref_files:
            logger.warning(f"[TCF] 以下文件未找到匹配的参考音频，TCF评分将为0: {no_ref_files}")
            logger.warning(f"[TCF] 文件命名需匹配 ref_ 模式，或包含 _XXX 后缀")

        # 遍历所有算法模型
        available_algs = []
        failed_algs = []
        for alg in self.sv_model_dict.keys():
            try:
                logger.info(f"[TCF] 开始计算 [{alg}] 模型，共{len(test_file_list)}个音频...")
                sv_pipeline = self._get_pipeline(alg)
                result = sv_pipeline(test_file_list, output_emb=True)

                # 清理缓存，避免爆显存
                if alg in self._pipeline_cache:
                    del self._pipeline_cache[alg]
                if cuda.is_available():
                    cuda.empty_cache()
                    cuda.synchronize()

                # 存储embedding
                all_embs = result['embs']
                for i in range(len(all_embs)):
                    file_embedding_score_dict[test_file_list[i]][alg] = {
                        "embedding": all_embs[i]
                    }

                available_algs.append(alg)
                logger.info(f"[TCF] ✓ [{alg}] 模型计算完成")

            except Exception as e:
                failed_algs.append(alg)
                logger.warning(f"[TCF] ✗ [{alg}] 模型计算失败: {e}")
                continue

        if not available_algs:
            logger.error(f"[TCF] 所有TCF模型都不可用！失败的模型: {failed_algs}")
            logger.error(f"[TCF] TCF评分全部为0，请检查ModelScope和模型文件配置")
            return {"tcf": total_score_list}

        if failed_algs:
            logger.warning(f"[TCF] 部分模型失败: {failed_algs}, 可用模型: {available_algs}")

        logger.info(f"[TCF] 可用算法模型: {available_algs}")
        logger.info(f"[TCF] 各算法权重: {[(alg, self.sv_model_dict[alg]['weight']) for alg in available_algs]}")

        # 计算加权得分
        matched_count = 0
        for file_index in range(file_num):
            file = input_test_file_list[file_index]
            ref_file = self.get_ref_file(file, ref_dir)

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
    
    def init_models(self):
        """初始化所有模型"""
        print(f"[init_models] 开始初始化模型，当前models: {list(self.models.keys())}")
        
        # 无参考模型
        try:
            print("[init_models] 初始化DNSMOS...")
            self.models['dnsmos'] = OptimizedDNSMOScore()
            print("[init_models] DNSMOS初始化成功")
        except Exception as e:
            print(f"[init_models] DNSMOS初始化失败: {e}")
        
        try:
            print("[init_models] 初始化NISQA...")
            self.models['nisqa'] = OptimizedNisqaMosScore()
            print("[init_models] NISQA初始化成功")
        except Exception as e:
            print(f"[init_models] NISQA初始化失败: {e}")
        
        try:
            print("[init_models] 初始化Scoreq...")
            self.models['scoreq'] = OptimizedScoreqScore()
            print("[init_models] Scoreq初始化成功")
        except Exception as e:
            print(f"[init_models] Scoreq初始化失败: {e}")
        
        # 有参考模型
        try:
            print("[init_models] 初始化RefScore...")
            self.models['ref_score'] = OptimizedRefScore()
            print("[init_models] RefScore初始化成功")
        except Exception as e:
            print(f"[init_models] RefScore初始化失败: {e}")
        
        try:
            print("[init_models] 初始化WER...")
            self.models['wer'] = OptimizedWerScore()
            print("[init_models] WER初始化成功")
        except Exception as e:
            print(f"[init_models] WER初始化失败: {e}")
        
        try:
            logger.info("[init_models] 初始化TCF...")
            self.models['tcf'] = OptimizedToneColorFidelityScore()
            logger.info(f"[init_models] TCF初始化成功，当前models: {list(self.models.keys())}")
        except Exception as e:
            logger.error(f"[init_models] TCF初始化失败: {e}")
            import traceback
            logger.error(f"[init_models] TCF初始化错误详情: {traceback.format_exc()}")

        # UTMOS模型
        try:
            if UTMOS_AVAILABLE:
                self.models['utmos'] = UTMOSCore()
                logger.info("✓ UTMOS模型初始化成功")
        except Exception as e:
            logger.warning(f"UTMOS初始化失败: {e}")
    
    def compute_all_no_ref(self, audio_files: List[str], selected_metrics: Optional[List[str]] = None) -> Dict:
        """
        并行计算所有无参考指标

        优化策略:
        1. NISQA使用批处理模式
        2. DNSMOS和Scoreq使用线程池并行
        """
        results = {}
        file_num = len(audio_files)

        # 如果没有指定计算项目，使用默认全部
        if selected_metrics is None:
            selected_metrics = ['dnsmos', 'nisqa', 'scoreq', 'utmos']

        # NISQA使用批处理(已经内部优化)
        if 'nisqa' in selected_metrics:
            if 'nisqa' in self.models:
                try:
                    results.update(self.models['nisqa'].get_mos(audio_files))
                except Exception as e:
                    print(f"NISQA计算失败: {e}")
                    results.update({
                        'mos_pred': [0.0]*file_num, 'noi_pred': [0.0]*file_num,
                        'dis_pred': [0.0]*file_num, 'col_pred': [0.0]*file_num,
                        'loud_pred': [0.0]*file_num
                    })
            else:
                results.update({
                    'mos_pred': [0.0]*file_num, 'noi_pred': [0.0]*file_num,
                    'dis_pred': [0.0]*file_num, 'col_pred': [0.0]*file_num,
                    'loud_pred': [0.0]*file_num
                })
        else:
            results.update({
                'mos_pred': [0.0]*file_num, 'noi_pred': [0.0]*file_num,
                'dis_pred': [0.0]*file_num, 'col_pred': [0.0]*file_num,
                'loud_pred': [0.0]*file_num
            })

        # DNSMOS、Scoreq和UTMOS并行计算
        def compute_dnsmos():
            try:
                if 'dnsmos' in self.models:
                    return self.models['dnsmos'].get_mos(audio_files)
            except Exception as e:
                print(f"DNSMOS计算失败: {e}")
            return {'OVRL': [0.0]*file_num, 'SIG': [0.0]*file_num, 'BAK': [0.0]*file_num, 'P808_MOS': [0.0]*file_num}

        def compute_scoreq():
            try:
                if 'scoreq' in self.models:
                    return self.models['scoreq'].get_mos(audio_files)
            except Exception as e:
                print(f"Scoreq计算失败: {e}")
            return {'scoreq': [0.0]*file_num}

        def compute_utmos():
            print(f"[DEBUG UTMOS] 开始计算UTMOS, models中是否有utmos: {'utmos' in self.models}")
            try:
                if 'utmos' in self.models:
                    print(f"[DEBUG UTMOS] 调用predict_files, 文件数: {len(audio_files)}")
                    result = self.models['utmos'].predict_files(audio_files)
                    print(f"[DEBUG UTMOS] predict_files返回: {result}")
                    return result
                else:
                    print(f"[DEBUG UTMOS] models中没有utmos, 可用models: {list(self.models.keys())}")
            except Exception as e:
                print(f"[DEBUG UTMOS] UTMOS计算失败: {e}")
                import traceback
                traceback.print_exc()
            return {'utmos': [0.0]*file_num}

        # 使用线程池并行执行
        print(f"[DEBUG] 开始并行计算无参考指标: {selected_metrics}")
        print(f"[DEBUG] 可用models: {list(self.models.keys())}")
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                if 'dnsmos' in selected_metrics:
                    futures.append(('dnsmos', executor.submit(compute_dnsmos)))
                if 'scoreq' in selected_metrics:
                    futures.append(('scoreq', executor.submit(compute_scoreq)))
                if 'utmos' in selected_metrics:
                    print(f"[DEBUG] 提交UTMOS计算任务")
                    futures.append(('utmos', executor.submit(compute_utmos)))

                for name, future in futures:
                    print(f"[DEBUG] 获取{name}结果...")
                    result = future.result()
                    print(f"[DEBUG] {name}结果: {result}")
                    results.update(result)
        except Exception as e:
            print(f"[DEBUG] 并行计算无参考指标失败: {e}")
            import traceback
            traceback.print_exc()

        # 确保有默认值 - 只填充缺失的键，不覆盖已有值
        if 'dnsmos' in selected_metrics:
            if 'OVRL' not in results:
                results['OVRL'] = [0.0]*file_num
            if 'SIG' not in results:
                results['SIG'] = [0.0]*file_num
            if 'BAK' not in results:
                results['BAK'] = [0.0]*file_num
            if 'P808_MOS' not in results:
                results['P808_MOS'] = [0.0]*file_num
        if 'scoreq' in selected_metrics and 'scoreq' not in results:
            results['scoreq'] = [0.0]*file_num
        if 'utmos' in selected_metrics and 'utmos' not in results:
            results['utmos'] = [0.0]*file_num

        return results
    
    def compute_all_with_ref(self, audio_files: List[str], ref_dir: str, selected_metrics: Optional[List[str]] = None) -> Dict:
        """
        并行计算所有有参考指标
        """
        results = {}
        file_num = len(audio_files)

        # 如果没有指定计算项目，使用默认全部
        if selected_metrics is None:
            selected_metrics = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf']
        
        # 保存到局部变量，避免闭包问题
        metrics = selected_metrics.copy()
        print(f"[compute_all_with_ref] 开始计算有参考指标: {metrics}")

        # 直接调用，不使用内部线程池（避免线程池嵌套问题）
        if any(m in metrics for m in ['pesq', 'stoi', 'sisdr']):
            print(f"[compute_all_with_ref] 计算ref指标...")
            try:
                if 'ref_score' in self.models:
                    ref_scores = self.models['ref_score'].get_mos(audio_files, ref_dir)
                    print(f"[compute_all_with_ref] ref_score返回: {ref_scores}")
                    # 只保留用户选择的指标
                    if 'pesq' not in metrics:
                        ref_scores.pop('pesq', None)
                    if 'stoi' not in metrics:
                        ref_scores.pop('STOI', None)
                    if 'sisdr' not in metrics:
                        ref_scores.pop('SISDR', None)
                    print(f"[compute_all_with_ref] 过滤后: {ref_scores}")
                    results.update(ref_scores)
                else:
                    print(f"[compute_all_with_ref] ref_score模型未加载")
                    results.update({'STOI': [0.0]*file_num, 'SISDR': [0.0]*file_num, 'pesq': [0.0]*file_num})
            except Exception as e:
                print(f"[compute_all_with_ref] RefScore计算失败: {e}")
                import traceback
                print(f"[compute_all_with_ref] 错误详情: {traceback.format_exc()}")
                results.update({'STOI': [0.0]*file_num, 'SISDR': [0.0]*file_num, 'pesq': [0.0]*file_num})
        
        if 'wer' in metrics:
            logger.info("[compute_all_with_ref] 计算WER...")
            try:
                if 'wer' in self.models:
                    wer_scores = self.models['wer'].get_wer(audio_files, ref_dir=ref_dir)
                    logger.info(f"[compute_all_with_ref] WER返回: non-zero wer={sum(1 for w in wer_scores['wer'] if w != 0)}")
                    results.update(wer_scores)
                else:
                    logger.warning("[compute_all_with_ref] WER模型未加载")
                    results.update({'wer': [0.0]*file_num, 'wcorr': [0.0]*file_num})
            except Exception as e:
                logger.error(f"[compute_all_with_ref] WER计算失败: {e}")
                results.update({'wer': [0.0]*file_num, 'wcorr': [0.0]*file_num})
        
        if 'tcf' in metrics:
            logger.info("[compute_all_with_ref] 计算TCF...")
            logger.info(f"[compute_all_with_ref] self.models中是否有tcf: {'tcf' in self.models}")
            try:
                if 'tcf' in self.models:
                    logger.info("[compute_all_with_ref] 调用TCF get_mos...")
                    tcf_scores = self.models['tcf'].get_mos(audio_files, ref_dir)
                    logger.info(f"[compute_all_with_ref] TCF返回: {tcf_scores}")
                    results.update(tcf_scores)
                else:
                    logger.warning("[compute_all_with_ref] TCF模型未加载，TCF评分将为0")
                    results.update({'tcf': [0.0]*file_num})
            except Exception as e:
                logger.error(f"[compute_all_with_ref] TCF计算失败: {e}")
                import traceback
                logger.error(f"[compute_all_with_ref] TCF错误详情: {traceback.format_exc()}")
                results.update({'tcf': [0.0]*file_num})

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
            print(f"[compute_mos_scores_optimized] 调用compute_all_with_ref...")
            ref_results = parallel_compute.compute_all_with_ref(audio_files, ref_dir, ref_metrics)
            print(f"[compute_mos_scores_optimized] ref_results类型={type(ref_results)}, id={id(ref_results)}")
            print(f"[compute_mos_scores_optimized] ref_results={ref_results}")
            print(f"[compute_mos_scores_optimized] 更新前的results={results}")
            results.update(ref_results)
            print(f"[compute_mos_scores_optimized] 更新后的results={results}")
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
