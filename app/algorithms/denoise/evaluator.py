"""
降噪算法测评模块
提供全面的降噪效果评估功能
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import json

from .base import DenoiseResult
from .registry import DenoiserRegistry


@dataclass
class DenoiseMetrics:
    """降噪测评指标"""
    # 有参考指标
    pesq: Optional[float] = None  # PESQ得分 (1-4.5)
    stoi: Optional[float] = None  # STOI得分 (0-1)
    sisdr: Optional[float] = None  # SI-SDR得分 (dB)
    
    # 无参考指标
    dnsmos_ovrl: Optional[float] = None  # DNSMOS总体质量
    dnsmos_sig: Optional[float] = None   # DNSMOS信号质量
    dnsmos_bak: Optional[float] = None   # DNSMOS背景质量
    nisqa_mos: Optional[float] = None    # NISQA MOS得分
    utmos: Optional[float] = None        # UTMOS得分
    
    # 计算指标
    processing_time: float = 0.0  # 处理时间(秒)
    rtf: Optional[float] = None   # 实时因子 (处理时间/音频时长)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'pesq': self.pesq,
            'stoi': self.stoi,
            'sisdr': self.sisdr,
            'dnsmos_ovrl': self.dnsmos_ovrl,
            'dnsmos_sig': self.dnsmos_sig,
            'dnsmos_bak': self.dnsmos_bak,
            'nisqa_mos': self.nisqa_mos,
            'utmos': self.utmos,
            'processing_time': self.processing_time,
            'rtf': self.rtf
        }


@dataclass
class DenoiseEvaluation:
    """单条降噪测评结果"""
    file_name: str
    algorithm_name: str
    noisy_audio_path: str
    denoised_audio_path: str
    reference_audio_path: Optional[str] = None
    metrics: DenoiseMetrics = field(default_factory=DenoiseMetrics)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'file_name': self.file_name,
            'algorithm_name': self.algorithm_name,
            'noisy_audio_path': self.noisy_audio_path,
            'denoised_audio_path': self.denoised_audio_path,
            'reference_audio_path': self.reference_audio_path,
            'metrics': self.metrics.to_dict()
        }


class _DNSMOSOnnxWrapper:
    """
    DNSMOS ONNX Runtime 包装器
    直接使用ONNX模型进行DNSMOS评分，避免modelscope依赖
    参考: Microsoft DNS-Challenge DNSMOS实现
    """

    def __init__(self, primary_model_path: str, p808_model_path: str):
        import onnxruntime as ort

        # 尝试CUDA，失败则回退CPU
        try:
            self.onnx_sess = ort.InferenceSession(primary_model_path,
                                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception:
            self.onnx_sess = ort.InferenceSession(primary_model_path, providers=['CPUExecutionProvider'])

        try:
            self.p808_onnx_sess = ort.InferenceSession(p808_model_path,
                                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception:
            self.p808_onnx_sess = ort.InferenceSession(p808_model_path, providers=['CPUExecutionProvider'])

        self.INPUT_LENGTH = 9.01
        self.SAMPLING_RATE = 16000

    @staticmethod
    def _audio_melspec(audio: np.ndarray, n_mels: int = 120, frame_size: int = 320,
                        hop_length: int = 160, sr: int = 16000, to_db: bool = True) -> np.ndarray:
        """计算梅尔频谱"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1,
                                                   hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    @staticmethod
    def _polyfit(sig: float, bak: float, ovr: float) -> tuple:
        """多项式拟合校正"""
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def compute_mos(self, audio: np.ndarray, sample_rate: int) -> tuple:
        """
        计算DNSMOS分数

        Args:
            audio: 音频numpy数组 (float32, [-1, 1])
            sample_rate: 采样率

        Returns:
            (OVRL, SIG, BAK) tuple
        """
        import onnxruntime as ort

        fs = self.SAMPLING_RATE
        if sample_rate != fs:
            audio = librosa.resample(audio.astype(np.float64), orig_sr=sample_rate, target_sr=fs)

        len_samples = int(self.INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_ovr_seg = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + self.INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(
                self._audio_melspec(audio=audio_seg[:-160])
            ).astype('float32')[np.newaxis, :, :]

            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}

            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self._polyfit(mos_sig_raw, mos_bak_raw, mos_ovr_raw)

            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        if not predicted_mos_ovr_seg:
            return (None, None, None)

        return (
            float(np.mean(predicted_mos_ovr_seg)),
            float(np.mean(predicted_mos_sig_seg)),
            float(np.mean(predicted_mos_bak_seg)),
        )


class DenoiseEvaluator:
    """
    降噪算法测评器
    提供全面的降噪效果评估
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cuda"):
        """
        初始化测评器
        
        Args:
            sample_rate: 采样率
            device: 计算设备
        """
        self.sample_rate = sample_rate
        self.device = device
        self._models = {}
        self._init_models()
    
    def _init_models(self):
        """初始化评估模型"""
        import sys
        import os
        from pathlib import Path

        project_root = str(Path(__file__).parent.parent.parent.parent)
        # 确保 app/algorithms/ 在 sys.path 最前面，避免 pip nisqa 包冲突
        algorithms_path = os.path.join(project_root, 'app', 'algorithms')
        if algorithms_path not in sys.path:
            sys.path.insert(0, algorithms_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # 初始化DNSMOS (使用ONNX Runtime，直接加载ONNX模型，避免modelscope依赖链)
        self._init_dnsmos_onnx(project_root)

        # 初始化NISQA (使用项目自带的nisqa模块 + 本地权重)
        self._init_nisqa(project_root)

        # 初始化UTMOS
    def _init_nisqa(self, project_root: str):
        """初始化NISQA模型（使用项目本地权重）"""
        import os
        try:
            # 项目自带nisqa模块在 app/algorithms/nisqa/
            # 权重在 app/algorithms/nisqa/weights/
            weights_dir = os.path.join(project_root, 'app', 'algorithms', 'nisqa', 'weights')
            default_model = 'nisqa_3000.tar'
            model_path = os.path.join(weights_dir, default_model)

            if not os.path.exists(model_path):
                print(f"NISQA权重文件不存在: {model_path}")
                return

            from nisqa.predict import nisqa_predict
            self._models['nisqa'] = {
                'predict_fn': nisqa_predict,
                'model': default_model,
            }
            print(f"NISQA评估模型初始化成功 (权重: {default_model})")
        except Exception as e:
            print(f"NISQA初始化失败（将跳过该指标）: {e}")

        try:
            from utmos.utmos_score import UTMOSCore
            self._models['utmos'] = UTMOSCore()
            print("UTMOS评估模型初始化成功")
        except Exception as e:
            print(f"UTMOS初始化失败（将跳过该指标）: {e}")

    def _init_dnsmos_onnx(self, project_root: str):
        """使用ONNX Runtime初始化DNSMOS模型（避免导入modelscope）"""
        import os
        try:
            import onnxruntime as ort

            primary_path = os.path.join(project_root, 'app', 'algorithms', 'dnsmos', 'pDNSMOS', 'sig_bak_ovr.onnx')
            p808_path = os.path.join(project_root, 'app', 'algorithms', 'dnsmos', 'DNSMOS', 'model_v8.onnx')

            # 检查备用路径
            if not os.path.exists(primary_path):
                primary_path = os.path.join(project_root, 'models', 'dnsmos', 'pDNSMOS', 'sig_bak_ovr.onnx')
                p808_path = os.path.join(project_root, 'models', 'dnsmos', 'DNSMOS', 'model_v8.onnx')

            if not os.path.exists(primary_path):
                print(f"DNSMOS ONNX模型不存在: {primary_path}")
                return

            self._models['dnsmos'] = _DNSMOSOnnxWrapper(primary_path, p808_path)
            print("DNSMOS评估模型初始化成功 (ONNX Runtime)")
        except Exception as e:
            print(f"DNSMOS初始化失败（将跳过该指标）: {e}")
    
    def evaluate_with_reference(
        self,
        denoised_audio: np.ndarray,
        reference_audio: np.ndarray,
        processing_time: float = 0.0
    ) -> DenoiseMetrics:
        """
        有参考音频的评估
        
        Args:
            denoised_audio: 降噪后音频
            reference_audio: 参考音频(干净语音)
            processing_time: 处理时间
            
        Returns:
            DenoiseMetrics对象
        """
        metrics = DenoiseMetrics(processing_time=processing_time)
        
        # 确保长度一致
        min_len = min(len(denoised_audio), len(reference_audio))
        denoised_audio = denoised_audio[:min_len]
        reference_audio = reference_audio[:min_len]
        
        # 计算PESQ
        try:
            from pesq import pesq
            metrics.pesq = pesq(self.sample_rate, reference_audio, denoised_audio, 'wb')
        except Exception as e:
            print(f"PESQ计算失败: {e}")
        
        # 计算STOI
        try:
            from pystoi import stoi
            metrics.stoi = stoi(reference_audio, denoised_audio, self.sample_rate, extended=False)
        except Exception as e:
            print(f"STOI计算失败: {e}")
        
        # 计算SI-SDR
        try:
            metrics.sisdr = self._compute_si_sdr(reference_audio, denoised_audio)
        except Exception as e:
            print(f"SI-SDR计算失败: {e}")
        
        # 计算RTF
        audio_duration = len(denoised_audio) / self.sample_rate
        if audio_duration > 0:
            metrics.rtf = processing_time / audio_duration
        
        return metrics
    
    def evaluate_without_reference(
        self,
        denoised_audio: np.ndarray,
        processing_time: float = 0.0
    ) -> DenoiseMetrics:
        """
        无参考音频的评估
        
        Args:
            denoised_audio: 降噪后音频
            processing_time: 处理时间
            
        Returns:
            DenoiseMetrics对象
        """
        metrics = DenoiseMetrics(processing_time=processing_time)
        
        # DNSMOS
        if 'dnsmos' in self._models:
            try:
                ovrl, sig, bak = self._models['dnsmos'].compute_mos(denoised_audio, self.sample_rate)
                metrics.dnsmos_ovrl = float(ovrl) if ovrl is not None else None
                metrics.dnsmos_sig = float(sig) if sig is not None else None
                metrics.dnsmos_bak = float(bak) if bak is not None else None
            except Exception as e:
                print(f"DNSMOS计算失败: {e}")

        # NISQA (使用项目本地权重)
        if 'nisqa' in self._models:
            try:
                import tempfile
                import os as _os
                nisqa_info = self._models['nisqa']
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as _tmp:
                    sf.write(_tmp.name, denoised_audio, self.sample_rate)
                    _tmp_path = _tmp.name
                try:
                    result_df = nisqa_info['predict_fn'](
                        mode='predict_file',
                        deg=_tmp_path,
                        model=nisqa_info['model'],
                    )
                    metrics.nisqa_mos = float(result_df['mos_pred'].values[0])
                finally:
                    try:
                        _os.unlink(_tmp_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"NISQA计算失败: {e}")
        
        # UTMOS
        if 'utmos' in self._models:
            try:
                import tempfile
                import os as _os
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as _tmp:
                    sf.write(_tmp.name, denoised_audio, self.sample_rate)
                    _tmp_path = _tmp.name
                try:
                    metrics.utmos = self._models['utmos'].predict_file(_tmp_path)
                finally:
                    try:
                        _os.unlink(_tmp_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"UTMOS计算失败: {e}")
        
        # 计算RTF
        audio_duration = len(denoised_audio) / self.sample_rate
        if audio_duration > 0:
            metrics.rtf = processing_time / audio_duration
        
        return metrics
    
    def _compute_si_sdr(self, reference: np.ndarray, estimated: np.ndarray) -> float:
        """
        计算SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
        
        Args:
            reference: 参考信号
            estimated: 估计信号
            
        Returns:
            SI-SDR值 (dB)
        """
        # 确保长度一致
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]
        
        # 归一化
        reference = reference - np.mean(reference)
        estimated = estimated - np.mean(estimated)
        
        # 计算投影
        alpha = np.dot(estimated, reference) / (np.dot(reference, reference) + 1e-10)
        
        # 计算目标和噪声
        target = alpha * reference
        noise = estimated - target
        
        # 计算SI-SDR
        si_sdr = 10 * np.log10(
            np.dot(target, target) / (np.dot(noise, noise) + 1e-10) + 1e-10
        )
        
        return float(si_sdr)
    
    def evaluate_file(
        self,
        denoised_path: str,
        reference_path: Optional[str] = None,
        processing_time: float = 0.0
    ) -> DenoiseMetrics:
        """
        评估音频文件
        
        Args:
            denoised_path: 降噪后音频路径
            reference_path: 参考音频路径(可选)
            processing_time: 处理时间
            
        Returns:
            DenoiseMetrics对象
        """
        # 加载降噪后音频
        denoised_audio, sr = librosa.load(denoised_path, sr=self.sample_rate)
        
        if reference_path and Path(reference_path).exists():
            # 有参考评估
            reference_audio, _ = librosa.load(reference_path, sr=self.sample_rate)
            return self.evaluate_with_reference(denoised_audio, reference_audio, processing_time)
        else:
            # 无参考评估
            return self.evaluate_without_reference(denoised_audio, processing_time)


class BatchEvaluator:
    """
    批量测评器
    支持多算法、多文件的批量测评
    """

    def __init__(self, output_dir: str = "./data/denoise_results"):
        """
        初始化批量测评器

        Args:
            output_dir: 结果输出目录
        """
        import logging
        self.logger = logging.getLogger("audiomos")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = DenoiseEvaluator()

        self.logger.info(f"[BatchEvaluator] 批量测评器初始化完成，输出目录: {output_dir}")
    
    def evaluate_algorithm(
        self,
        algorithm_name: str,
        noisy_files: List[str],
        reference_files: Optional[List[str]] = None,
        output_subdir: Optional[str] = None
    ) -> List[DenoiseEvaluation]:
        """
        评估单个算法

        Args:
            algorithm_name: 算法名称
            noisy_files: 带噪音频文件列表
            reference_files: 参考音频文件列表(可选)
            output_subdir: 输出子目录

        Returns:
            测评结果列表
        """
        self.logger.info("=" * 60)
        self.logger.info(f"[BatchEvaluator] 开始评估算法: {algorithm_name}")
        self.logger.info(f"[BatchEvaluator] 文件数量: {len(noisy_files)}")

        # 获取降噪器
        self.logger.info(f"[BatchEvaluator] 从Registry获取降噪器: {algorithm_name}")
        model_start = time.time()
        denoiser = DenoiserRegistry.get(algorithm_name)

        if denoiser is None:
            self.logger.error(f"[BatchEvaluator] 未找到算法: {algorithm_name}")
            raise ValueError(f"未找到算法: {algorithm_name}")

        self.logger.info(f"[BatchEvaluator] 获取降噪器成功 (耗时: {time.time() - model_start:.2f}s)")

        # 检查并初始化
        if not denoiser.is_initialized():
            self.logger.info(f"[BatchEvaluator] 降噪器未初始化，开始初始化...")
            init_start = time.time()
            init_success = denoiser.initialize()
            init_time = time.time() - init_start

            if init_success:
                self.logger.info(f"[BatchEvaluator] ✓ 降噪器初始化成功 (耗时: {init_time:.2f}s)")
            else:
                self.logger.error(f"[BatchEvaluator] ✗ 降噪器初始化失败 (耗时: {init_time:.2f}s)")
                raise RuntimeError(f"降噪器初始化失败: {algorithm_name}")
        else:
            self.logger.info(f"[BatchEvaluator] 降噪器已初始化，跳过初始化")
        
        # 创建输出目录
        if output_subdir:
            result_dir = self.output_dir / output_subdir / algorithm_name
        else:
            result_dir = self.output_dir / algorithm_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total_files = len(noisy_files)

        for i, noisy_file in enumerate(noisy_files):
            file_name = Path(noisy_file).name
            self.logger.info(f"[BatchEvaluator] 处理文件 {i+1}/{total_files}: {file_name}")
            
            # 执行降噪
            output_path = str(result_dir / f"denoised_{file_name}")
            self.logger.info(f"[BatchEvaluator]   执行降噪: {file_name}")
            denoise_start = time.time()
            denoise_result = denoiser.denoise_file(noisy_file, output_path)
            denoise_time = time.time() - denoise_start
            self.logger.info(f"[BatchEvaluator]   降噪完成: {file_name} (耗时: {denoise_time:.2f}s, RTF: {denoise_result.rtf:.3f})")

            # 评估
            ref_path = reference_files[i] if reference_files and i < len(reference_files) else None
            self.logger.info(f"[BatchEvaluator]   开始评估: {file_name}")
            eval_start = time.time()
            metrics = self.evaluator.evaluate_file(
                output_path,
                ref_path,
                denoise_result.processing_time
            )
            eval_time = time.time() - eval_start
            self.logger.info(f"[BatchEvaluator]   评估完成: {file_name} (耗时: {eval_time:.2f}s)")

            # 创建评估结果
            evaluation = DenoiseEvaluation(
                file_name=file_name,
                algorithm_name=algorithm_name,
                noisy_audio_path=noisy_file,
                denoised_audio_path=output_path,
                reference_audio_path=ref_path,
                metrics=metrics
            )

            results.append(evaluation)

        self.logger.info(f"[BatchEvaluator] 算法评估完成: {algorithm_name}, 共处理 {len(results)} 个文件")
        self.logger.info("=" * 60)

        return results
    
    def evaluate_multiple_algorithms(
        self,
        algorithm_names: List[str],
        noisy_files: List[str],
        reference_files: Optional[List[str]] = None
    ) -> Dict[str, List[DenoiseEvaluation]]:
        """
        评估多个算法
        
        Args:
            algorithm_names: 算法名称列表
            noisy_files: 带噪音频文件列表
            reference_files: 参考音频文件列表(可选)
            
        Returns:
            算法名称到测评结果列表的映射
        """
        all_results = {}
        
        for algorithm_name in algorithm_names:
            print(f"\n评估算法: {algorithm_name}")
            print("=" * 60)
            
            try:
                results = self.evaluate_algorithm(
                    algorithm_name,
                    noisy_files,
                    reference_files
                )
                all_results[algorithm_name] = results
                
                # 打印汇总
                self._print_summary(algorithm_name, results)
                
            except Exception as e:
                print(f"算法 {algorithm_name} 评估失败: {e}")
                all_results[algorithm_name] = []
        
        return all_results
    
    def _print_summary(self, algorithm_name: str, results: List[DenoiseEvaluation]):
        """打印算法评估汇总"""
        print(f"\n{algorithm_name} 评估汇总:")
        print("-" * 60)
        
        if not results:
            print("无结果")
            return
        
        # 收集所有指标
        pesq_scores = [r.metrics.pesq for r in results if r.metrics.pesq is not None]
        stoi_scores = [r.metrics.stoi for r in results if r.metrics.stoi is not None]
        sisdr_scores = [r.metrics.sisdr for r in results if r.metrics.sisdr is not None]
        processing_times = [r.metrics.processing_time for r in results]
        
        if pesq_scores:
            print(f"  PESQ: {np.mean(pesq_scores):.3f} ± {np.std(pesq_scores):.3f}")
        if stoi_scores:
            print(f"  STOI: {np.mean(stoi_scores):.3f} ± {np.std(stoi_scores):.3f}")
        if sisdr_scores:
            print(f"  SI-SDR: {np.mean(sisdr_scores):.2f} ± {np.std(sisdr_scores):.2f} dB")
        if processing_times:
            print(f"  平均处理时间: {np.mean(processing_times):.3f}s")
        
        print("-" * 60)
    
    def export_results(
        self,
        results: Dict[str, List[DenoiseEvaluation]],
        output_file: Optional[str] = None
    ) -> str:
        """
        导出测评结果到JSON
        
        Args:
            results: 测评结果字典
            output_file: 输出文件路径(可选)
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"evaluation_results_{timestamp}.json")
        
        # 转换为可序列化的字典
        export_data = {}
        for algo_name, evaluations in results.items():
            export_data[algo_name] = [e.to_dict() for e in evaluations]
        
        # 保存为JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"结果已导出到: {output_file}")
        return output_file


# 导入os模块(用于文件路径处理)
import os
