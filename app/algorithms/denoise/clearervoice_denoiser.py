"""
ClearerVoice-Studio语音增强算法
阿里巴巴开源的先进语音处理工具包
包含FRCRN、MossFormer等模型
"""

import numpy as np
import torch
import librosa
from typing import Optional
import time
import os

from .base import BaseDenoiser, DenoiseResult
from .registry import DenoiserRegistry


def _patch_modelscope_compat():
    """在导入 modelscope 前应用 datasets 兼容性补丁"""
    try:
        import datasets

        if not hasattr(datasets, "LargeList"):
            class _LargeListStub(list):
                pass

            datasets.LargeList = _LargeListStub

        import datasets.features.features as _ds_ff

        if not hasattr(_ds_ff, "_FEATURE_TYPES"):
            from datasets.features.features import (
                Value,
                ClassLabel,
                Array2D,
                Array3D,
                Array4D,
                Array5D,
            )

            _FEATURE_TYPES = {
                "Value": Value,
                "ClassLabel": ClassLabel,
                "Sequence": datasets.Sequence,
                "Array2D": Array2D,
                "Array3D": Array3D,
                "Array4D": Array4D,
                "Array5D": Array5D,
            }
            _ds_ff._FEATURE_TYPES = _FEATURE_TYPES
    except Exception:
        pass


class ClearerVoiceDenoiser(BaseDenoiser):
    """
    ClearerVoice-Studio语音增强器
    支持FRCRN和MossFormer系列模型
    """
    
    def __init__(self, model_type: str = "frcrn", sample_rate: int = 16000,
                 device: str = "cuda", model_dir: str = "./models/clearervoice"):
        """
        初始化ClearerVoice降噪器
        
        Args:
            model_type: 模型类型 (frcrn/mossformer/mossformer2)
            sample_rate: 采样率
            device: 计算设备
            model_dir: 模型保存目录
        """
        name = f"clearervoice_{model_type}"
        super().__init__(name, sample_rate, device)
        self.model_type = model_type
        self.model_dir = model_dir
        self._model = None
        self._model_path = None
    
    def initialize(self) -> bool:
        """
        初始化ClearerVoice模型
        
        Returns:
            是否初始化成功
        """
        try:
            # 检查ClearerVoice是否已安装
            try:
                from clearervoice import VoiceEnhancer
            except ImportError:
                print("ClearerVoice-Studio未安装，尝试使用备用方案")
                return self._init_fallback()
            
            # 确保模型目录存在
            os.makedirs(self.model_dir, exist_ok=True)
            
            # 根据模型类型选择配置
            if self.model_type == "frcrn":
                self._model = VoiceEnhancer(
                    model_name="frcrn",
                    device=self.device,
                    model_dir=self.model_dir
                )
            elif self.model_type in ["mossformer", "mossformer2"]:
                self._model = VoiceEnhancer(
                    model_name=self.model_type,
                    device=self.device,
                    model_dir=self.model_dir
                )
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"ClearerVoice模型初始化失败: {e}")
            return self._init_fallback()
    
    def _init_fallback(self) -> bool:
        """
        备用初始化方案
        使用ModelScope加载模型
        """
        try:
            # 在导入 modelscope 之前应用兼容性补丁
            _patch_modelscope_compat()
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            # 使用ModelScope的语音增强pipeline
            model_id_map = {
                "frcrn": "damo/speech_frcrn_ans_cirm_16k",
                "mossformer": "damo/speech_mossformer_separation_16k",
                "mossformer2": "damo/speech_mossformer2_separation_16k"
            }
            
            model_id = model_id_map.get(self.model_type, model_id_map["frcrn"])
            
            self._model = pipeline(
                Tasks.acoustic_noise_suppression,
                model=model_id,
                device=self.device
            )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"备用初始化也失败: {e}")
            self._is_initialized = False
            return False
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """
        执行语音增强
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            
        Returns:
            DenoiseResult对象
        """
        start_time = time.time()
        
        if not self._is_initialized:
            self.initialize()
        
        # 重采样
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        try:
            # 根据模型类型执行增强
            if hasattr(self._model, 'enhance'):
                # ClearerVoice原生API
                enhanced = self._model.enhance(audio, sr=self.sample_rate)
            elif hasattr(self._model, '__call__'):
                # ModelScope pipeline
                result = self._model(audio)
                enhanced = result['output']
            else:
                raise ValueError("未知的模型接口")
            
            processing_time = time.time() - start_time
            
            return DenoiseResult(
                audio=enhanced,
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name
            )
            
        except Exception as e:
            print(f"ClearerVoice增强失败: {e}")
            return DenoiseResult(
                audio=audio,
                sample_rate=self.sample_rate,
                processing_time=time.time() - start_time,
                algorithm_name=self.name
            )


class FRCRNDenoiser(BaseDenoiser):
    """
    FRCRN (Feature Recurrent Convolutional Recurrent Network)
    阿里达摩院开源的实时语音增强模型
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cuda",
                 model_dir: str = "./models/clearervoice"):
        """
        初始化FRCRN降噪器
        
        Args:
            sample_rate: 采样率
            device: 计算设备
            model_dir: 模型保存目录
        """
        super().__init__("clearervoice_frcrn", sample_rate, device)
        self.model_dir = model_dir
        self._enhancer = None
        self._model_source = None
    
    def initialize(self) -> bool:
        """初始化FRCRN模型"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            # 方案1: 尝试从本地模型目录加载
            local_model_path = os.path.join(self.model_dir, "frcrn")
            if os.path.exists(local_model_path) and any(os.listdir(local_model_path)):
                print(f"尝试从本地加载FRCRN模型: {local_model_path}")
                try:
                    _patch_modelscope_compat()
                    from modelscope.pipelines import pipeline
                    from modelscope.utils.constant import Tasks
                    
                    self._enhancer = pipeline(
                        Tasks.acoustic_noise_suppression,
                        model=local_model_path,
                        device=self.device
                    )
                    self._model_source = "local"
                    self._is_initialized = True
                    print("FRCRN本地模型加载成功")
                    return True
                except Exception as e1:
                    print(f"本地加载失败: {e1}")
            
            # 方案2: 从ModelScope在线加载
            print("尝试从ModelScope在线加载FRCRN模型...")
            try:
                _patch_modelscope_compat()
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks

                self._enhancer = pipeline(
                    Tasks.acoustic_noise_suppression,
                    model="iic/speech_frcrn_ans_cirm_16k",
                    device=self.device
                )
                self._model_source = "modelscope"
                self._is_initialized = True
                print("FRCRN在线模型加载成功")
                return True
                
            except Exception as e2:
                print(f"ModelScope在线加载失败: {e2}")
            
            # 方案3: 回退到SpeechBrain
            print("FRCRN所有加载方案失败，回退到SpeechBrain MetricGAN+")
            try:
                from speechbrain.inference.enhancement import SpectralMaskEnhancement
                
                sb_model_dir = os.path.join(self.model_dir, "speechbrain_fallback")
                self._enhancer = SpectralMaskEnhancement.from_hparams(
                    source="speechbrain/metricgan-plus-voicebank",
                    savedir=sb_model_dir,
                    run_opts={"device": self.device}
                )
                self._model_source = "speechbrain_fallback"
                self._is_initialized = True
                print("FRCRN回退模型(MetricGAN+)加载成功")
                return True
                
            except Exception as e3:
                print(f"SpeechBrain回退也失败: {e3}")
            
            self._is_initialized = False
            return False
            
        except Exception as e:
            print(f"FRCRN模型初始化失败: {e}")
            self._is_initialized = False
            return False
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """执行增强"""
        start_time = time.time()
        
        if not self._is_initialized:
            self.initialize()
        
        # FRCRN要求16kHz
        target_sr = 16000
        if sample_rate is not None and sample_rate != target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        
        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        try:
            # 根据模型来源执行增强
            if self._model_source == "speechbrain_fallback":
                # SpeechBrain API
                import torch
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                with torch.no_grad():
                    enhanced = self._enhancer.enhance_batch(audio_tensor, lengths=torch.tensor([1.0]))
                if isinstance(enhanced, torch.Tensor):
                    enhanced = enhanced.squeeze(0).cpu().numpy()
            else:
                # ModelScope pipeline - 需要传入文件路径而不是numpy数组
                # ModelScope ANSPipeline有bug，不支持直接传入numpy数组
                import tempfile
                import soundfile as sf
                import os
                
                # 创建临时文件
                temp_input = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_input_path = temp_input.name
                temp_input.close()
                
                try:
                    # 保存输入音频到临时文件
                    sf.write(temp_input_path, audio, target_sr)
                    
                    # 执行增强 - 传入文件路径
                    result = self._enhancer(temp_input_path)
                    
                    # 解析输出
                    if isinstance(result, dict):
                        if 'output_pcm' in result:
                            # ModelScope ANSPipeline返回bytes格式的16-bit PCM
                            pcm_data = result['output_pcm']
                            if isinstance(pcm_data, bytes):
                                enhanced = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                            elif isinstance(pcm_data, np.ndarray):
                                enhanced = pcm_data.astype(np.float32)
                            else:
                                enhanced = audio
                        elif 'output' in result:
                            enhanced = result['output']
                            if isinstance(enhanced, bytes):
                                enhanced = np.frombuffer(enhanced, dtype=np.int16).astype(np.float32) / 32768.0
                        else:
                            # 尝试获取第一个值
                            for key, val in result.items():
                                if isinstance(val, np.ndarray):
                                    enhanced = val.astype(np.float32)
                                    break
                                elif isinstance(val, bytes):
                                    enhanced = np.frombuffer(val, dtype=np.int16).astype(np.float32) / 32768.0
                                    break
                    else:
                        enhanced = audio
                        
                except Exception as pipeline_e:
                    print(f"ModelScope pipeline执行失败: {pipeline_e}")
                    enhanced = audio
                finally:
                    # 清理临时文件
                    try:
                        os.unlink(temp_input_path)
                    except:
                        pass
            
            # 重采样回目标采样率
            if self.sample_rate != target_sr:
                enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=self.sample_rate)
            
            processing_time = time.time() - start_time
            
            return DenoiseResult(
                audio=enhanced,
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name
            )
            
        except Exception as e:
            print(f"FRCRN增强失败: {e}")
            return DenoiseResult(
                audio=audio,
                sample_rate=self.sample_rate,
                processing_time=time.time() - start_time,
                algorithm_name=self.name
            )


class MossFormerDenoiser(BaseDenoiser):
    """
    MossFormer语音分离/增强器
    基于混合注意力机制的先进模型
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cuda",
                 model_dir: str = "./models/clearervoice", version: str = "2"):
        """
        初始化MossFormer降噪器
        
        Args:
            sample_rate: 采样率
            device: 计算设备
            model_dir: 模型保存目录
            version: 版本 ("1" 或 "2")
        """
        name = f"clearervoice_mossformer{version}"
        super().__init__(name, sample_rate, device)
        self.model_dir = model_dir
        self.version = version
        self._separator = None
        self._model_source = None
    
    def initialize(self) -> bool:
        """初始化MossFormer模型"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            # 方案1: 尝试从本地模型目录加载
            local_model_path = os.path.join(self.model_dir, f"mossformer{self.version}")
            if os.path.exists(local_model_path) and any(os.listdir(local_model_path)):
                print(f"尝试从本地加载MossFormer{self.version}模型: {local_model_path}")
                try:
                    _patch_modelscope_compat()
                    from modelscope.pipelines import pipeline
                    from modelscope.utils.constant import Tasks
                    
                    self._separator = pipeline(
                        Tasks.speech_separation,
                        model=local_model_path,
                        device=self.device
                    )
                    self._model_source = "local"
                    self._is_initialized = True
                    print(f"MossFormer{self.version}本地模型加载成功")
                    return True
                except Exception as e1:
                    print(f"本地加载失败: {e1}")
            
            # 方案2: 从ModelScope在线加载
            print(f"尝试从ModelScope在线加载MossFormer{self.version}模型...")
            try:
                _patch_modelscope_compat()
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks

                model_id = f"iic/speech_mossformer{'2' if self.version == '2' else ''}_separation_temporal_8k"
                
                self._separator = pipeline(
                    Tasks.speech_separation,
                    model=model_id,
                    device=self.device
                )
                self._model_source = "modelscope"
                self._is_initialized = True
                print(f"MossFormer{self.version}在线模型加载成功")
                return True
                
            except Exception as e2:
                print(f"ModelScope在线加载失败: {e2}")
            
            # 方案3: 回退到SpeechBrain SepFormer
            print(f"MossFormer{self.version}所有加载方案失败，回退到SpeechBrain SepFormer")
            try:
                from speechbrain.inference.separation import SepformerSeparation
                
                sb_model_dir = os.path.join(self.model_dir, "speechbrain_fallback")
                self._separator = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-wham-enhancement",
                    savedir=sb_model_dir,
                    run_opts={"device": self.device}
                )
                self._model_source = "speechbrain_fallback"
                self._is_initialized = True
                print(f"MossFormer{self.version}回退模型(SepFormer)加载成功")
                return True
                
            except Exception as e3:
                print(f"SpeechBrain回退也失败: {e3}")
            
            self._is_initialized = False
            return False
            
        except Exception as e:
            print(f"MossFormer模型初始化失败: {e}")
            self._is_initialized = False
            return False
    
    def denoise(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> DenoiseResult:
        """执行分离/增强"""
        start_time = time.time()
        
        if not self._is_initialized:
            self.initialize()
        
        # MossFormer要求16kHz，MossFormer2要求8kHz
        target_sr = 8000 if self.version == "2" else 16000
        if sample_rate is not None and sample_rate != target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        
        # 确保单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        try:
            import torch
            
            # 根据模型来源执行增强
            if self._model_source == "speechbrain_fallback":
                # SpeechBrain API
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                with torch.no_grad():
                    enhanced = self._separator.separate_batch(audio_tensor)
                if isinstance(enhanced, torch.Tensor):
                    enhanced = enhanced.squeeze(0).cpu().numpy()
            else:
                # ModelScope pipeline - 需要传入文件路径而不是numpy数组
                # ModelScope SeparationPipeline有bug，不支持直接传入numpy数组
                import tempfile
                import soundfile as sf
                import os
                
                # 创建临时文件
                temp_input = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_input_path = temp_input.name
                temp_input.close()
                
                try:
                    # 保存输入音频到临时文件
                    sf.write(temp_input_path, audio, target_sr)
                    
                    # 执行分离 - 传入文件路径
                    result = self._separator(temp_input_path)
                    
                    # 解析输出 - 取第一个分离结果(语音)
                    enhanced = audio  # 默认值
                    
                    if isinstance(result, dict):
                        if 'output_pcm' in result:
                            pcm_data = result['output_pcm']
                            if isinstance(pcm_data, bytes):
                                enhanced = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                            elif isinstance(pcm_data, np.ndarray):
                                enhanced = pcm_data.astype(np.float32)
                            elif isinstance(pcm_data, list):
                                enhanced = np.array(pcm_data[0] if pcm_data else audio, dtype=np.float32)
                            else:
                                enhanced = audio
                        elif 'output' in result:
                            output_data = result['output']
                            if isinstance(output_data, bytes):
                                enhanced = np.frombuffer(output_data, dtype=np.int16).astype(np.float32) / 32768.0
                            elif isinstance(output_data, list):
                                enhanced = np.array(output_data[0] if output_data else audio, dtype=np.float32)
                            elif isinstance(output_data, np.ndarray):
                                enhanced = output_data.astype(np.float32)
                            else:
                                enhanced = audio
                        elif 'separated' in result:
                            sep_data = result['separated']
                            if isinstance(sep_data, list):
                                enhanced = np.array(sep_data[0], dtype=np.float32) if sep_data else audio
                            elif isinstance(sep_data, np.ndarray):
                                enhanced = sep_data.astype(np.float32)
                            else:
                                enhanced = audio
                        else:
                            # 尝试获取第一个numpy数组或bytes值
                            found = False
                            for key, val in result.items():
                                if isinstance(val, np.ndarray):
                                    enhanced = val.astype(np.float32)
                                    found = True
                                    break
                                elif isinstance(val, bytes):
                                    enhanced = np.frombuffer(val, dtype=np.int16).astype(np.float32) / 32768.0
                                    found = True
                                    break
                                elif isinstance(val, list) and len(val) > 0:
                                    if isinstance(val[0], np.ndarray):
                                        enhanced = val[0].astype(np.float32)
                                        found = True
                                        break
                            if not found:
                                enhanced = audio
                    else:
                        enhanced = audio
                        
                except Exception as pipeline_e:
                    print(f"ModelScope pipeline执行失败: {pipeline_e}")
                    # 回退到SpeechBrain SepFormer
                    print("尝试回退到SpeechBrain SepFormer...")
                    try:
                        from speechbrain.inference.separation import SepformerSeparation
                        
                        sb_model_dir = os.path.join(self.model_dir, "speechbrain_fallback")
                        if not hasattr(self, '_fallback_sepformer') or self._fallback_sepformer is None:
                            self._fallback_sepformer = SepformerSeparation.from_hparams(
                                source="speechbrain/sepformer-wham-enhancement",
                                savedir=sb_model_dir,
                                run_opts={"device": self.device}
                            )
                        
                        import torch
                        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                        with torch.no_grad():
                            enhanced = self._fallback_sepformer.separate_batch(audio_tensor)
                        if isinstance(enhanced, torch.Tensor):
                            enhanced = enhanced.squeeze(0).cpu().numpy()
                        
                        self._model_source = "speechbrain_fallback"
                        print("SpeechBrain SepFormer回退成功")
                        
                    except Exception as fallback_e:
                        print(f"SpeechBrain回退也失败: {fallback_e}")
                        enhanced = audio
                finally:
                    # 清理临时文件
                    try:
                        os.unlink(temp_input_path)
                    except:
                        pass
            
            # 确保enhanced是numpy数组且形状正确
            if isinstance(enhanced, torch.Tensor):
                enhanced = enhanced.cpu().numpy()
            
            # 处理多维输出 (取第一个通道)
            if len(enhanced.shape) > 1:
                enhanced = enhanced[0] if enhanced.shape[0] <= 2 else enhanced[:, 0]
            
            # 确保长度匹配
            if len(enhanced) != len(audio):
                print(f"输出长度不匹配: {len(enhanced)} vs {len(audio)}")
                enhanced = audio
            
            # 重采样回目标采样率
            if self.sample_rate != target_sr:
                enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=self.sample_rate)
            
            processing_time = time.time() - start_time
            
            return DenoiseResult(
                audio=enhanced,
                sample_rate=self.sample_rate,
                processing_time=processing_time,
                algorithm_name=self.name
            )
            
        except Exception as e:
            print(f"MossFormer增强失败: {e}")
            return DenoiseResult(
                audio=audio,
                sample_rate=self.sample_rate,
                processing_time=time.time() - start_time,
                algorithm_name=self.name
            )


class MossFormer2Denoiser(MossFormerDenoiser):
    """
    MossFormer2语音分离/增强器
    MossFormer的改进版本，性能更优
    """

    def __init__(self, sample_rate: int = 16000, device: str = "cuda",
                 model_dir: str = "./models/clearervoice"):
        super().__init__(sample_rate=sample_rate, device=device,
                         model_dir=model_dir, version="2")


# 注册ClearerVoice降噪算法
DenoiserRegistry.register("clearervoice_frcrn", FRCRNDenoiser)
DenoiserRegistry.register("clearervoice_mossformer", MossFormerDenoiser)
DenoiserRegistry.register("clearervoice_mossformer2", MossFormer2Denoiser)
