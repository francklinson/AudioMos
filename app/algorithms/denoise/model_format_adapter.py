"""
模型格式适配器

统一不同推理框架的加载和推理接口。
支持: PyTorch → ONNX → TensorRT 的转换和推理。

使用方式:
    # 加载
    model = ModelFormatAdapter.load_model("model.pt", format="pytorch")
    model_onnx = ModelFormatAdapter.load_model("model.onnx", format="onnx")

    # 转换
    ModelFormatAdapter.convert_pytorch_to_onnx(pytorch_model, dummy_input, "model.onnx")
"""

import os
import logging
from typing import Any, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ModelFormatAdapter:
    """
    模型格式适配器

    封装不同推理框架的差异，提供统一的加载和推理接口。
    """

    # 支持的格式
    SUPPORTED_FORMATS = ["pytorch", "onnx", "tensorrt"]

    # ===========================
    # 格式检测
    # ===========================

    @staticmethod
    def detect_format(model_path: str) -> Optional[str]:
        """
        自动检测模型格式

        Args:
            model_path: 模型文件路径

        Returns:
            格式类型或None
        """
        path = Path(model_path)

        if not path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return None

        # 基于扩展名
        ext = path.suffix.lower()
        ext_map = {
            ".pt": "pytorch",
            ".pth": "pytorch",
            ".ckpt": "pytorch",
            ".onnx": "onnx",
            ".trt": "tensorrt",
            ".engine": "tensorrt",
            ".plan": "tensorrt",
        }
        if ext in ext_map:
            return ext_map[ext]

        # 基于目录结构
        if path.is_dir():
            # 检查目录内容
            contents = os.listdir(model_path)
            for f in contents:
                for ext, fmt in ext_map.items():
                    if f.endswith(ext):
                        return fmt

        logger.warning(f"无法检测模型格式: {model_path}")
        return None

    # ===========================
    # 模型加载
    # ===========================

    @staticmethod
    def load_model(
        model_path: str,
        format: Optional[str] = None,
        device: str = "cuda",
    ) -> Any:
        """
        加载模型（自动检测或指定格式）

        Args:
            model_path: 模型路径
            format: 模型格式（None=自动检测）
            device: 目标设备

        Returns:
            加载的模型对象
        """
        if format is None:
            format = ModelFormatAdapter.detect_format(model_path)

        if format is None:
            raise ValueError(f"无法确定模型格式: {model_path}")

        if format == "pytorch":
            return ModelFormatAdapter.load_pytorch_model(model_path, device)
        elif format == "onnx":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
            return ModelFormatAdapter.load_onnx_model(model_path, providers)
        elif format == "tensorrt":
            return ModelFormatAdapter.load_tensorrt_model(model_path)
        else:
            raise ValueError(f"不支持的格式: {format}，支持: {ModelFormatAdapter.SUPPORTED_FORMATS}")

    @staticmethod
    def load_pytorch_model(model_path: str, device: str = "cuda") -> Any:
        """
        加载PyTorch模型

        支持:
        - 单个 .pt/.pth 文件
        - HuggingFace 目录 (config.json + pytorch_model.bin)
        - 任意目录中的模型文件

        Args:
            model_path: 模型路径
            device: 设备

        Returns:
            加载的模型
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch 未安装")

        path = Path(model_path)

        # HuggingFace 格式
        if path.is_dir():
            config_path = path / "config.json"
            if config_path.exists():
                try:
                    import json
                    with open(config_path) as f:
                        config = json.load(f)
                    # 尝试使用transformers加载
                    try:
                        from transformers import AutoModel, AutoConfig
                        hf_config = AutoConfig.from_pretrained(str(path))
                        model = AutoModel.from_config(hf_config)
                        model.to(device)
                        return model
                    except Exception:
                        pass
                except Exception:
                    pass

            # 查找.pt/.bin文件
            for ext in [".pt", ".pth", ".bin", ".ckpt"]:
                for f in path.glob(f"*{ext}"):
                    try:
                        model = torch.load(str(f), map_location=device)
                        return model
                    except Exception:
                        continue

        # 单个文件
        elif path.is_file():
            try:
                model = torch.load(str(path), map_location=device)
                return model
            except Exception:
                pass

        raise FileNotFoundError(f"无法加载PyTorch模型: {model_path}")

    @staticmethod
    def load_onnx_model(
        model_path: str,
        providers: Optional[List[str]] = None,
    ) -> Any:
        """
        加载ONNX模型

        Args:
            model_path: ONNX模型路径
            providers: 执行提供商列表

        Returns:
            onnxruntime.InferenceSession
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime 未安装，请运行: pip install onnxruntime-gpu")

        if providers is None:
            # 自动选择最优provider
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )

        logger.info(f"ONNX模型加载成功: {model_path}")
        logger.info(f"  输入: {[i.name for i in session.get_inputs()]}")
        logger.info(f"  输出: {[o.name for o in session.get_outputs()]}")
        logger.info(f"  Provider: {session.get_providers()}")

        return session

    @staticmethod
    def load_tensorrt_model(model_path: str) -> Any:
        """
        加载TensorRT模型

        Args:
            model_path: .trt/.engine 文件路径

        Returns:
            TensorRT 推理引擎
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT 未安装")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        logger.info(f"TensorRT模型加载成功: {model_path}")
        return engine

    # ===========================
    # 格式转换
    # ===========================

    @staticmethod
    def convert_pytorch_to_onnx(
        model: Any,
        dummy_input_shape: tuple = (1, 16000),
        save_path: str = "model.onnx",
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[dict] = None,
    ) -> bool:
        """
        PyTorch → ONNX 转换

        Args:
            model: PyTorch模型
            dummy_input_shape: 虚拟输入shape
            save_path: 保存路径
            input_names: 输入节点名
            output_names: 输出节点名
            dynamic_axes: 动态轴配置

        Returns:
            是否转换成功
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch 未安装")

        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # 创建虚拟输入
        dummy_input = torch.randn(*dummy_input_shape, device=next(model.parameters()).device)

        # 导出
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # 验证
        try:
            import onnx
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX模型验证通过: {save_path}")
        except Exception as e:
            logger.warning(f"ONNX验证失败: {e}")

        logger.info(f"PyTorch → ONNX 转换成功: {save_path}")
        return True

    @staticmethod
    def convert_onnx_to_tensorrt(
        onnx_path: str,
        save_path: str = "model.engine",
        fp16: bool = True,
        max_workspace_size: int = 2 * 1024 * 1024 * 1024,  # 2GB
    ) -> bool:
        """
        ONNX → TensorRT 转换

        Args:
            onnx_path: ONNX模型路径
            save_path: TensorRT模型保存路径
            fp16: 是否使用FP16精度
            max_workspace_size: 最大工作空间大小

        Returns:
            是否转换成功
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT 未安装")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # 解析ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX解析错误: {parser.get_error(i)}")
                return False

        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("启用FP16精度")

        # 构建引擎
        engine = builder.build_serialized_network(network, config)

        if engine is None:
            logger.error("TensorRT引擎构建失败")
            return False

        with open(save_path, "wb") as f:
            f.write(engine)

        logger.info(f"ONNX → TensorRT 转换成功: {save_path}")
        return True

    # ===========================
    # 统一推理接口
    # ===========================

    @staticmethod
    def infer(model: Any, audio: np.ndarray, format: str = "pytorch") -> np.ndarray:
        """
        统一推理接口

        Args:
            model: 模型对象
            audio: 输入音频 [T] 或 [1, T]
            format: 模型格式

        Returns:
            输出音频 [T]
        """
        if format == "pytorch":
            return ModelFormatAdapter._infer_pytorch(model, audio)
        elif format == "onnx":
            return ModelFormatAdapter._infer_onnx(model, audio)
        elif format == "tensorrt":
            return ModelFormatAdapter._infer_tensorrt(model, audio)
        else:
            raise ValueError(f"不支持的格式: {format}")

    @staticmethod
    def _infer_pytorch(model: Any, audio: np.ndarray) -> np.ndarray:
        """PyTorch推理"""
        import torch

        model.eval()
        with torch.no_grad():
            if audio.ndim == 1:
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            else:
                audio_tensor = torch.from_numpy(audio).float()

            audio_tensor = audio_tensor.to(next(model.parameters()).device)
            output = model(audio_tensor)

            if isinstance(output, tuple):
                output = output[0]

            return output.squeeze().cpu().numpy()

    @staticmethod
    def _infer_onnx(session: Any, audio: np.ndarray) -> np.ndarray:
        """ONNX Runtime推理"""
        if audio.ndim == 1:
            audio_input = audio[np.newaxis, :].astype(np.float32)
        else:
            audio_input = audio.astype(np.float32)

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: audio_input})

        return output[0].squeeze()

    @staticmethod
    def _infer_tensorrt(engine: Any, audio: np.ndarray) -> np.ndarray:
        """TensorRT推理（简化版）"""
        # TensorRT推理需要更复杂的设置（buffer分配等）
        # 此处提供基础框架，完整实现需要根据具体情况定制
        raise NotImplementedError(
            "TensorRT推理需要根据模型具体实现I/O buffer管理。"
            "请参考NVIDIA官方文档: "
            "https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/"
        )
