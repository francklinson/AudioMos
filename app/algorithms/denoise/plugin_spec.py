"""
自研降噪算法插件规范

提供清晰的插件接口规范，让自研算法能无缝接入测评系统。

插件目录结构:
    models/custom_denoisers/
    └── my_denoiser/
        ├── plugin.py          # 插件入口（必需）
        ├── model.pt           # 模型权重
        ├── README.md          # 插件说明（可选）
        └── requirements.txt   # 额外依赖（可选）

plugin.py 示例:
    from denoise.base import BaseDenoiser, DenoiseResult
    from denoise.plugin_spec import PluginMetadata, register_plugin

    class MyDenoiser(BaseDenoiser):
        def initialize(self) -> bool: ...
        def denoise(self, audio, sample_rate=None) -> DenoiseResult: ...

    @register_plugin
    class MyPlugin(MyDenoiser):
        METADATA = PluginMetadata(
            name="my_denoiser",
            version="1.0.0",
            description="我的降噪算法",
            model_format="pytorch",
            supported_sample_rates=[16000],
            author="Your Name",
        )
"""

import os
import sys
import json
import logging
import importlib
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseDenoiser, DenoiseResult

logger = logging.getLogger(__name__)


# ===========================
# 插件元数据
# ===========================


@dataclass
class PluginMetadata:
    """插件元数据"""
    name: str                           # 插件名称（唯一标识）
    version: str = "1.0.0"              # 版本
    description: str = ""               # 描述
    model_format: str = "pytorch"       # 模型格式: pytorch/onnx/tensorrt
    supported_sample_rates: List[int] = field(default_factory=lambda: [16000])
    author: str = ""
    paper_url: Optional[str] = None
    repository_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    known_metrics: Dict[str, float] = field(default_factory=dict)  # 已知指标基线


# ===========================
# 插件基类
# ===========================


class CustomDenoiserPlugin(BaseDenoiser):
    """
    自研降噪算法插件基类

    开发者只需实现以下方法即可接入测评系统:
    - initialize(): 加载模型
    - denoise(audio, sample_rate): 执行降噪

    可选实现:
    - get_plugin_metadata(): 返回插件元数据
    - export_to_onnx(save_path): 导出为ONNX格式
    - cleanup(): 清理资源
    """

    # 插件元数据（子类覆盖）
    METADATA: PluginMetadata = PluginMetadata(
        name="custom_denoiser",
        description="自定义降噪算法",
    )

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化自定义降噪插件

        Args:
            model_path: 模型权重路径
            device: 计算设备
        """
        super().__init__(
            name=self.METADATA.name,
            sample_rate=self.METADATA.supported_sample_rates[0],
            device=device,
        )
        self.model_path = model_path
        self._plugin_dir = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def get_plugin_metadata(cls) -> PluginMetadata:
        """获取插件元数据"""
        return cls.METADATA

    def export_to_onnx(self, save_path: str, dummy_input_shape: tuple = (1, 16000)) -> bool:
        """
        导出模型为ONNX格式

        Args:
            save_path: 保存路径
            dummy_input_shape: 虚拟输入shape

        Returns:
            是否导出成功
        """
        try:
            from .model_format_adapter import ModelFormatAdapter
            return ModelFormatAdapter.convert_pytorch_to_onnx(
                self._model, dummy_input_shape, save_path
            )
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            return False

    def cleanup(self):
        """清理资源（可选覆盖）"""
        self._model = None
        self._is_initialized = False


# ===========================
# 插件注册装饰器
# ===========================


def register_plugin(cls):
    """
    装饰器: 将自定义降噪类注册为插件

    使用方式:
        @register_plugin
        class MyDenoiser(CustomDenoiserPlugin):
            ...
    """
    if not issubclass(cls, BaseDenoiser):
        raise TypeError(f"{cls.__name__} 必须继承 BaseDenoiser")

    metadata = getattr(cls, 'METADATA', None)
    if metadata is None:
        metadata = PluginMetadata(name=cls.__name__)

    # 延迟注册（在导入时通过DenoiserRegistry完成）
    cls._is_plugin = True
    cls._plugin_metadata = metadata

    return cls


# ===========================
# 插件发现与加载
# ===========================


class PluginLoader:
    """
    插件发现和加载器

    扫描指定目录，自动发现并加载符合规范的插件。

    使用方式:
        loader = PluginLoader()
        plugins = loader.discover_plugins("./models/custom_denoisers")
        for name, plugin_class in plugins.items():
            denoiser = plugin_class()
            denoiser.initialize()
    """

    # 默认插件搜索路径
    DEFAULT_PLUGIN_DIRS = [
        "./models/custom_denoisers",
        "./app/algorithms/denoise/plugins",
    ]

    def __init__(self, additional_paths: Optional[List[str]] = None):
        """
        初始化插件加载器

        Args:
            additional_paths: 额外的插件搜索路径
        """
        self.plugin_dirs = list(self.DEFAULT_PLUGIN_DIRS)
        if additional_paths:
            self.plugin_dirs.extend(additional_paths)

    def discover_plugins(self, plugin_dir: Optional[str] = None) -> Dict[str, Type[BaseDenoiser]]:
        """
        发现并加载所有可用插件

        Args:
            plugin_dir: 指定搜索目录（None=搜索所有默认目录）

        Returns:
            {plugin_name: plugin_class} 字典
        """
        discovered = {}

        search_dirs = [plugin_dir] if plugin_dir else self.plugin_dirs

        for directory in search_dirs:
            abs_dir = os.path.abspath(directory)
            if not os.path.isdir(abs_dir):
                continue

            for item in sorted(os.listdir(abs_dir)):
                plugin_path = os.path.join(abs_dir, item)

                # 跳过非目录
                if not os.path.isdir(plugin_path):
                    continue

                # 跳过特殊目录
                if item.startswith(".") or item.startswith("__"):
                    continue

                # 查找 plugin.py
                plugin_file = os.path.join(plugin_path, "plugin.py")
                if not os.path.isfile(plugin_file):
                    continue

                try:
                    plugin_class = self._load_plugin_from_file(plugin_file, plugin_path)
                    if plugin_class:
                        metadata = plugin_class.get_plugin_metadata() if hasattr(plugin_class, 'get_plugin_metadata') else None
                        name = metadata.name if metadata else item
                        discovered[name] = plugin_class
                        logger.info(f"发现插件: {name} v{metadata.version if metadata else '?'} ({plugin_path})")
                except Exception as e:
                    logger.warning(f"加载插件失败 {plugin_path}: {e}")

        return discovered

    def load_plugin(self, plugin_name_or_path: str) -> Optional[Type[BaseDenoiser]]:
        """
        加载单个插件

        Args:
            plugin_name_or_path: 插件名称或路径

        Returns:
            插件类或None
        """
        # 尝试作为路径
        if os.path.isdir(plugin_name_or_path):
            plugin_file = os.path.join(plugin_name_or_path, "plugin.py")
            if os.path.isfile(plugin_file):
                return self._load_plugin_from_file(plugin_file, plugin_name_or_path)

        # 尝试在默认搜索路径中查找
        for plugin_dir in self.plugin_dirs:
            full_path = os.path.join(plugin_dir, plugin_name_or_path)
            if os.path.isdir(full_path):
                plugin_file = os.path.join(full_path, "plugin.py")
                if os.path.isfile(plugin_file):
                    return self._load_plugin_from_file(plugin_file, full_path)

        # 尝试作为已注册名称查找
        all_plugins = self.discover_plugins()
        return all_plugins.get(plugin_name_or_path)

    def _load_plugin_from_file(
        self, plugin_file: str, plugin_dir: str
    ) -> Optional[Type[BaseDenoiser]]:
        """
        从 plugin.py 文件加载插件类

        Args:
            plugin_file: plugin.py 的绝对路径
            plugin_dir: 插件目录

        Returns:
            插件类或None
        """
        # 将插件目录添加到 sys.path
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        # 动态导入
        spec = importlib.util.spec_from_file_location(
            f"denoise_plugin_{os.path.basename(plugin_dir)}",
            plugin_file,
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 查找继承自 BaseDenoiser 且有 _is_plugin 标记的类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseDenoiser) and
                attr is not BaseDenoiser and
                attr is not CustomDenoiserPlugin and
                getattr(attr, '_is_plugin', False)):
                return attr

        # 回退: 查找任何继承自 BaseDenoiser 的类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseDenoiser) and
                attr is not BaseDenoiser and
                attr is not CustomDenoiserPlugin):
                return attr

        return None

    def get_plugin_info(self, plugin_class: Type[BaseDenoiser]) -> Optional[Dict]:
        """获取插件的详细信息"""
        if hasattr(plugin_class, 'get_plugin_metadata'):
            meta = plugin_class.get_plugin_metadata()
            return {
                "name": meta.name,
                "version": meta.version,
                "description": meta.description,
                "model_format": meta.model_format,
                "supported_sample_rates": meta.supported_sample_rates,
                "author": meta.author,
                "paper_url": meta.paper_url,
                "tags": meta.tags,
            }
        return None

    def validate_plugin(self, plugin_class: Type[BaseDenoiser]) -> Dict[str, bool]:
        """
        验证插件是否符合规范

        Args:
            plugin_class: 插件类

        Returns:
            验证结果字典
        """
        results = {
            "inherits_base_denoiser": issubclass(plugin_class, BaseDenoiser),
            "has_initialize": hasattr(plugin_class, 'initialize'),
            "has_denoise": hasattr(plugin_class, 'denoise'),
            "has_metadata": hasattr(plugin_class, 'METADATA'),
        }

        # 检查方法签名
        try:
            instance = plugin_class()
            results["can_instantiate"] = True
            results["has_is_initialized"] = hasattr(instance, 'is_initialized')
        except Exception:
            results["can_instantiate"] = False

        return results
