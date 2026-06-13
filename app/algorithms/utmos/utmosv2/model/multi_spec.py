from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from utmosv2.dataset._utils import get_dataset_num

if TYPE_CHECKING:
    from utmosv2._settings._config import Config


def get_project_root() -> Path:
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    # 从 multi_spec.py 向上回溯到项目根目录
    # app/algorithms/utmos/utmosv2/model/multi_spec.py -> 项目根目录
    return current_file.parent.parent.parent.parent.parent.parent


def load_model_from_local(cfg: Config, device: torch.device = torch.device("cpu")):
    """
    从本地 models/timm 目录加载模型，避免从 Hugging Face 下载
    
    Args:
        cfg: 配置对象
        device: 加载设备
    
    Returns:
        加载了预训练权重的模型
    """
    backbone_name = cfg.model.multi_spec.backbone
    
    # 首先创建模型架构 (pretrained=False 避免自动下载)
    print(f"[load_model_from_local] 创建模型架构: {backbone_name} (pretrained=False)")
    model = timm.create_model(
        backbone_name,
        pretrained=False,
        num_classes=0,
    )
    
    # 构建本地模型文件路径
    project_root = get_project_root()
    local_model_path = project_root / "models" / "timm" / f"{backbone_name}.safetensors"
    
    print(f"[load_model_from_local] 查找本地模型: {local_model_path}")
    
    if local_model_path.exists():
        print(f"[load_model_from_local] 找到本地模型，开始加载权重...")
        state_dict = load_file(str(local_model_path))
        
        # 过滤掉不需要的键（如 classifier，因为 num_classes=0）
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('classifier.'):
                # 跳过 classifier 层的权重
                continue
            filtered_state_dict[key] = value
        
        # 加载过滤后的权重
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"[load_model_from_local] 缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"[load_model_from_local] 忽略的键: {unexpected_keys}")
        
        print(f"[load_model_from_local] 本地权重加载完成")
    else:
        # 离线部署：不尝试网络下载，直接报错
        raise RuntimeError(
            f"本地 timm 模型不存在: {local_model_path}\n"
            f"  离线部署前请将 {backbone_name}.safetensors 放入 models/timm/ 目录。\n"
            f"  下载方式: python -c \"import timm; timm.create_model('{backbone_name}', pretrained=True)\""
        )
    
    return model


class MultiSpecModelV2(nn.Module):
    """
    A multi-spectrogram model (version 2) that processes multiple spectrograms
    and combines their outputs using learnable weights. This model supports
    attention-based pooling and a flexible number of spectrogram frames.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing model and dataset settings.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        print(f"[MultiSpecModelV2] 开始初始化...")
        print(f"[MultiSpecModelV2] backbone名称: {cfg.model.multi_spec.backbone}")
        print(f"[MultiSpecModelV2] specs数量: {len(cfg.dataset.specs)}")
        print(f"[MultiSpecModelV2] 使用本地模型加载...")
        
        self.backbones = nn.ModuleList()
        for i in range(len(cfg.dataset.specs)):
            print(f"[MultiSpecModelV2] 创建 backbone {i+1}/{len(cfg.dataset.specs)}: {cfg.model.multi_spec.backbone}")
            backbone = load_model_from_local(cfg)
            self.backbones.append(backbone)
            print(f"[MultiSpecModelV2] backbone {i+1} 创建完成")
        
        print(f"[MultiSpecModelV2] 所有 backbone 创建完成")
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(
            F.softmax(torch.randn(len(cfg.dataset.specs)), dim=0)
        )

        self.pooling = timm.layers.SelectAdaptivePool2d(
            output_size=(None, 1) if self.cfg.model.multi_spec.atten else 1,  # type: ignore
            pool_type=self.cfg.model.multi_spec.pool_type,
            flatten=False,
        )

        if self.cfg.model.multi_spec.atten:
            self.attn = nn.MultiheadAttention(
                embed_dim=cast(int, self.backbones[0].num_features)
                * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1),
                num_heads=8,
                dropout=0.2,
                batch_first=True,
            )

        fc_in_features = (
            cast(int, self.backbones[0].num_features)
            * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1)
            * (2 if self.cfg.model.multi_spec.atten else 1)
        )

        self.fc: nn.Linear | nn.Identity = nn.Linear(
            fc_in_features, cfg.model.multi_spec.num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiSpecModelV2.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, num_frames, channels, width, height).

        Returns:
            torch.Tensor:
                Output tensor after applying backbones, pooling, and fully connected layers.
        """
        xl = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(
                self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
            )
        ]
        xl = [
            self.backbones[i % len(self.cfg.dataset.specs)](t) for i, t in enumerate(xl)
        ]
        xl = [
            sum(
                [
                    xl[i * len(self.cfg.dataset.specs) + j] * w
                    for j, w in enumerate(self.weights)
                ]
            )
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        x = torch.cat(xl, dim=3)
        x = self.pooling(x).squeeze(3)
        if self.cfg.model.multi_spec.atten:
            xt = torch.permute(x, (0, 2, 1))
            y, _ = self.attn(xt, xt, xt)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)
        x = self.fc(x)
        return x


class MultiSpecExtModel(nn.Module):
    """
    An extended version of the MultiSpecModel that incorporates data-domain id
    in addition to the spectrograms. This model allows the fusion of
    data-domain embeddings with multi-spectrogram features.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing model and dataset settings.

    Returns:
        torch.Tensor:
            The model's output after processing the input and data-domain id
            through backbones, pooling, and fully connected layers.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        print(f"[MultiSpecExtModel] 开始初始化...")
        print(f"[MultiSpecExtModel] backbone名称: {cfg.model.multi_spec.backbone}")
        print(f"[MultiSpecExtModel] specs数量: {len(cfg.dataset.specs)}")
        print(f"[MultiSpecExtModel] 使用本地模型加载...")
        
        self.backbones = nn.ModuleList()
        for i in range(len(cfg.dataset.specs)):
            print(f"[MultiSpecExtModel] 创建 backbone {i+1}/{len(cfg.dataset.specs)}: {cfg.model.multi_spec.backbone}")
            backbone = load_model_from_local(cfg)
            self.backbones.append(backbone)
            print(f"[MultiSpecExtModel] backbone {i+1} 创建完成")
        
        print(f"[MultiSpecExtModel] 所有 backbone 创建完成")
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(
            F.softmax(torch.randn(len(cfg.dataset.specs)), dim=0)
        )

        self.pooling = timm.layers.SelectAdaptivePool2d(
            output_size=(None, 1) if self.cfg.model.multi_spec.atten else 1,  # type: ignore
            pool_type=self.cfg.model.multi_spec.pool_type,
            flatten=False,
        )

        if self.cfg.model.multi_spec.atten:
            self.attn = nn.MultiheadAttention(
                embed_dim=cast(int, self.backbones[0].num_features)
                * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1),
                num_heads=8,
                dropout=0.2,
                batch_first=True,
            )

        fc_in_features = (
            cast(int, self.backbones[0].num_features)
            * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1)
            * (2 if self.cfg.model.multi_spec.atten else 1)
        )

        self.num_dataset = get_dataset_num(cfg)

        self.fc: nn.Linear | nn.Identity = nn.Linear(
            fc_in_features + self.num_dataset, cfg.model.multi_spec.num_classes
        )

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        xl = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(
                self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
            )
        ]
        xl = [
            self.backbones[i % len(self.cfg.dataset.specs)](t) for i, t in enumerate(xl)
        ]
        xl = [
            sum(
                [
                    xl[i * len(self.cfg.dataset.specs) + j] * w
                    for j, w in enumerate(self.weights)
                ]
            )
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        x = torch.cat(xl, dim=3)
        x = self.pooling(x).squeeze(3)
        if self.cfg.model.multi_spec.atten:
            xt = torch.permute(x, (0, 2, 1))
            y, _ = self.attn(xt, xt, xt)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        return x
