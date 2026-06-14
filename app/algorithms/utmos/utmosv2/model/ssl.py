from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from utmosv2.dataset._utils import get_dataset_num

if TYPE_CHECKING:
    from utmosv2._settings._config import Config


import os
import time
from pathlib import Path

class _SSLEncoder(nn.Module):
    def __init__(self, sr: int, model_name: str, freeze: bool):
        super().__init__()
        self.sr = sr
        
        print(f"\n[UTMOS._SSLEncoder] 初始化SSL编码器")
        print(f"  模型名称: {model_name}")
        print(f"  采样率: {sr} Hz")
        print(f"  冻结参数: {freeze}")
        
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        local_model_path = project_root / "models" / "wav2vec2" / "facebook--wav2vec2-base"
        
        load_start = time.time()
        
        # 离线优先: 本地项目路径 > HF缓存路径 > 报错
        loaded = False

        # 路径1: 项目本地模型
        if "facebook/wav2vec2-base" in model_name and local_model_path.exists():
            print(f"\n[UTMOS._SSLEncoder] 使用本地wav2vec2模型")
            print(f"  路径: {local_model_path}")
            try:
                self.processor = AutoFeatureExtractor.from_pretrained(str(local_model_path), local_files_only=True)
                self.model = AutoModel.from_pretrained(str(local_model_path), local_files_only=True)
                loaded = True
                print(f"  ✓ 本地模型加载完成")
            except Exception as e:
                print(f"  ⚠️ 本地模型加载失败: {e}")

        # 路径2: HuggingFace缓存 (离线可用)
        if not loaded:
            hf_cache = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
            if os.path.exists(hf_cache):
                print(f"\n[UTMOS._SSLEncoder] 使用HuggingFace缓存模型")
                print(f"  路径: {hf_cache}")
                try:
                    self.processor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
                    self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
                    loaded = True
                    print(f"  ✓ HF缓存模型加载完成")
                except Exception as e:
                    print(f"  ⚠️ HF缓存加载失败: {e}")

        # 路径3: 报错（不再尝试网络下载）
        if not loaded:
            raise RuntimeError(
                f"无法离线加载 wav2vec2 模型 '{model_name}'。\n"
                f"  请确保以下路径之一存在:\n"
                f"  - 项目路径: {local_model_path}\n"
                f"  - HF缓存:   {os.path.expanduser('~/.cache/huggingface/hub/')}\n"
                f"  离线部署前请预先下载模型文件。"
            )
        
        total_time = time.time() - load_start
        print(f"\n[UTMOS._SSLEncoder] SSL编码器初始化完成 (总耗时: {total_time:.2f}s)")
        print(f"  模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        
        if freeze:
            print(f"  冻结模型参数")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        x = self.processor(
            [t.cpu().numpy() for t in x],
            sampling_rate=self.sr,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(**x, output_hidden_states=True)  # type: ignore
        return outputs.hidden_states


class SSLExtModel(nn.Module):
    """
    A self-supervised learning (SSL) model extended with data-domain id.
    This model uses an encoder to process input data, applies attention layers if configured,
    and combines the features with data-domain embeddings before classification.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing model and dataset settings.
        name (str | None):
            Optional name for the SSL encoder. Defaults to the name specified in `cfg.model.ssl.name`.
    """

    def __init__(self, cfg: Config, name: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = _SSLEncoder(
            cfg.sr, name or cfg.model.ssl.name, cfg.model.ssl.freeze
        )
        hidden_num, in_features = get_ssl_output_shape(name or cfg.model.ssl.name)
        self.weights = nn.Parameter(F.softmax(torch.randn(hidden_num), dim=0))
        if cfg.model.ssl.attn:
            self.attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=in_features,
                        num_heads=8,
                        dropout=0.2,
                        batch_first=True,
                    )
                    for _ in range(cfg.model.ssl.attn)
                ]
            )
        self.num_dataset = get_dataset_num(cfg)
        self.fc: nn.Linear | nn.Identity = nn.Linear(
            in_features * 2 + self.num_dataset, cfg.model.ssl.num_classes
        )

    def forward(self, xt: tuple[torch.Tensor], d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SSLExtModel.

        Args:
            x (torch.Tensor):
                Input tensor representing the features to be processed by the SSL encoder.
            d (torch.Tensor):
                Dataset-specific information tensor.

        Returns:
            torch.Tensor:
                Output tensor after applying the SSL encoder, attention (if configured), and fully connected layers.
        """
        xt = self.encoder(xt)
        x: torch.Tensor = sum([t * w for t, w in zip(xt, self.weights)])
        if self.cfg.model.ssl.attn:
            y = x
            for attn in self.attn:
                y, _ = attn(y, y, y)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=1)[0]], dim=1)
        else:
            x = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        return x


def get_ssl_output_shape(name: str) -> tuple[int, int]:
    if name in [
        "facebook/w2v-bert-2.0",
        "facebook/wav2vec2-large",
        "facebook/wav2vec2-large-robust",
        "facebook/wav2vec2-large-960h",
        "microsoft/wavlm-large",
        "facebook/wav2vec2-large-xlsr-53",
    ]:
        return 25, 1024
    elif name in [
        "facebook/hubert-base-ls960",
        "facebook/data2vec-audio-base-960h",
        "microsoft/wavlm-base",
        "microsoft/wavlm-base-plus",
        "microsoft/wavlm-base-plus-sv",
        "facebook/wav2vec2-base",
    ]:
        return 13, 768
    else:
        raise NotImplementedError
