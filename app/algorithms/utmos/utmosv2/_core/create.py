from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from utmosv2._core.model import UTMOSv2Model
from utmosv2._settings import configure_execution
from utmosv2.utils._constants import _UTMOSV2_CHACHE
from utmosv2.utils._download import download_pretrained_weights_from_hf

if TYPE_CHECKING:
    from typing import Literal


def get_project_root() -> Path:
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    # 从 create.py 向上回溯到项目根目录
    # app/algorithms/utmos/utmosv2/_core/create.py -> 项目根目录
    return current_file.parent.parent.parent.parent.parent.parent


def create_model(
    pretrained: bool = True,
    config: str = "fusion_stage3",
    fold: int = 0,
    checkpoint_path: Path | str | None = None,
    seed: int = 42,
    device: torch.device | str | Literal["auto"] = "auto",
) -> UTMOSv2Model:
    """
    Create a UTMOSv2 model with the specified configuration and optional pretrained weights.

    Args:
        pretrained (bool):
            If True, loads pretrained weights. Defaults to True.
        config (str):
            The configuration name to load for the model. Defaults to "fusion_stage3".
        fold (int):
            The fold number for the pretrained weights (used for model selection). Defaults to 0.
        checkpoint_path (Path | str | None):
            Path to a specific model checkpoint. If None, the checkpoint downloaded from GitHub is used. Defaults to None.
        seed (int):
            The seed used for model training to select the correct checkpoint. Defaults to 42.

    Returns:
        UTMOSv2Model: The initialized UTMOSv2 model.

    Raises:
        FileNotFoundError: If the specified checkpoint file is not found.

    Notes:
        - The configuration is dynamically loaded from `utmosv2.config`.
        - If `pretrained` is True and `checkpoint_path` is not provided, the function attempts to download pretrained weights from GitHub.
    """
    import time
    print(f"\n[UTMOS.create_model] 开始创建模型")
    print(f"  配置: {config}")
    print(f"  fold: {fold}")
    print(f"  pretrained: {pretrained}")
    
    start_time = time.time()
    
    print(f"\n[UTMOS.create_model] 步骤1/4: 加载配置")
    _cfg = importlib.import_module(f"utmosv2.config.{config}")
    # Avoid issues with pickling `types.ModuleType`,
    # making it easier to use with multiprocessing, DDP, etc.
    cfg = SimpleNamespace(
        **{k: v for k, v in _cfg.__dict__.items() if not k.startswith("__")}
    )
    configure_execution(cfg)
    print(f"  ✓ 配置加载完成")
    print(f"    - 模型名称: {cfg.model.name}")
    print(f"    - 采样率: {cfg.sr} Hz")
    print(f"    - SSL模型: {cfg.model.ssl.name}")
    print(f"    - 批量大小: {cfg.batch_size}")

    print(f"\n[UTMOS.create_model] 步骤2/4: 创建模型架构")
    model = UTMOSv2Model(cfg)
    print(f"  ✓ 模型架构创建完成")
    print(f"    - 模型类型: {type(model._model).__name__}")

    if pretrained:
        print(f"\n[UTMOS.create_model] 步骤3/4: 加载预训练权重")
        if checkpoint_path is None:
            # 优先检查项目目录下的 models/utmos/ 路径
            project_root = get_project_root()
            project_checkpoint_path = (
                project_root
                / "models"
                / "utmos"
                / config
                / f"fold{fold}_s{seed}_best_model.pth"
            )
            
            # 然后检查默认缓存路径
            cache_checkpoint_path = (
                _UTMOSV2_CHACHE
                / "models"
                / config
                / f"fold{fold}_s{seed}_best_model.pth"
            )
            
            print(f"  查找本地检查点...")
            print(f"    项目路径: {project_checkpoint_path}")
            print(f"    缓存路径: {cache_checkpoint_path}")
            
            if project_checkpoint_path.exists():
                checkpoint_path = project_checkpoint_path
                print(f"  ✓ 找到项目路径检查点")
            elif cache_checkpoint_path.exists():
                checkpoint_path = cache_checkpoint_path
                print(f"  ✓ 找到缓存路径检查点")
            else:
                print(f"  ⚠ 本地检查点不存在，尝试下载...")
                download_pretrained_weights_from_hf(config, fold)
                checkpoint_path = cache_checkpoint_path
                
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        print(f"  加载权重到设备: {device}")
        checkpoint_size = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  检查点大小: {checkpoint_size:.1f} MB")
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  ✓ 权重加载完成")

    print(f"\n[UTMOS.create_model] 步骤4/4: 移动模型到设备")
    model = model.to(device)
    model.eval()
    print(f"  ✓ 模型已移动到 {device}")
    
    elapsed = time.time() - start_time
    print(f"\n[UTMOS.create_model] 模型创建完成 (总耗时: {elapsed:.2f}s)")

    return model
