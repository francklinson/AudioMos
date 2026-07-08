#!/usr/bin/env python3
"""
ASR 模型下载脚本
将模型下载到项目本地 models/asr/ 目录，而非默认缓存路径。

用法:
    python download_asr_models.py                  # 下载全部9个模型
    python download_asr_models.py --model paraformer-large  # 仅下载指定模型
    python download_asr_models.py --list           # 查看已下载的模型

支持的模型:
    paraformer-large       - Paraformer-large (HuggingFace: FunASR/paraformer-zh)
    sensevoice-small       - SenseVoice-Small  (HuggingFace: FunAudioLLM/SenseVoiceSmall)
    wenet-u2pp             - WeNet U2++        (本地复制: models/wenet/)
    whisper-large-v3-turbo - Whisper large-v3-turbo (HuggingFace: openai/whisper-large-v3-turbo)
    firered-asr2           - FireRedASR2-AED   (HuggingFace: FireRedTeam/FireRedASR2-AED)
    qwen3-asr              - Qwen3-ASR-1.7B    (HuggingFace: Qwen/Qwen3-ASR-1.7B)
    funasr-llm             - Fun-ASR-Nano 800M (HuggingFace: FunAudioLLM/Fun-ASR-Nano-2512)
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ASR_MODELS_DIR = PROJECT_ROOT / "models" / "asr"

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_asr_models")

# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------
MODEL_DEFS = {
    "paraformer-large": {
        "display_name": "Paraformer-Large",
        "source": "huggingface",
        "repo_id": "FunASR/paraformer-zh",
        "local_dir": ASR_MODELS_DIR / "paraformer-large",
        "marker_files": ["model.pt", "configuration.json"],
    },
    "sensevoice-small": {
        "display_name": "SenseVoice-Small",
        "source": "huggingface",
        "repo_id": "FunAudioLLM/SenseVoiceSmall",
        "local_dir": ASR_MODELS_DIR / "sensevoice-small",
        "marker_files": ["model.pt", "configuration.json"],
    },
    "wenet-u2pp": {
        "display_name": "WeNet U2++",
        "source": "local_copy",
        "src_dir": PROJECT_ROOT / "models" / "wenet",
        "local_dir": ASR_MODELS_DIR / "wenet-u2pp",
        "marker_files": ["final.pt", "train.yaml"],
    },
    "whisper-large-v3-turbo": {
        "display_name": "Whisper Large-v3 Turbo",
        "source": "huggingface",
        "repo_id": "openai/whisper-large-v3-turbo",
        "local_dir": ASR_MODELS_DIR / "whisper-large-v3-turbo",
        "marker_files": ["model.safetensors", "config.json"],
    },
    "firered-asr2": {
        "display_name": "FireRedASR2-AED (1.1B)",
        "source": "huggingface",
        "repo_id": "FireRedTeam/FireRedASR2-AED",
        "local_dir": ASR_MODELS_DIR / "firered-asr2",
        "marker_files": ["config.yaml", "model.pth.tar"],
    },
    "qwen3-asr": {
        "display_name": "Qwen3-ASR-1.7B",
        "source": "huggingface",
        "repo_id": "Qwen/Qwen3-ASR-1.7B",
        "local_dir": ASR_MODELS_DIR / "qwen3-asr",
        "marker_files": ["config.json", "model.safetensors"],
    },
    "funasr-llm": {
        "display_name": "Fun-ASR-Nano (800M)",
        "source": "huggingface",
        "repo_id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "local_dir": ASR_MODELS_DIR / "funasr-llm",
        "marker_files": ["model.pt", "configuration.json"],
    },
}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def is_model_downloaded(model_name: str) -> bool:
    """检查模型是否已下载到本地目录（通过标记文件判断）"""
    model_def = MODEL_DEFS[model_name]
    local_dir = model_def["local_dir"]
    if not local_dir.exists():
        return False
    # 至少存在一个标记文件即视为已下载
    marker = model_def.get("marker_files", [])
    if not marker:
        return any(local_dir.iterdir())
    return any((local_dir / f).exists() for f in marker)


def list_models():
    """列出所有模型及其下载状态"""
    print(f"\n{'模型名称':<28} {'显示名称':<25} {'状态':<10} {'本地路径'}")
    print("-" * 90)
    for name, defn in MODEL_DEFS.items():
        downloaded = is_model_downloaded(name)
        status = "✅ 已下载" if downloaded else "⬜ 未下载"
        print(f"{name:<28} {defn['display_name']:<25} {status:<10} {defn['local_dir']}")
    print()


# ---------------------------------------------------------------------------
# 下载进度回调
# ---------------------------------------------------------------------------

def _tqdm_progress_hook(t):
    """返回一个 urlretrieve 兼容的进度回调"""
    last_b = [0]

    def update(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update


# ---------------------------------------------------------------------------
# ModelScope 下载 (Paraformer / SenseVoice)
# ---------------------------------------------------------------------------

def download_modelscope_model(model_name: str):
    """通过 modelscope.hub.snapshot_download 下载到指定本地目录"""
    model_def = MODEL_DEFS[model_name]
    model_id = model_def["model_id"]
    local_dir = str(model_def["local_dir"])

    logger.info("[%s] 开始从 ModelScope 下载: %s → %s", model_def["display_name"], model_id, local_dir)

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        logger.error("[%s] modelscope 未安装，请执行: pip install modelscope", model_def["display_name"])
        return False

    try:
        local_dir = snapshot_download(
            model_id=model_id,
            local_dir=local_dir,
        )
        logger.info("[%s] 下载完成: %s", model_def["display_name"], local_dir)
        return True
    except Exception as e:
        logger.error("[%s] 下载失败: %s", model_def["display_name"], e)
        return False


# ---------------------------------------------------------------------------
# WeNet Hub 下载
# ---------------------------------------------------------------------------

def download_wenet_model(model_name: str):
    """
    从 wenet ModelScope 数据集下载模型到本地目录。
    复用 wenet Hub 的下载逻辑，但将文件保存到项目本地目录。
    """
    import requests as req_lib

    model_def = MODEL_DEFS[model_name]
    dataset_model = model_def["dataset_model"]
    local_dir = model_def["local_dir"]
    tar_name = f"{dataset_model}.tar.gz"

    logger.info("[%s] 开始从 WeNet Hub 下载: %s → %s", model_def["display_name"], tar_name, local_dir)

    # 1. 查询 ModelScope 数据集 API 获取下载链接
    api_url = "https://modelscope.cn/api/v1/datasets/wenet/wenet_pretrained_models/oss/tree"
    try:
        resp = req_lib.get(api_url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("Data", [])
        model_info = next((d for d in data if d["Key"] == tar_name), None)
        if model_info is None:
            logger.error("[%s] 在 WeNet Hub 中未找到模型文件: %s", model_def["display_name"], tar_name)
            # 尝试回退到 wenetspeech_u2pp
            fallback = "wenetspeech_u2pp_conformer_exp.tar.gz"
            model_info = next((d for d in data if d["Key"] == fallback), None)
            if model_info:
                logger.info("[%s] 回退使用: %s", model_def["display_name"], fallback)
            else:
                return False
        model_url = model_info["Url"]
    except Exception as e:
        logger.error("[%s] 查询 WeNet Hub API 失败: %s", model_def["display_name"], e)
        return False

    # 2. 下载 tar.gz 到本地目录
    os.makedirs(local_dir, exist_ok=True)
    tar_path = local_dir / tar_name

    try:
        from tqdm import tqdm

        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=tar_name) as t:
            urlretrieve(model_url, filename=str(tar_path), reporthook=_tqdm_progress_hook(t))
            t.total = t.n
    except ImportError:
        logger.info("[%s] 下载中... (安装 tqdm 可显示进度条)", model_def["display_name"])
        urlretrieve(model_url, filename=str(tar_path))

    # 3. 解压到同一目录
    logger.info("[%s] 解压模型文件...", model_def["display_name"])
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(str(tar_path), "r") as tar:
                tar.extractall(path=temp_dir)
            # 解压后通常有一个子目录，把内容移到 local_dir
            contents = os.listdir(temp_dir)
            if len(contents) == 1 and os.path.isdir(os.path.join(temp_dir, contents[0])):
                extracted_dir = os.path.join(temp_dir, contents[0])
            else:
                extracted_dir = temp_dir
            for item in os.listdir(extracted_dir):
                src = os.path.join(extracted_dir, item)
                dst = local_dir / item
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.move(src, local_dir)
        # 删除 tar.gz
        tar_path.unlink(missing_ok=True)
        logger.info("[%s] 下载并解压完成: %s", model_def["display_name"], local_dir)
        return True
    except Exception as e:
        logger.error("[%s] 解压失败: %s", model_def["display_name"], e)
        # 清理残留
        if tar_path.exists():
            tar_path.unlink()
        return False


# ---------------------------------------------------------------------------
# HuggingFace 下载 (Whisper)
# ---------------------------------------------------------------------------

def download_huggingface_model(model_name: str):
    """通过 huggingface_hub.snapshot_download 下载到指定本地目录"""
    model_def = MODEL_DEFS[model_name]
    repo_id = model_def["repo_id"]
    local_dir = str(model_def["local_dir"])

    logger.info("[%s] 开始从 HuggingFace 下载: %s → %s", model_def["display_name"], repo_id, local_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("[%s] huggingface_hub 未安装，请执行: pip install huggingface-hub", model_def["display_name"])
        return False

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
        )
        logger.info("[%s] 下载完成: %s", model_def["display_name"], local_dir)
        return True
    except Exception as e:
        logger.error("[%s] 下载失败: %s", model_def["display_name"], e)
        return False


# ---------------------------------------------------------------------------
# 本地复制 (WeNet — 从项目已有 models/wenet/ 复制)
# ---------------------------------------------------------------------------

def download_local_copy_model(model_name: str):
    """从项目已有目录复制模型到ASR模型目录"""
    import shutil

    model_def = MODEL_DEFS[model_name]
    src_dir = model_def.get("src_dir")
    local_dir = model_def["local_dir"]

    if not src_dir or not src_dir.exists():
        logger.error("[%s] 源目录不存在: %s", model_def["display_name"], src_dir)
        return False

    logger.info("[%s] 从本地复制模型: %s → %s", model_def["display_name"], src_dir, local_dir)
    os.makedirs(local_dir, exist_ok=True)

    try:
        for item in os.listdir(src_dir):
            src = src_dir / item
            dst = local_dir / item
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        logger.info("[%s] 复制完成: %s", model_def["display_name"], local_dir)
        return True
    except Exception as e:
        logger.error("[%s] 复制失败: %s", model_def["display_name"], e)
        return False


# ---------------------------------------------------------------------------
# ModelScope 下载 (FunASR-LLM 等阿里系模型)
# ---------------------------------------------------------------------------

def download_modelscope_model(model_name: str):
    """通过 modelscope.snapshot_download 下载到指定本地目录"""
    model_def = MODEL_DEFS[model_name]
    repo_id = model_def["repo_id"]
    local_dir = str(model_def["local_dir"])

    logger.info("[%s] 开始从 ModelScope 下载: %s → %s", model_def["display_name"], repo_id, local_dir)

    try:
        from modelscope import snapshot_download
    except ImportError:
        logger.error("[%s] modelscope 未安装，请执行: pip install modelscope", model_def["display_name"])
        return False

    try:
        snapshot_download(
            model_id=repo_id,
            local_dir=local_dir,
        )
        logger.info("[%s] 下载完成: %s", model_def["display_name"], local_dir)
        return True
    except Exception as e:
        logger.error("[%s] 下载失败: %s", model_def["display_name"], e)
        return False


# ---------------------------------------------------------------------------
# 下载调度
# ---------------------------------------------------------------------------

DOWNLOAD_HANDLERS = {
    "huggingface": download_huggingface_model,
    "local_copy": download_local_copy_model,
    "modelscope": download_modelscope_model,
}


def download_model(model_name: str, force: bool = False):
    """下载单个模型"""
    if model_name not in MODEL_DEFS:
        logger.error("未知模型: %s", model_name)
        logger.info("可用模型: %s", ", ".join(MODEL_DEFS.keys()))
        return False

    model_def = MODEL_DEFS[model_name]

    # 检查是否已下载
    if not force and is_model_downloaded(model_name):
        logger.info("[%s] 已存在，跳过下载。使用 --force 可强制重新下载。", model_def["display_name"])
        return True

    # 创建目录
    os.makedirs(model_def["local_dir"], exist_ok=True)

    # 调用对应的下载处理器
    source = model_def["source"]
    handler = DOWNLOAD_HANDLERS.get(source)
    if handler is None:
        logger.error("[%s] 未知的下载源: %s", model_def["display_name"], source)
        return False

    return handler(model_name)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASR 模型下载工具 — 下载到项目本地 models/asr/ 目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_asr_models.py                        下载全部模型
  python download_asr_models.py --model paraformer-large   仅下载 Paraformer
  python download_asr_models.py --model wenet-u2pp sensevoice-small  下载多个
  python download_asr_models.py --list                 查看已下载的模型
  python download_asr_models.py --force --model paraformer-large  强制重新下载

可用模型:
  paraformer-large, sensevoice-small, wenet-u2pp, whisper-large-v3-turbo
""",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(MODEL_DEFS.keys()),
        help="指定要下载的模型（可多选），不指定则下载全部",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有模型及其下载状态",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使已存在）",
    )

    args = parser.parse_args()

    # 列出模型
    if args.list:
        list_models()
        return

    # 确定要下载的模型列表
    models_to_download = args.model if args.model else list(MODEL_DEFS.keys())

    print(f"\n{'='*60}")
    print(f"  ASR 模型下载工具")
    print(f"  目标目录: {ASR_MODELS_DIR}")
    print(f"  待下载: {', '.join(models_to_download)}")
    print(f"{'='*60}\n")

    # 确保基础目录存在
    os.makedirs(ASR_MODELS_DIR, exist_ok=True)

    # 逐个下载
    results = {}
    for model_name in models_to_download:
        ok = download_model(model_name, force=args.force)
        results[model_name] = ok

    # 汇总
    print(f"\n{'='*60}")
    print("  下载结果汇总")
    print(f"{'='*60}")
    success_count = 0
    for model_name, ok in results.items():
        display = MODEL_DEFS[model_name]["display_name"]
        status = "✅ 成功" if ok else "❌ 失败"
        print(f"  {display:<25} {status}")
        if ok:
            success_count += 1
    print(f"\n  成功: {success_count}/{len(results)}")
    print(f"{'='*60}\n")

    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
