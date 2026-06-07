#!/usr/bin/env python3
"""
ClearVoice 模型批量下载脚本
将所有预训练模型从 HuggingFace 下载到项目本地 ./models/clearvoice/ 目录

模型列表:
  1. FRCRN_SE_16K         (~200MB) - 16kHz 实时语音增强
  2. MossFormerGAN_SE_16K (~600MB) - 16kHz GAN 语音增强
  3. MossFormer2_SS_16K   (~800MB) - 16kHz 语音分离
  4. MossFormer2_SE_48K   (~1.2GB) - 48kHz 语音增强
  5. MossFormer2_SR_48K   (~1.5GB) - 48kHz 超分辨率

总计约 4.3GB
"""

import os
import sys

# 确保项目根目录在 path 中
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, "app", "algorithms"))

from huggingface_hub import snapshot_download

MODELS = [
    {
        "repo_id": "alibabasglab/FRCRN_SE_16K",
        "local_dir": "models/clearvoice/FRCRN_SE_16K",
        "name": "FRCRN_SE_16K",
        "desc": "16kHz 实时语音增强 (轻量)",
    },
    {
        "repo_id": "alibabasglab/MossFormerGAN_SE_16K",
        "local_dir": "models/clearvoice/MossFormerGAN_SE_16K",
        "name": "MossFormerGAN_SE_16K",
        "desc": "16kHz GAN 语音增强 (SOTA)",
    },
    {
        "repo_id": "alibabasglab/MossFormer2_SS_16K",
        "local_dir": "models/clearvoice/MossFormer2_SS_16K",
        "name": "MossFormer2_SS_16K",
        "desc": "16kHz 语音分离",
    },
    {
        "repo_id": "alibabasglab/MossFormer2_SE_48K",
        "local_dir": "models/clearvoice/MossFormer2_SE_48K",
        "name": "MossFormer2_SE_48K",
        "desc": "48kHz 语音增强 (最高质量)",
    },
    {
        "repo_id": "alibabasglab/MossFormer2_SR_48K",
        "local_dir": "models/clearvoice/MossFormer2_SR_48K",
        "name": "MossFormer2_SR_48K",
        "desc": "48kHz 超分辨率",
    },
]


def download_model(model_info):
    """下载单个模型"""
    name = model_info["name"]
    desc = model_info["desc"]
    local_dir = model_info["local_dir"]
    repo_id = model_info["repo_id"]

    # 检查是否已下载
    best_path = os.path.join(local_dir, "last_best_checkpoint")
    if os.path.isfile(best_path):
        # 检查 checkpoint 文件是否实际存在
        with open(best_path, "r") as f:
            ckpt_name = f.readline().strip()
        ckpt_path = os.path.join(local_dir, ckpt_name)
        if os.path.isfile(ckpt_path):
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            print(f"  ✓ {name} 已下载 ({size_mb:.0f}MB)，跳过")
            return True

    print(f"  ⬇ 下载 {name}: {desc}")
    print(f"    源: {repo_id}")
    print(f"    目标: {local_dir}")

    try:
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ {name} 下载完成")
        return True
    except Exception as e:
        print(f"  ✗ {name} 下载失败: {e}")
        return False


def main():
    print("=" * 60)
    print("ClearVoice 模型批量下载")
    print(f"项目根目录: {project_root}")
    print(f"目标目录: {os.path.join(project_root, "models", "clearvoice")}")
    print("=" * 60)
    print()

    success_count = 0
    fail_count = 0

    for i, model in enumerate(MODELS, 1):
        print(f"[{i}/{len(MODELS)}] {model['name']}")
        if download_model(model):
            success_count += 1
        else:
            fail_count += 1
        print()

    print("=" * 60)
    print(f"下载完成: {success_count} 成功, {fail_count} 失败")
    print("=" * 60)

    # 显示磁盘使用
    import subprocess
    checkpoint_dir = os.path.join(project_root, "models", "clearvoice")
    if os.path.isdir(checkpoint_dir):
        result = subprocess.run(
            ["du", "-sh", checkpoint_dir],
            capture_output=True, text=True
        )
        print(f"models/clearvoice/ 总大小: {result.stdout.split()[0]}")
        print()
        result = subprocess.run(
            ["du", "-sh"] + [
                os.path.join(checkpoint_dir, d)
                for d in sorted(os.listdir(checkpoint_dir))
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ],
            capture_output=True, text=True
        )
        print(result.stdout)


if __name__ == "__main__":
    main()
