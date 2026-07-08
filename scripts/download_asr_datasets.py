#!/usr/bin/env python3
"""
ASR Benchmark 数据集下载脚本
支持 AISHELL-1 / THCHS-30 标准测试数据集下载

用法:
    python download_asr_datasets.py                        # 下载所有
    python download_asr_datasets.py --dataset aishell1     # 仅 AISHELL-1
    python download_asr_datasets.py --dataset thchs30      # 仅 THCHS-30
"""

import argparse
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("download")

PROJECT_ROOT = Path(__file__).parent.resolve()
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """下载文件，支持断点续传"""
    import urllib.request
    from urllib.error import URLError

    logger.info(f"  ⬇️  正在下载 {desc}...")
    logger.info(f"      源: {url}")
    logger.info(f"      目标: {dest}")

    try:
        def report(block, blocksize, totalsize):
            downloaded = block * blocksize / 1024 / 1024
            total_mb = totalsize / 1024 / 1024 if totalsize > 0 else 0
            if totalsize > 0:
                pct = min(100, block * blocksize * 100 / totalsize)
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                sys.stdout.write(f"\r      [{bar}] {downloaded:.0f}/{total_mb:.0f} MB ({pct:.0f}%)")
            else:
                sys.stdout.write(f"\r      Downloaded: {downloaded:.0f} MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, report)
        sys.stdout.write("\n")
        logger.info(f"  ✅ {desc} 下载完成 ({dest.stat().st_size / 1024**3:.1f} GB)")
        return True
    except URLError as e:
        logger.error(f"  ❌ 下载失败: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning("\n  ⚠️  下载被中断")
        if dest.exists():
            dest.unlink()
        return False


def extract_aishell1_test(archive_path: Path, output_dir: Path):
    """从AISHELL-1完整压缩包中只提取test子集+transcript"""
    logger.info(f"  解压中 (仅test子集)...")

    test_wav_dir = output_dir / "test" / "wav"
    transcript_dir = output_dir / "test" / "transcript"
    test_wav_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    test_wavs = []
    all_transcripts = {}

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            # 提取test子集的WAV
            if m.name.startswith("data_aishell/wav/test/") and m.name.endswith(".wav"):
                m.name = os.path.relpath(m.name, "data_aishell/wav/test")
                tar.extract(m, test_wav_dir)
                test_wavs.append(os.path.basename(m.name))

            # 提取完整transcript
            if m.name == "data_aishell/transcript/aishell_transcript_v0.8.txt":
                f = tar.extractfile(m)
                if f:
                    content = f.read().decode("utf-8")
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            all_transcripts[parts[0]] = " ".join(parts[1:])

    # 过滤出test子集的transcript并写入适配器期望的格式
    test_transcript_path = transcript_dir / "aishell1_test.txt"
    with open(test_transcript_path, "w", encoding="utf-8") as f:
        for wav_file in sorted(test_wavs):
            utt_id = Path(wav_file).stem
            text = all_transcripts.get(utt_id, "")
            if text:
                f.write(f"{utt_id} {text}\n")

    logger.info(f"  ✅ 提取完成: {len(test_wavs)} 条测试音频")
    logger.info(f"     目录: {output_dir / 'test'}")


def download_aishell1():
    """下载 AISHELL-1 测试集"""
    output_dir = DATASETS_DIR / "aishell1"
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.openslr.org/resources/33/data_aishell.tgz"
    archive_path = DATASETS_DIR / "data_aishell.tgz"

    if archive_path.exists():
        logger.info(f"  📦 压缩包已存在，跳过下载")
    else:
        if not download_file(url, archive_path, "AISHELL-1 (完整, ~15GB)"):
            return False

    if not (output_dir / "test" / "wav").exists():
        extract_aishell1_test(archive_path, output_dir)
    else:
        logger.info(f"  📁 测试集已存在，跳过解压"
                    f" ({len(list((output_dir/'test'/'wav').rglob('*.wav')))} 条)")

    # 可选：删除压缩包
    if archive_path.exists():
        logger.info(f"  清理: 删除压缩包以释放空间 (~15GB)")
        archive_path.unlink()
        logger.info(f"  ✅ 压缩包已删除")

    return True


def download_thchs30():
    """下载 THCHS-30 测试集"""
    output_dir = DATASETS_DIR / "thchs30"
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.openslr.org/resources/18/data_thchs30.tgz"
    archive_path = DATASETS_DIR / "data_thchs30.tgz"

    if archive_path.exists():
        logger.info(f"  📦 压缩包已存在，跳过下载")
    else:
        if not download_file(url, archive_path, "THCHS-30 (~6.5GB)"):
            return False

    if not (output_dir / "test").exists():
        logger.info(f"  解压中...")
        with tarfile.open(archive_path, "r:gz") as tar:
            # 只提取test子集
            members = [m for m in tar.getmembers()
                       if m.name.startswith("data_thchs30/test/")]
            for m in members:
                m.name = os.path.relpath(m.name, "data_thchs30")
                tar.extract(m, output_dir)
        logger.info(f"  ✅ THCHS-30 test 提取完成"
                    f" ({len(list((output_dir/'test').glob('*.wav')))} 条)")
    else:
        logger.info(f"  📁 测试集已存在，跳过解压")
        archive_path.unlink(missing_ok=True)

    return True


def main():
    parser = argparse.ArgumentParser(description="下载ASR Benchmark数据集")
    parser.add_argument("--dataset", choices=["aishell1", "thchs30", "all"],
                        default="all", help="数据集名称")
    parser.add_argument("--keep-archive", action="store_true",
                        help="保留压缩包（不删除）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ASR Benchmark 数据集下载")
    logger.info("=" * 60)

    if args.keep_archive:
        logger.info(f"  压缩包保留策略: 保留")

    datasets = ["aishell1", "thchs30"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        logger.info(f"\n📊 数据集: {ds}")
        logger.info("-" * 40)

        if not DATASETS_DIR.exists():
            DATASETS_DIR.mkdir(parents=True)

        try:
            if ds == "aishell1":
                download_aishell1()
            elif ds == "thchs30":
                download_thchs30()
        except Exception as e:
            logger.error(f"  ❌ {ds} 下载失败: {e}")
            import traceback
            traceback.print_exc()

    # 最终目录结构
    logger.info("\n" + "=" * 60)
    logger.info("最终数据集目录结构:")
    logger.info("=" * 60)
    for ds_dir in sorted(DATASETS_DIR.iterdir()):
        if ds_dir.is_dir() and ds_dir.name != "asr_builtin":
            wav_count = len(list(ds_dir.rglob("*.wav")))
            logger.info(f"  📁 {ds_dir.name}/  ({wav_count} 条音频)")


if __name__ == "__main__":
    main()
