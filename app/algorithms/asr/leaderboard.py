"""
ASR 测评榜单管理模块
持久化合并公开基准 + 本地测评最佳结果，提供排名查询
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .baselines import BASELINES, DATASET_DISPLAY_NAMES, get_baseline
from .registry import ASR_ALGORITHM_DESCRIPTIONS

logger = logging.getLogger("audiomos")

# 榜单中支持的数据集（与 baselines.py 中一致）
LEADERBOARD_DATASETS = ["aishell1_test", "wenetspeech_test", "wenetspeech_test_meeting", "thchs30_test", "builtin"]

# 不在榜单中展示的算法（无公开基准数据且无本地测评结果）
LEADERBOARD_EXCLUDED_ALGOS = {"step-audio-2-mini", "vibevoice-asr"}


def _make_entry(algorithm: str) -> dict:
    """为指定算法创建空榜单条目（含元信息）"""
    desc = ASR_ALGORITHM_DESCRIPTIONS.get(algorithm, {})
    return {
        "algorithm": algorithm,
        "display_name": desc.get("display_name", algorithm),
        "params": desc.get("params", ""),
        "baseline_cer": None,
        "baseline_source": None,
        "local_cer": None,
        "local_wer": None,
        "local_rtf": None,
        "local_num_utterances": None,
        "local_bench_id": None,
        "local_updated_at": None,
    }


class LeaderboardManager:
    """榜单管理器 — 持久化 JSON 文件，内存缓存"""

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._data: Dict[str, Any] = {"updated_at": None, "datasets": {}}
        self._loaded = False

    # ── 文件读写 ──

    def _ensure_loaded(self):
        if self._loaded:
            return
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"[Leaderboard] 从文件加载: {self._filepath}")
            except Exception as e:
                logger.error(f"[Leaderboard] 加载失败，将重建: {e}")
                self._seed()
        else:
            logger.info("[Leaderboard] 文件不存在，从公开基准初始化")
            self._seed()
        self._loaded = True

    def save(self):
        self._data["updated_at"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        logger.info(f"[Leaderboard] 已保存: {self._filepath}")

    # ── 初始化 ──

    def _seed(self):
        """从 baselines.py 初始化榜单（仅公开基准，无本地结果）"""
        self._data = {"updated_at": datetime.now().isoformat(), "datasets": {}}

        for ds_key in LEADERBOARD_DATASETS:
            entries = []
            # 收集所有在 BASELINES 中出现的算法
            algorithms_seen = set()
            for algo in BASELINES:
                algorithms_seen.add(algo)
            # 也包含仅在 registry 中的算法（未在 baselines.py 中注册的）
            for algo in ASR_ALGORITHM_DESCRIPTIONS:
                algorithms_seen.add(algo)

            for algo in sorted(algorithms_seen):
                if algo in LEADERBOARD_EXCLUDED_ALGOS:
                    continue
                entry = _make_entry(algo)
                baseline = get_baseline(algo, ds_key)
                if baseline and baseline.get("cer") is not None:
                    entry["baseline_cer"] = baseline["cer"]
                    entry["baseline_source"] = baseline.get("source", "")
                entries.append(entry)

            self._data["datasets"][ds_key] = {
                "name": DATASET_DISPLAY_NAMES.get(ds_key, ds_key),
                "description": "",
                "entries": entries,
            }

        self.save()

    def seed_from_history(self, results_dir: str):
        """扫描 results/{dataset}/*.json，提取各算法在各数据集上的最佳 CER"""
        self._ensure_loaded()
        if not os.path.isdir(results_dir):
            logger.info(f"[Leaderboard] results 目录不存在，跳过历史扫描: {results_dir}")
            return

        updated = 0
        for ds_name in os.listdir(results_dir):
            ds_dir = os.path.join(results_dir, ds_name)
            if not os.path.isdir(ds_dir):
                continue
            for fname in os.listdir(ds_dir):
                if not fname.endswith(".json"):
                    continue
                filepath = os.path.join(ds_dir, fname)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                dataset = data.get("dataset", ds_name)
                algorithm = data.get("algorithm", fname[:-5])
                metrics = data.get("metrics", {})
                cer = metrics.get("cer")
                if cer is None:
                    continue
                cer_pct = round(cer * 100, 2)
                wer = metrics.get("wer")
                wer_pct = round(wer * 100, 2) if wer is not None else None
                rtf = metrics.get("rtf")
                num_utt = metrics.get("num_utterances", 0)
                bench_updated = data.get("updated_at", "")

                if self._update_local(dataset, algorithm, cer_pct, wer_pct, rtf, num_utt, data.get("bench_id", ""), bench_updated):
                    updated += 1

        if updated:
            self.save()
            logger.info(f"[Leaderboard] 从历史记录更新 {updated} 条本地最佳结果")

    # ── 查询 ──

    def get_full(self) -> dict:
        """返回完整榜单数据"""
        self._ensure_loaded()
        return self._data

    def get_leaderboard(self, dataset_key: str) -> dict:
        """返回指定数据集的排序后榜单"""
        self._ensure_loaded()
        ds = self._data.get("datasets", {}).get(dataset_key)
        if ds is None:
            # 动态创建空数据集
            ds = {"name": DATASET_DISPLAY_NAMES.get(dataset_key, dataset_key), "description": "", "entries": []}

        # 排序: 按 local_cer 或 baseline_cer 升序，无数据的排末尾
        def sort_key(entry):
            cer = entry.get("local_cer")
            if cer is not None:
                return cer
            cer = entry.get("baseline_cer")
            if cer is not None:
                return cer
            return float("inf")

        ds["entries"] = sorted(ds.get("entries", []), key=sort_key)
        return ds

    # ── 更新 ──

    def _update_local(self, dataset: str, algorithm: str, cer_pct: float, wer_pct: Optional[float], rtf: Optional[float], num_utt: int, bench_id: str, updated_at: str) -> bool:
        """更新单条本地结果（仅当 CER 优于当前值时），返回是否更新"""
        if not dataset or algorithm in LEADERBOARD_EXCLUDED_ALGOS:
            return False

        # 确保数据集存在
        if dataset not in self._data["datasets"]:
            self._data["datasets"][dataset] = {
                "name": DATASET_DISPLAY_NAMES.get(dataset, dataset),
                "description": "",
                "entries": [],
            }

        entries = self._data["datasets"][dataset]["entries"]
        entry = next((e for e in entries if e["algorithm"] == algorithm), None)

        if entry is None:
            entry = _make_entry(algorithm)
            entries.append(entry)

        current = entry.get("local_cer")
        if current is not None and cer_pct >= current:
            return False  # 不是更好的结果

        entry["local_cer"] = cer_pct
        entry["local_wer"] = wer_pct
        entry["local_rtf"] = round(rtf, 4) if rtf is not None else None
        entry["local_num_utterances"] = num_utt
        entry["local_bench_id"] = bench_id
        entry["local_updated_at"] = updated_at
        return True

    def update_from_benchmark(self, bench_id: str, bench_data: dict) -> int:
        """从完成的 benchmark 更新榜单，返回更新条数"""
        self._ensure_loaded()

        if bench_data.get("status") != "completed":
            return 0

        dataset = bench_data.get("dataset", bench_data.get("dataset_name", ""))
        results = bench_data.get("results", {})
        bench_updated = bench_data.get("updated_at", bench_data.get("created_at", ""))

        updated = 0
        for algo_name, algo_result in results.items():
            metrics = algo_result.get("metrics", {})
            cer = metrics.get("cer")
            if cer is None:
                continue
            cer_pct = round(cer * 100, 2)
            wer = metrics.get("wer")
            wer_pct = round(wer * 100, 2) if wer is not None else None
            rtf = metrics.get("rtf")
            num_utt = metrics.get("num_utterances", 0)

            if self._update_local(dataset, algo_name, cer_pct, wer_pct, rtf, num_utt, bench_id, bench_updated):
                updated += 1
                logger.info(f"[Leaderboard] 更新: {dataset}/{algo_name} CER={cer_pct}%")

        if updated:
            self.save()

        return updated

    def refresh(self, results_dir: str) -> int:
        """完全重建榜单"""
        self._seed()
        self.seed_from_history(results_dir)
        self.save()
        return len(self._data.get("datasets", {}))
