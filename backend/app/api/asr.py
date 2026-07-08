"""
ASR语音识别API路由
提供语音识别、Benchmark评测等功能的RESTful API接口
"""
import os
import sys
import json
import uuid
import time
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.api.auth import get_current_active_user, get_current_user_optional
from app.core.security import User
from app.core.config import settings
from app.core.logging_config import logger
from app.core.task_queue import TaskQueue, Task, TaskStatus
from app.core.websocket import ConnectionManager

import torch

# 导入ASR模块
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'app', 'algorithms'))

try:
    from asr import ASRRegistry, ASR_ALGORITHM_DESCRIPTIONS
    from asr.benchmark import ASRBenchmark
    from asr.dataset_manager import DatasetManager
    from asr.report_generator import ASRReportGenerator
    from asr.evaluator import compute_cer
    from asr.leaderboard import LeaderboardManager

    ASR_AVAILABLE = True
    logger.info("✓ ASR语音识别模块加载成功")
except ImportError as e:
    ASR_AVAILABLE = False
    logger.warning(f"ASR语音识别模块加载失败: {e}")

# ── 全局依赖 ──


def _ensure_asr_available():
    """依赖项：确保ASR模块可用"""
    if not ASR_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR模块不可用",
        )


def _safe_create_task(coro, log_prefix="[ASR]"):
    """创建异步task并确保异常被记录"""
    task = asyncio.ensure_future(coro)
    task.add_done_callback(
        lambda t: logger.error(f"{log_prefix} 异步任务异常: {t.exception()}")
        if t.exception() else None
    )
    return task


router = APIRouter(
    prefix="/asr",
    tags=["ASR语音识别"],
    dependencies=[Depends(_ensure_asr_available)],
)

# 确保目录存在
asr_upload_dir = os.path.join(settings.paths.upload_dir, "asr")
asr_result_dir = os.path.join(settings.paths.result_dir, "asr")
asr_report_dir = os.path.join(settings.paths.result_dir, "asr", "reports")
asr_model_dir = os.path.join(settings.paths.models_dir, "asr")

for d in [asr_upload_dir, asr_result_dir, asr_report_dir, asr_model_dir]:
    os.makedirs(d, exist_ok=True)

# ASR独立任务队列（带JSON持久化）
asr_task_queue = TaskQueue(
    max_workers=settings.asr.benchmark_workers,
    persistence_dir=os.path.join(asr_result_dir, "_tasks"),
)

# WebSocket连接管理器
asr_manager = ConnectionManager()

# ── 测评结果持久化（按 数据集/算法 组织）──
asr_results_dir = os.path.join(asr_result_dir, "results")
os.makedirs(asr_results_dir, exist_ok=True)

# asr_results[dataset][algorithm] = {metrics, per_utterance, errors, ...}
asr_results: Dict[str, Dict[str, dict]] = {}

# Benchmark 运行状态（仅内存，存正在跑的）
asr_benchmarks: Dict[str, Dict[str, Any]] = {}

# 旧格式目录（用于迁移）
_asr_benchmarks_legacy_dir = os.path.join(asr_result_dir, "_benchmarks")


def _migrate_legacy_benchmarks():
    """将旧 _benchmarks/*.json 迁移到 results/{dataset}/{algorithm}.json"""
    if not os.path.isdir(_asr_benchmarks_legacy_dir):
        return
    files = [f for f in os.listdir(_asr_benchmarks_legacy_dir) if f.endswith(".json")]
    if not files:
        return

    logger.info(f"[ASR] 检测到旧格式 benchmark {len(files)} 个，开始迁移...")
    migrated = 0
    for fname in files:
        try:
            with open(os.path.join(_asr_benchmarks_legacy_dir, fname), "r") as f:
                bench = json.load(f)
        except Exception:
            continue

        if bench.get("status") != "completed":
            continue

        dataset = bench.get("dataset", bench.get("dataset_name", ""))
        results = bench.get("results", {})
        if isinstance(results, dict) and "results" in results:
            results = results["results"]  # 兼容旧嵌套格式

        if not isinstance(results, dict):
            continue

        for algo_name, algo_result in results.items():
            if not isinstance(algo_result, dict):
                continue
            _save_result(dataset, algo_name, algo_result)
            migrated += 1

    if migrated:
        # 迁移完成后重命名旧目录
        bak = _asr_benchmarks_legacy_dir + ".bak"
        os.rename(_asr_benchmarks_legacy_dir, bak)
        logger.info(f"[ASR] 迁移完成: {migrated} 条结果 → {asr_results_dir}, 旧数据备份至 {bak}")


def _load_all_results():
    """启动时从 results/{dataset}/*.json 加载所有测评结果"""
    global asr_results
    asr_results = {}
    if not os.path.isdir(asr_results_dir):
        return
    loaded = 0
    for ds_name in os.listdir(asr_results_dir):
        ds_dir = os.path.join(asr_results_dir, ds_name)
        if not os.path.isdir(ds_dir):
            continue
        asr_results[ds_name] = {}
        for fname in os.listdir(ds_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(ds_dir, fname), "r") as f:
                    data = json.load(f)
                algo = data.get("algorithm", fname[:-5])
                asr_results[ds_name][algo] = data
                loaded += 1
            except Exception as e:
                logger.error(f"[ASR] 加载结果失败 {ds_dir}/{fname}: {e}")
    if loaded:
        logger.info(f"[ASR] 加载 {loaded} 条测评结果 ({len(asr_results)} 个数据集)")


def _save_result(dataset: str, algorithm: str, result_data: dict):
    """保存单个算法的测评结果到 results/{dataset}/{algorithm}.json"""
    ds_dir = os.path.join(asr_results_dir, dataset)
    os.makedirs(ds_dir, exist_ok=True)

    # 与已有结果比较，仅保留更优 CER（优先从内存读取）
    existing_path = os.path.join(ds_dir, f"{algorithm}.json")
    existing = asr_results.get(dataset, {}).get(algorithm)
    if existing:
        old_cer = existing.get("metrics", {}).get("cer")
        new_cer = result_data.get("metrics", {}).get("cer")
        if old_cer is not None and new_cer is not None and new_cer >= old_cer:
            return  # 已有更好的结果，跳过
    elif os.path.exists(existing_path):
        try:
            with open(existing_path, "r") as f:
                existing = json.load(f)
            old_cer = existing.get("metrics", {}).get("cer")
            new_cer = result_data.get("metrics", {}).get("cer")
            if old_cer is not None and new_cer is not None and new_cer >= old_cer:
                return  # 已有更好的结果，跳过
        except Exception:
            pass

    data = {
        "algorithm": algorithm,
        "dataset": dataset,
        "metrics": result_data.get("metrics", {}),
        "per_utterance": result_data.get("per_utterance", [])[:settings.asr.per_utterance_limit],
        "errors": result_data.get("errors", []),
        "updated_at": datetime.now().isoformat(),
    }
    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 同步内存
    if dataset not in asr_results:
        asr_results[dataset] = {}
    asr_results[dataset][algorithm] = data


# 模块加载：迁移旧数据 → 加载所有结果
_migrate_legacy_benchmarks()
_load_all_results()


# 数据集管理器（延迟初始化）
_dataset_manager: Optional[DatasetManager] = None

# 进度步骤映射
ASR_PROGRESS_STEPS = {
    0: 'queued',
    10: 'loading',
    30: 'reading',
    50: 'transcribing',
    80: 'saving',
    100: 'done',
}


def _get_step_name(progress: int) -> str:
    """根据进度值推断当前步骤名"""
    step = 'processing'
    for p, s in sorted(ASR_PROGRESS_STEPS.items()):
        if progress >= p:
            step = s
    return step


def _benchmark_cache_key(algorithms: List[str], dataset: str, max_samples: int) -> str:
    """生成benchmark缓存键（算法排序后取指纹）"""
    import hashlib
    raw = f"{'|'.join(sorted(algorithms))}|{dataset}|{max_samples}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _find_cached_benchmark(algorithms: List[str], dataset: str, max_samples: int) -> Optional[Dict[str, Any]]:
    """查找已完成的benchmark结果 — 从 asr_results 字典 O(1) 查找每个算法"""
    ds_results = asr_results.get(dataset, {})
    if not ds_results:
        logger.info(f"[ASR] 缓存未命中: dataset={dataset} 无任何结果")
        return None

    found = {}
    missing = []
    for algo in algorithms:
        if algo in ds_results:
            entry = ds_results[algo]
            found[algo] = {
                "metrics": entry.get("metrics", {}),
                "per_utterance": entry.get("per_utterance", []),
                "errors": entry.get("errors", []),
                "algorithm_name": algo,
            }
        else:
            missing.append(algo)

    if missing:
        if found:
            logger.info(f"[ASR] 部分缓存命中: 已有={set(found.keys())} 缺失={missing}")
        else:
            logger.info(f"[ASR] 缓存未命中: algos={algorithms} dataset={dataset}")
        return None

    logger.info(f"[ASR] 缓存全部命中 ({len(found)}个算法) dataset={dataset}")
    return {
        "bench_id": f"cached_{dataset}",
        "algorithms": algorithms,
        "dataset": dataset,
        "max_samples": max_samples,
        "status": "completed",
        "progress": 100.0,
        "results": found,
        "cached": True,
    }


def _get_dataset_manager() -> DatasetManager:
    """获取数据集管理器（懒加载单例，从全局配置读取数据集路径）"""
    global _dataset_manager
    if _dataset_manager is None:
        datasets_config = settings.asr.datasets if hasattr(settings, 'asr') else {}
        _dataset_manager = DatasetManager.from_config(datasets_config, project_root=project_root)
    return _dataset_manager


# 榜单管理器（延迟初始化）
_leaderboard_manager: Optional[LeaderboardManager] = None


def _get_leaderboard_manager() -> LeaderboardManager:
    """获取榜单管理器（懒加载单例，首次加载时从公开基准+历史记录初始化）"""
    global _leaderboard_manager
    if _leaderboard_manager is None:
        _leaderboard_manager = LeaderboardManager(
            filepath=os.path.join(asr_result_dir, "leaderboard.json")
        )
        # 首次加载后，扫描历史 benchmark 补充本地结果
        _leaderboard_manager.seed_from_history(asr_results_dir)
    return _leaderboard_manager


def _get_device() -> str:
    """获取计算设备"""
    return "cuda" if (settings.cuda.enabled and torch.cuda.is_available()) else "cpu"


def _task_to_response(task: Task) -> dict:
    """Task → API返回格式（兼容旧版dict字段）"""
    return {
        "task_id": task.task_id,
        "algorithm": task.data.get("algorithm"),
        "filename": task.data.get("filename"),
        "language": task.data.get("language"),
        "type": task.data.get("type", "single"),
        "total_files": task.data.get("total_files"),
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
        "result_file": task.result_file,
        "result": task.data.get("result"),
        "results": task.data.get("results"),
        "processing_time": task.data.get("processing_time"),
        "cer": task.data.get("cer"),
        "cer_detail": task.data.get("cer_detail"),
        "reference_text": task.data.get("reference_text"),
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "user": task.user,
    }


async def update_asr_progress(
    task_id: str, progress: int, message: str,
    status: Optional[TaskStatus] = None,
):
    """更新ASR任务进度（单数据源：只写TaskQueue）"""
    kwargs = {"progress": progress, "message": f"[{_get_step_name(progress)}]{message}"}
    if status:
        kwargs["status"] = status

    await asr_task_queue.update_task(task_id, **kwargs)

    step_name = _get_step_name(progress)
    structured_msg = f"[{step_name}]{message}"
    logger.info(f"[ASR] 任务进度 {task_id}: [{step_name}] {progress}% - {message}")

    # 从TaskQueue重新读取以获取最新状态
    task = await asr_task_queue.get_task(task_id)
    current_status = task.status.value if task else "unknown"

    _safe_create_task(asr_manager.send_progress(task_id, {
        "status": current_status,
        "progress": progress,
        "message": structured_msg,
        "step": step_name,
    }))


async def process_asr_task(queue_task: Task):
    """
    ASR转写任务处理器（由 asr_task_queue 调度执行）
    所有运行态数据写入 queue_task.data，通过 TaskQueue.update_task 持久化
    """
    task_id = queue_task.task_id
    task_data = queue_task.data

    algorithm = task_data.get("algorithm")
    file_path = task_data.get("file_path")
    filename = task_data.get("filename")
    language = task_data.get("language", "zh")
    reference_text = task_data.get("reference_text")

    logger.info(f"[ASR] 开始处理任务: {task_id}, 算法: {algorithm}, 文件: {filename}, "
                f"reference_text={'有' if reference_text else '无'}")

    try:
        # 步骤1: 加载算法模型
        await update_asr_progress(task_id, 10, "正在加载ASR算法模型...",
                                  status=TaskStatus.PROCESSING)
        loop = asyncio.get_event_loop()

        def _get_asr_instance():
            if not ASR_AVAILABLE:
                raise RuntimeError("ASR模块不可用")
            instance = ASRRegistry.get(
                algorithm, device=_get_device(),
                model_dir=os.path.join(asr_model_dir, algorithm),
                offline=settings.asr.offline)
            if not instance:
                raise ValueError(f"未知的ASR算法: {algorithm}")
            if not instance.is_initialized():
                instance.initialize()
            return instance

        asr_instance = await loop.run_in_executor(None, _get_asr_instance)
        logger.info(f"[ASR] ✓ 算法实例已获取: {algorithm}")

        # 步骤2: 读取音频 + 步骤3: 执行转写
        await update_asr_progress(task_id, 30, "正在读取音频文件...")
        await update_asr_progress(task_id, 50, "正在执行语音识别...")

        def _transcribe():
            return asr_instance.transcribe_file(file_path)

        result = await loop.run_in_executor(None, _transcribe)
        logger.info(f"[ASR] 转写完成 - 文本长度: {len(result.text)}, "
                    f"处理时间: {result.processing_time:.3f}s")

        # 步骤4: 保存结果文件
        await update_asr_progress(task_id, 80, "正在保存转写结果...")

        def _save_result():
            result_dir = os.path.join(asr_result_dir, task_id)
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, f"result_{filename}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            return result_path

        result_path = await loop.run_in_executor(None, _save_result)

        # 构造最终结果
        final_result = result.to_dict()
        update_kw = {
            "result_file": result_path,
            "progress": 100,
        }

        # 步骤5: 如果有参考文本，计算CER
        if reference_text:
            from asr.evaluator import compute_cer
            cer, cer_del, cer_ins, cer_sub = compute_cer(reference_text, result.text)
            cer_detail = {
                "delete": round(cer_del, 4),
                "insert": round(cer_ins, 4),
                "substitute": round(cer_sub, 4),
            }
            final_result["cer"] = round(cer, 4)
            final_result["cer_detail"] = cer_detail
            final_result["reference_text"] = reference_text
            logger.info(f"[ASR] CER: {cer:.4f} (del={cer_del:.4f}, "
                        f"ins={cer_ins:.4f}, sub={cer_sub:.4f})")

        # 数据全部写入 Task.data，由 TaskQueue 统一持久化
        update_kw["data"] = {
            **task_data,
            "result": final_result,
            "processing_time": result.processing_time,
        }
        await asr_task_queue.update_task(task_id, **update_kw)

        logger.info(f"[ASR] 任务完成: {task_id}")

        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024 ** 2
            reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
            logger.info(f"[ASR] 显存状态: {allocated_mb:.1f}MB / {reserved_mb:.1f}MB")

        _safe_create_task(asr_manager.send_progress(task_id, {
            "status": "completed",
            "progress": 100,
            "message": "[done]转写完成",
            "step": "done",
        }))

    except Exception as e:
        logger.error(f"[ASR] 任务失败: {task_id}, 错误: {e}")
        import traceback
        logger.error(f"[ASR] 错误堆栈: {traceback.format_exc()}")
        # 不在此设置 status — 由 TaskQueue._worker 统一管理重试/终态
        raise


async def process_batch_asr_task(queue_task: Task):
    """
    批量ASR转写任务处理器
    """
    task_id = queue_task.task_id
    task_data = queue_task.data

    algorithm = task_data.get("algorithm")
    files = task_data.get("files", [])
    language = task_data.get("language", "zh")
    reference_texts = task_data.get("reference_texts", [])

    logger.info(f"[ASR] 开始批量处理任务: {task_id}, 算法: {algorithm}, 文件数: {len(files)}")

    results = []
    try:
        # 加载算法
        await update_asr_progress(task_id, 10, "正在加载ASR算法模型...",
                                  status=TaskStatus.PROCESSING)
        loop = asyncio.get_event_loop()

        def _get_asr_instance():
            if not ASR_AVAILABLE:
                raise RuntimeError("ASR模块不可用")
            instance = ASRRegistry.get(
                algorithm, device=_get_device(),
                model_dir=os.path.join(asr_model_dir, algorithm),
                offline=settings.asr.offline)
            if not instance:
                raise ValueError(f"未知的ASR算法: {algorithm}")
            if not instance.is_initialized():
                instance.initialize()
            return instance

        asr_instance = await loop.run_in_executor(None, _get_asr_instance)

        # 逐文件转写
        total = len(files)
        for i, file_info in enumerate(files):
            file_path = file_info["file_path"]
            filename = file_info["filename"]

            progress = 30 + int(50 * i / total)
            await update_asr_progress(task_id, progress,
                                      f"正在转写 {i + 1}/{total}: {filename}")

            def _transcribe(fp=file_path):
                return asr_instance.transcribe_file(fp)

            result = await loop.run_in_executor(None, _transcribe)
            result_dict = result.to_dict()

            # 如果有参考文本，计算CER
            if i < len(reference_texts) and reference_texts[i]:
                cer, cer_del, cer_ins, cer_sub = compute_cer(
                    reference_texts[i], result.text)
                result_dict["cer"] = round(cer, 4)
                result_dict["cer_detail"] = {
                    "delete": round(cer_del, 4),
                    "insert": round(cer_ins, 4),
                    "substitute": round(cer_sub, 4),
                }
                result_dict["reference_text"] = reference_texts[i]

            results.append({
                "filename": filename,
                **result_dict,
            })

        # 保存结果
        await update_asr_progress(task_id, 80, "正在保存批量转写结果...")

        result_dir = os.path.join(asr_result_dir, task_id)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "batch_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 数据全部写入 Task.data
        await asr_task_queue.update_task(
            task_id,
            result_file=result_path,
            progress=100,
            data={
                **task_data,
                "type": "batch",
                "results": results,
                "total_files": len(results),
            },
        )

        logger.info(f"[ASR] 批量任务完成: {task_id}, 共 {len(results)} 个文件")

        _safe_create_task(asr_manager.send_progress(task_id, {
            "status": "completed",
            "progress": 100,
            "message": f"[done]批量转写完成，共 {len(results)} 个文件",
            "step": "done",
        }))

    except Exception as e:
        logger.error(f"[ASR] 批量任务失败: {task_id}, 错误: {e}")
        import traceback
        logger.error(f"[ASR] 错误堆栈: {traceback.format_exc()}")
        raise


# ===========================
# 数据模型
# ===========================


class ASRAlgorithmInfo(BaseModel):
    """ASR算法信息"""
    name: str
    display_name: str
    description: str
    architecture: str = ""
    params: str = ""
    cer_aishell1: str = ""
    streaming: bool = False
    languages: List[str] = []
    license: str = ""
    initialized: bool = False
    tags: List[str] = []


class BenchmarkRequest(BaseModel):
    """Benchmark请求"""
    algorithms: List[str]
    dataset: str
    max_samples: int = 100
    metrics: List[str] = ["cer", "wer", "rtf"]


# ===========================
# 1. 算法管理
# ===========================


@router.get("/algorithms")
async def list_algorithms(current_user: User = Depends(get_current_active_user)) -> List[ASRAlgorithmInfo]:
    """获取所有可用的ASR算法"""
    algorithms = []
    available = ASRRegistry.list_available()

    for info in available:
        name = info["name"]
        desc = info.get("description", ASR_ALGORITHM_DESCRIPTIONS.get(name, {}))
        if isinstance(desc, dict):
            pass
        else:
            desc = {}

        algorithms.append(ASRAlgorithmInfo(
            name=name,
            display_name=desc.get("display_name", name),
            description=desc.get("description", ""),
            architecture=desc.get("architecture", ""),
            params=desc.get("params", ""),
            cer_aishell1=desc.get("cer_aishell1", ""),
            streaming=desc.get("streaming", False),
            languages=desc.get("languages", []),
            license=desc.get("license", ""),
            initialized=info.get("initialized", False),
            tags=desc.get("tags", []),
        ))

    return algorithms


@router.post("/algorithms/{name}/initialize")
async def initialize_algorithm(
    name: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """初始化指定ASR算法"""
    available = ASRRegistry.list_available()
    algo_names = [a["name"] for a in available]
    if name not in algo_names:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"未知的ASR算法: {name}")

    # 检查是否已初始化
    existing = ASRRegistry.get_initialized(name)
    if existing:
        return {"message": f"算法 {name} 已初始化", "name": name, "initialized": True}

    try:
        instance = ASRRegistry.get(name, device=_get_device(), model_dir=os.path.join(asr_model_dir, name),
                                   offline=settings.asr.offline)
        if not instance:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"无法创建算法实例: {name}")

        loop = asyncio.get_event_loop()

        def _init():
            return instance.initialize()

        success = await loop.run_in_executor(None, _init)

        if success:
            logger.info(f"[ASR] 算法 {name} 初始化成功")
            return {"message": f"算法 {name} 初始化成功", "name": name, "initialized": True}
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"算法 {name} 初始化失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ASR] 算法 {name} 初始化异常: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"初始化异常: {str(e)}")


@router.post("/algorithms/{name}/unload")
async def unload_algorithm(
    name: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """卸载指定ASR算法以释放GPU显存"""
    try:
        ASRRegistry.unload(name)
        logger.info(f"[ASR] 算法 {name} 已卸载")
        return {"message": f"算法 {name} 已卸载", "name": name, "initialized": False}
    except Exception as e:
        logger.error(f"[ASR] 算法 {name} 卸载失败: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"卸载失败: {str(e)}")


# ===========================
# 2. 转写（单文件 + 批量）
# ===========================


@router.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    algorithm: str = Form("paraformer-large"),
    language: str = Form("zh"),
    reference_text: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """上传单个音频文件进行转写"""
    if not audio_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件名为空")

    # 验证文件格式
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in settings.audio.supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的音频格式: {ext}，请使用 {settings.audio.supported_formats}",
        )

    # 验证算法
    available = ASRRegistry.list_available()
    algo_names = [a["name"] for a in available]
    if algorithm not in algo_names:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的算法: {algorithm}")

    # 保存文件（流式写入，避免全量读内存）
    task_id = str(uuid.uuid4())
    upload_dir = os.path.join(asr_upload_dir, task_id)
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, audio_file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    # 提交到队列（数据全部放入 Task.data）
    queue_task = Task(
        task_id=task_id,
        user=current_user.username,
        timeout=settings.asr.transcribe_timeout,
        max_retries=settings.asr.transcribe_max_retries,
        data={
            "algorithm": algorithm,
            "filename": audio_file.filename,
            "file_path": file_path,
            "language": language,
            "reference_text": reference_text,
        },
    )
    submitted = await asr_task_queue.submit(queue_task)
    if not submitted:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务已存在于队列中")

    logger.info(f"[ASR] 转写任务已提交: {task_id}, 算法: {algorithm}, 文件: {audio_file.filename}")

    return {"task_id": task_id, "message": "任务已提交到队列"}


@router.post("/transcribe/batch")
async def transcribe_audio_batch(
    files: List[UploadFile] = File(...),
    algorithm: str = Form("paraformer-large"),
    reference_texts: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """批量上传多个音频文件进行转写"""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请选择文件")

    # 验证算法
    available = ASRRegistry.list_available()
    algo_names = [a["name"] for a in available]
    if algorithm not in algo_names:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的算法: {algorithm}")

    # 解析参考文本
    ref_texts = []
    if reference_texts:
        try:
            ref_texts = json.loads(reference_texts)
            if not isinstance(ref_texts, list):
                ref_texts = [ref_texts]
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="reference_texts 必须是JSON数组")

    # 保存文件
    task_id = str(uuid.uuid4())
    upload_dir = os.path.join(asr_upload_dir, task_id)
    os.makedirs(upload_dir, exist_ok=True)

    file_list = []
    for file in files:
        if not file.filename:
            continue
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.audio.supported_formats:
            continue

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_list.append({
            "filename": file.filename,
            "file_path": file_path,
        })

    if not file_list:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="没有有效的音频文件")

    # 提交到队列（数据全部放入 Task.data）
    queue_task = Task(
        task_id=task_id,
        user=current_user.username,
        timeout=settings.asr.batch_timeout,
        max_retries=settings.asr.batch_max_retries,
        data={
            "algorithm": algorithm,
            "files": file_list,
            "reference_texts": ref_texts,
            "type": "batch",
            "total_files": len(file_list),
        },
    )
    submitted = await asr_task_queue.submit(queue_task)
    if not submitted:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务已存在于队列中")

    logger.info(f"[ASR] 批量转写任务已提交: {task_id}, 算法: {algorithm}, 文件数: {len(file_list)}")

    return {
        "task_id": task_id,
        "total_files": len(file_list),
        "message": f"批量任务已提交，共 {len(file_list)} 个文件",
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    """获取ASR任务状态和结果"""
    task = await asr_task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if task.user != current_user.username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问此任务")
    return _task_to_response(task)


@router.get("/tasks")
async def list_tasks(current_user: User = Depends(get_current_active_user)) -> List[Dict[str, Any]]:
    """获取当前用户的所有ASR任务"""
    tasks = await asr_task_queue.get_user_tasks(current_user.username)
    return [_task_to_response(t) for t in tasks]


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    """取消ASR任务（排队中或处理中均可取消）"""
    task = await asr_task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if task.user != current_user.username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权操作此任务")

    cancelled = await asr_task_queue.cancel(task_id)
    if not cancelled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="任务已完成或已取消，无法取消")
    return {"message": "任务已取消", "task_id": task_id}


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    """删除ASR任务及文件"""
    task = await asr_task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if task.user != current_user.username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权删除此任务")

    # 删除上传文件
    upload_dir = os.path.join(asr_upload_dir, task_id)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)

    # 删除结果文件
    result_dir = os.path.join(asr_result_dir, task_id)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    # 从队列和持久化中移除
    await asr_task_queue.delete_task(task_id)

    return {"message": "任务已删除"}


# ===========================
# 3. Benchmark评测
# ===========================


@router.get("/datasets")
async def list_datasets(current_user: User = Depends(get_current_active_user)) -> List[Dict[str, Any]]:
    """获取可用的ASR评测数据集列表"""
    dm = _get_dataset_manager()
    return dm.list_datasets()


@router.post("/benchmark/run")
async def run_benchmark(
    request: BenchmarkRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """启动ASR Benchmark评测"""
    # 获取数据集样本
    dm = _get_dataset_manager()
    samples = dm.get_test_samples(request.dataset, max_samples=request.max_samples)
    if not samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"数据集 '{request.dataset}' 不可用或没有测试样本",
        )

    # 验证算法
    available = ASRRegistry.list_available()
    algo_names = [a["name"] for a in available]
    for algo in request.algorithms:
        if algo not in algo_names:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的算法: {algo}")

    # ── 记忆功能：相同配置直接返回已有结果 ──
    cache_key = _benchmark_cache_key(request.algorithms, request.dataset, request.max_samples)
    cached = _find_cached_benchmark(request.algorithms, request.dataset, request.max_samples)
    if cached:
        logger.info(f"[ASR] 命中缓存: {cached['bench_id']} (算法={request.algorithms}, 数据集={request.dataset})")
        return {
            **cached,
            "message": "该配置已有测评结果，直接返回（未重新计算）",
        }

    bench_id = f"bench_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()

    asr_benchmarks[bench_id] = {
        "bench_id": bench_id,
        "algorithms": request.algorithms,
        "dataset": request.dataset,
        "max_samples": request.max_samples,
        "metrics": request.metrics,
        "status": "running",
        "progress": 0.0,
        "results": None,
        "ranking": None,
        "report_files": None,
        "created_at": now,
        "updated_at": now,
        "user": current_user.username,
    }

    # 在后台线程执行benchmark
    async def _run_benchmark():
        try:
            benchmark = ASRBenchmark(
                model_dir=asr_model_dir, device=_get_device(),
                max_workers=settings.asr.benchmark_workers,
                per_utterance_limit=settings.asr.per_utterance_limit,
            )
            loop = asyncio.get_event_loop()

            # 进度回调
            def _progress_callback(progress: float):
                asr_benchmarks[bench_id]["progress"] = round(progress, 2)
                asr_benchmarks[bench_id]["updated_at"] = datetime.now().isoformat()

            def _run():
                run = benchmark.run_benchmark(
                    algorithms=request.algorithms,
                    samples=samples,
                    dataset_name=request.dataset,
                    bench_id=bench_id,
                )
                return run

            run_result = await loop.run_in_executor(None, _run)

            # 生成报告
            report_formats = request.metrics if request.metrics else ["json", "html"]
            valid_formats = [f for f in report_formats if f in ["json", "markdown", "excel", "html"]]
            if not valid_formats:
                valid_formats = ["json", "html"]

            report_files = ASRReportGenerator.generate(
                run_result,
                os.path.join(asr_report_dir, bench_id),
                formats=valid_formats,
            )

            # 获取排名
            ranking = benchmark.get_ranking(bench_id)

            # 更新状态
            asr_benchmarks[bench_id]["status"] = "completed"
            asr_benchmarks[bench_id]["progress"] = 100.0
            asr_benchmarks[bench_id]["results"] = run_result.to_dict()["results"]
            asr_benchmarks[bench_id]["ranking"] = ranking
            asr_benchmarks[bench_id]["report_files"] = report_files
            asr_benchmarks[bench_id]["updated_at"] = datetime.now().isoformat()

            # 逐算法持久化到 results/{dataset}/{algorithm}.json
            results_dict = run_result.to_dict()["results"]
            for algo_name, algo_result in results_dict.items():
                if isinstance(algo_result, dict):
                    _save_result(request.dataset, algo_name, algo_result)

            # 更新测评榜单
            try:
                lm = _get_leaderboard_manager()
                updated = lm.update_from_benchmark(bench_id, asr_benchmarks[bench_id])
                if updated:
                    logger.info(f"[ASR] 榜单已更新: {updated} 条记录")
            except Exception as e:
                logger.error(f"[ASR] 更新榜单失败: {e}")

            logger.info(f"[ASR] Benchmark完成: {bench_id}")

        except Exception as e:
            asr_benchmarks[bench_id]["status"] = "failed"
            asr_benchmarks[bench_id]["message"] = str(e)
            asr_benchmarks[bench_id]["updated_at"] = datetime.now().isoformat()

            logger.error(f"[ASR] Benchmark失败: {bench_id}, 错误: {e}")
            import traceback
            logger.error(f"[ASR] 错误堆栈: {traceback.format_exc()}")

    _safe_create_task(_run_benchmark(), log_prefix="[ASR-Benchmark]")

    return {
        "bench_id": bench_id,
        "status": "running",
        "message": "Benchmark评测已启动",
    }


@router.get("/benchmark/{bench_id}")
async def get_benchmark(bench_id: str, current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    """获取Benchmark评测状态和结果"""
    if bench_id not in asr_benchmarks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark不存在")
    return asr_benchmarks[bench_id]


@router.get("/benchmark/{bench_id}/ranking")
async def get_benchmark_ranking(bench_id: str, current_user: User = Depends(get_current_active_user)) -> List[Dict[str, Any]]:
    """获取Benchmark排名"""
    if bench_id not in asr_benchmarks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark不存在")

    ranking = asr_benchmarks[bench_id].get("ranking")
    if ranking is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="评测尚未完成或无排名数据")

    return ranking


@router.get("/benchmark/{bench_id}/report")
async def download_benchmark_report(
    bench_id: str,
    format: str = "json",
    current_user: User = Depends(get_current_user_optional),
):
    """下载Benchmark评测报告"""
    if bench_id not in asr_benchmarks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark不存在")

    report_files = asr_benchmarks[bench_id].get("report_files")
    if not report_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="报告尚未生成")

    if format not in report_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的报告格式: {format}，可用格式: {list(report_files.keys())}",
        )

    file_path = report_files[format]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="报告文件不存在")

    # 根据格式设置媒体类型和文件名
    media_type_map = {
        "json": "application/json",
        "markdown": "text/markdown",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "html": "text/html",
    }
    ext_map = {
        "json": ".json",
        "markdown": ".md",
        "excel": ".xlsx",
        "html": ".html",
    }

    media_type = media_type_map.get(format, "application/octet-stream")
    filename = f"{bench_id}{ext_map.get(format, '.dat')}"

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
    )


@router.get("/benchmark")
async def list_benchmarks(current_user: User = Depends(get_current_active_user)) -> List[Dict[str, Any]]:
    """获取所有Benchmark评测列表"""
    user_benchmarks = [
        b for b in asr_benchmarks.values()
        if b.get("user") == current_user.username
    ]
    return user_benchmarks


# ===========================
# 3.5 测评榜单
# ===========================


@router.get("/leaderboard")
async def get_leaderboard_full() -> Dict[str, Any]:
    """获取完整测评榜单（所有数据集）"""
    lm = _get_leaderboard_manager()
    return lm.get_full()


@router.get("/leaderboard/{dataset_key}")
async def get_leaderboard_dataset(dataset_key: str) -> Dict[str, Any]:
    """获取指定数据集的测评榜单"""
    lm = _get_leaderboard_manager()
    result = lm.get_leaderboard(dataset_key)
    if not result.get("entries"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"数据集 '{dataset_key}' 不存在")
    return result


@router.post("/leaderboard/refresh")
async def refresh_leaderboard(current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    """手动刷新榜单（从公开基准+历史benchmark重建）"""
    lm = _get_leaderboard_manager()
    count = lm.refresh(asr_results_dir)
    return {"message": f"榜单已刷新，共 {count} 个数据集", "datasets": count}


# ===========================
# 4. 外部API（第三方调用）
# ===========================


def _validate_api_key(api_key: Optional[str]) -> bool:
    """验证API Key（配置路径: settings.asr.api_keys.keys）"""
    if not api_key:
        return False
    try:
        if not settings.asr.api_keys.enabled:
            return False
        return api_key in settings.asr.api_keys.keys
    except Exception:
        return False


async def _get_user_from_api_key_or_jwt(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> User:
    """支持API Key或JWT认证（用于v1外部API）"""
    if x_api_key:
        if _validate_api_key(x_api_key):
            from app.core.security import User as UserModel
            return UserModel(username="api_client", is_active=True, hashed_password="")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API Key",
        )

    if current_user:
        return current_user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="请提供X-API-Key头或JWT认证",
    )


@router.post("/v1/recognize")
async def recognize_audio(
    audio_file: UploadFile = File(...),
    algorithm: str = Form("paraformer-large"),
    language: str = Form("zh"),
    current_user: User = Depends(_get_user_from_api_key_or_jwt),
) -> Dict[str, Any]:
    """
    简单识别API — 支持X-API-Key或JWT认证
    同步返回识别结果，适用于第三方集成
    """
    if not audio_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件名为空")

    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in settings.audio.supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的音频格式: {ext}",
        )

    # 验证算法
    available = ASRRegistry.list_available()
    algo_names = [a["name"] for a in available]
    if algorithm not in algo_names:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的算法: {algorithm}")

    # 保存临时文件
    temp_id = str(uuid.uuid4())
    temp_dir = os.path.join(asr_upload_dir, f"v1_{temp_id}")
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, audio_file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    try:
        # 同步执行识别
        loop = asyncio.get_event_loop()

        def _recognize():
            instance = ASRRegistry.get(algorithm, device=_get_device(), model_dir=os.path.join(asr_model_dir, algorithm),
                                       offline=settings.asr.offline)
            if not instance:
                raise ValueError(f"算法 {algorithm} 不可用")
            if not instance.is_initialized():
                instance.initialize()
            return instance.transcribe_file(file_path)

        result = await loop.run_in_executor(None, _recognize)

        return {
            "text": result.text,
            "language": result.language,
            "rtf": round(result.rtf, 4) if result.rtf else None,
            "processing_time": round(result.processing_time, 3),
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "confidence": s.confidence,
                }
                for s in (result.segments or [])
            ],
        }

    except Exception as e:
        logger.error(f"[ASR] v1/recognize 失败: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"识别失败: {str(e)}")

    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/v1/algorithms")
async def list_algorithms_public() -> List[Dict[str, Any]]:
    """获取可用的ASR算法列表（无需认证）"""
    algorithms = []
    for name, desc in ASR_ALGORITHM_DESCRIPTIONS.items():
        algorithms.append({
            "name": name,
            "display_name": desc.get("display_name", name),
            "description": desc.get("description", ""),
            "languages": desc.get("languages", []),
            "streaming": desc.get("streaming", False),
            "tags": desc.get("tags", []),
        })

    return algorithms


# ===========================
# WebSocket
# ===========================


@router.websocket("/ws/{task_id}")
async def asr_ws_endpoint(websocket: WebSocket, task_id: str):
    """ASR任务实时进度推送"""
    await asr_manager.connect(websocket, task_id)
    try:
        # 立即推送一次当前状态
        task = await asr_task_queue.get_task(task_id)
        if task:
            await websocket.send_json({
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
                "step": _get_step_name(task.progress),
            })
            # 终态就不继续轮询了
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                               TaskStatus.TIMEOUT, TaskStatus.CANCELLED):
                return

        # 等待主动推送（asr_manager.send_progress 在状态变更时调用）
        # 同时每5秒做一次保活心跳以防推送丢失
        while True:
            await asyncio.sleep(5)
            task = await asr_task_queue.get_task(task_id)
            if not task:
                break
            await websocket.send_json({
                "status": task.status.value,
                "progress": task.progress,
                "message": task.message,
                "step": _get_step_name(task.progress),
            })
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                               TaskStatus.TIMEOUT, TaskStatus.CANCELLED):
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        asr_manager.disconnect(task_id)


# ===========================
# 5. 健康检查
# ===========================


@router.get("/health")
async def asr_health():
    """ASR服务健康检查（GPU、队列、模型状态）"""
    status = "healthy" if ASR_AVAILABLE else "unavailable"

    info = {
        "status": status,
        "asr_available": ASR_AVAILABLE,
        "queue_depth": asr_task_queue.get_queue_size(),
        "processing_count": asr_task_queue.get_processing_count(),
    }

    if ASR_AVAILABLE:
        # 算法初始化状态
        algorithms = []
        for a in ASRRegistry.list_available():
            algorithms.append({
                "name": a["name"],
                "initialized": a.get("initialized", False),
            })
        info["algorithms"] = algorithms
        info["total_algorithms"] = len(algorithms)
        info["initialized_count"] = sum(1 for a in algorithms if a["initialized"])

    # GPU状态
    if torch.cuda.is_available():
        try:
            device_id = torch.cuda.current_device()
            info["gpu"] = {
                "available": True,
                "device_id": device_id,
                "device_name": torch.cuda.get_device_name(device_id),
                "memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                "memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
            }
        except Exception:
            info["gpu"] = {"available": False, "error": "query_failed"}
    else:
        info["gpu"] = {"available": False}

    return info


# ===========================
# 初始化
# ===========================


def init_asr():
    """初始化ASR模块（预加载常用算法）"""
    if not ASR_AVAILABLE:
        logger.warning("[ASR初始化] ASR模块不可用，跳过初始化")
        return

    logger.info("=" * 60)
    logger.info("[ASR初始化] 开始预加载ASR算法")
    logger.info("=" * 60)

    device = _get_device()
    logger.info(f"[ASR初始化] 使用设备: {device}")

    # 预加载配置为 preload=True 的算法
    preload_list = [
        name for name, desc in ASR_ALGORITHM_DESCRIPTIONS.items()
        if desc.get("preload", False)
    ]
    logger.info(f"[ASR初始化] 预加载列表: {preload_list}")

    init_stats = {"success": [], "failed": []}

    for idx, name in enumerate(preload_list, 1):
        try:
            logger.info(f"[ASR初始化] [{idx}/{len(preload_list)}] 正在预加载 '{name}'...")
            model_start = time.time()

            instance = ASRRegistry.get(name, device=device, model_dir=os.path.join(asr_model_dir, name),
                                       offline=settings.asr.offline)
            if not instance:
                init_stats["failed"].append((name, "无法创建实例"))
                continue

            success = instance.initialize()
            model_time = time.time() - model_start

            if success:
                init_stats["success"].append((name, model_time))
                logger.info(f"[ASR初始化] ✓ '{name}' 预加载成功 (耗时: {model_time:.2f}s)")
            else:
                init_stats["failed"].append((name, "初始化返回False"))
                logger.warning(f"[ASR初始化] ✗ '{name}' 初始化失败 (耗时: {model_time:.2f}s)")

        except Exception as e:
            init_stats["failed"].append((name, str(e)))
            logger.error(f"[ASR初始化] ✗ '{name}' 预加载异常: {e}")

    # 汇总
    logger.info("=" * 60)
    logger.info(f"[ASR初始化] 预加载成功: {len(init_stats['success'])}个")
    if init_stats["failed"]:
        logger.warning(f"[ASR初始化] 预加载失败: {len(init_stats['failed'])}个")
    logger.info("=" * 60)
