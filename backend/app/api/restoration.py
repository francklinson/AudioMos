"""
音频修复API路由
提供去混响、超分辨率等音频修复功能的RESTful API接口
"""

import os
import shutil
import uuid
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.auth import get_current_active_user, get_current_user_optional
from app.core.security import User
from app.core.config import settings
from app.core.logging_config import logger
from app.core.task_queue import TaskQueue, Task, TaskStatus
from app.core.websocket import ConnectionManager

# 导入修复模块
import sys
import torch

project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "app", "algorithms"))

try:
    from restoration import RestorationRegistry, get_available_restorers, get_restoration_description

    RESTORATION_AVAILABLE = True
    logger.info("✓ 音频修复模块加载成功")
except ImportError as e:
    RESTORATION_AVAILABLE = False
    logger.warning(f"音频修复模块加载失败: {e}")

router = APIRouter(prefix="/restoration", tags=["音频修复"])

# 确保目录存在
Path(settings.paths.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.paths.result_dir).mkdir(parents=True, exist_ok=True)

# 任务存储（API 层状态源，对齐 MOS 模式）
restoration_tasks: Dict[str, Dict[str, Any]] = {}

# 修复算法实例缓存（预加载）
_restorer_instances: Dict[str, Any] = {}

# 音频修复独立任务队列（与 MOS 队列隔离，避免互相阻塞）
restoration_task_queue = TaskQueue(max_workers=1)

# WebSocket 连接管理器（restoration 专属实例）
restoration_manager = ConnectionManager()

# 进度步骤映射（与前端 restorationStepNames 对应）
RESTORATION_PROGRESS_STEPS = {
    0: 'queued',
    10: 'loading',
    30: 'reading',
    50: 'processing',
    80: 'saving',
    100: 'done',
}


def _get_step_name(progress: int) -> str:
    """根据进度值推断当前步骤名"""
    step = 'processing'
    for p, s in sorted(RESTORATION_PROGRESS_STEPS.items()):
        if progress >= p:
            step = s
    return step


async def update_restoration_progress(task_id: str, progress: int, message: str):
    """更新修复任务进度（同步本地 dict + task_queue + WebSocket 推送）"""
    if task_id not in restoration_tasks:
        return

    # 首次进度更新时将状态从 queued → processing
    if restoration_tasks[task_id].get("status") == "queued":
        restoration_tasks[task_id]["status"] = "processing"

    restoration_tasks[task_id]["progress"] = progress
    step_name = _get_step_name(progress)
    structured_msg = f"[{step_name}]{message}"
    restoration_tasks[task_id]["message"] = structured_msg
    restoration_tasks[task_id]["updated_at"] = datetime.now().isoformat()

    # 同步到 task_queue
    await restoration_task_queue.update_task(task_id, progress=progress, message=structured_msg)

    logger.info(f"[音频修复] 任务进度 {task_id}: [{step_name}] {progress}% - {message}")

    # WebSocket 推送
    asyncio.ensure_future(restoration_manager.send_progress(task_id, {
        "status": restoration_tasks[task_id]["status"],
        "progress": progress,
        "message": structured_msg,
        "step": step_name,
    }))


async def process_restoration_task(queue_task: Task):
    """
    音频修复任务处理器（由 restoration_task_queue 调度执行）

    从 queue_task.data 取业务参数，用 run_in_executor 跑同步 restorer.restore，
    各阶段更新进度（本地 dict + task_queue + WebSocket）。
    """
    task_id = queue_task.task_id
    task_data = queue_task.data

    algorithm = task_data.get("algorithm")
    filename = task_data.get("filename")
    file_path = task_data.get("file_path")

    logger.info(f"[音频修复] 开始处理任务: {task_id}, 算法: {algorithm}, 文件: {filename}")

    try:
        # 步骤1: 加载算法模型
        await update_restoration_progress(task_id, 10, "正在加载算法模型...")
        loop = asyncio.get_event_loop()

        def _get_restorer_sync():
            return _get_restorer(algorithm)

        restorer = await loop.run_in_executor(None, _get_restorer_sync)
        logger.info(f"[音频修复] ✓ 算法实例已获取: {algorithm}")

        # 步骤2: 读取音频
        await update_restoration_progress(task_id, 30, "正在读取音频文件...")

        def _read_audio():
            import soundfile as sf
            import numpy as np
            audio, sr = sf.read(file_path)
            logger.info(f"[音频修复] 音频读取完成 - 形状: {audio.shape}, 采样率: {sr}")
            logger.info(f"[音频修复] 音频范围: [{audio.min():.4f}, {audio.max():.4f}], RMS: {np.sqrt(np.mean(audio**2)):.4f}")
            return audio, sr

        audio, sr = await loop.run_in_executor(None, _read_audio)

        # 步骤3: 执行修复
        await update_restoration_progress(task_id, 50, "正在执行音频修复...")

        def _restore():
            return restorer.restore(audio, sr)

        result = await loop.run_in_executor(None, _restore)
        logger.info(f"[音频修复] 修复完成 - 输出形状: {result.audio.shape}, 采样率: {result.sample_rate}")
        logger.info(f"[音频修复] 处理时间: {result.processing_time:.3f}s")
        logger.info(f"[音频修复] 元数据: {result.metadata}")

        # 步骤4: 保存结果
        await update_restoration_progress(task_id, 80, "正在保存修复结果...")

        def _save_result():
            result_dir = os.path.join(settings.paths.result_dir, task_id)
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, f"restored_{filename}")
            import soundfile as sf
            sf.write(result_path, result.audio, result.sample_rate)
            # 验证保存
            saved_audio, saved_sr = sf.read(result_path)
            logger.info(f"[音频修复] 验证保存文件 - 形状: {saved_audio.shape}, 采样率: {saved_sr}")
            return result_path

        result_path = await loop.run_in_executor(None, _save_result)

        # 完成
        restoration_tasks[task_id]["status"] = "completed"
        restoration_tasks[task_id]["progress"] = 100
        restoration_tasks[task_id]["message"] = "[done]处理完成"
        restoration_tasks[task_id]["result_file"] = result_path
        restoration_tasks[task_id]["processing_time"] = result.processing_time
        restoration_tasks[task_id]["metadata"] = result.metadata
        restoration_tasks[task_id]["updated_at"] = datetime.now().isoformat()

        await restoration_task_queue.update_task(task_id, progress=100, message="[done]处理完成")

        logger.info(f"[音频修复] 任务完成: {task_id}")

        # 显存状态报告(不主动清理,依赖GPU监控线程)
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"[音频修复] 显存状态: {allocated_mb:.1f}MB / {reserved_mb:.1f}MB")

        # 推送完成状态
        asyncio.ensure_future(restoration_manager.send_progress(task_id, {
            "status": "completed",
            "progress": 100,
            "message": "[done]处理完成",
            "step": "done",
        }))

    except Exception as e:
        restoration_tasks[task_id]["status"] = "failed"
        restoration_tasks[task_id]["message"] = str(e)
        restoration_tasks[task_id]["updated_at"] = datetime.now().isoformat()

        logger.error(f"[音频修复] 任务失败: {task_id}, 错误: {e}")
        import traceback
        logger.error(f"[音频修复] 错误堆栈: {traceback.format_exc()}")

        asyncio.ensure_future(restoration_manager.send_progress(task_id, {
            "status": "failed",
            "progress": restoration_tasks[task_id].get("progress", 0),
            "message": str(e),
            "step": "error",
        }))


def init_restoration():
    """
    初始化音频修复算法（预加载常用算法）
    生产环境(4090 24GB显存)足够容纳所有模型
    """
    global _restorer_instances
    import time

    init_start = time.time()

    if not RESTORATION_AVAILABLE:
        logger.warning("[音频修复初始化] 音频修复模块不可用，跳过初始化")
        return

    logger.info("=" * 60)
    logger.info("[音频修复初始化] 开始预加载音频修复算法")
    logger.info(f"[音频修复初始化] 当前已加载实例: {list(_restorer_instances.keys())}")
    logger.info("=" * 60)

    # 获取可用算法
    restorers = get_available_restorers()
    logger.info(f"[音频修复初始化] 发现 {len(restorers)} 个可用算法: {list(restorers.keys())}")

    # 检查模块可用性状态
    try:
        from restoration import DEREVERB_AVAILABLE, SUPERRES_AVAILABLE, DENOISE_ADAPTER_AVAILABLE
        logger.info(f"[音频修复初始化] 模块可用性状态:")
        logger.info(f"[音频修复初始化]   - DEREVERB_AVAILABLE: {DEREVERB_AVAILABLE}")
        logger.info(f"[音频修复初始化]   - SUPERRES_AVAILABLE: {SUPERRES_AVAILABLE}")
        logger.info(f"[音频修复初始化]   - DENOISE_ADAPTER_AVAILABLE: {DENOISE_ADAPTER_AVAILABLE}")

        # 如果模块标记为可用但实际没有算法，给出警告
        if not restorers:
            logger.warning("[音频修复初始化] 警告: 所有修复算法模块都标记为不可用或导入失败!")
            logger.warning(f"[音频修复初始化]   DEREVERB_AVAILABLE={DEREVERB_AVAILABLE}, SUPERRES_AVAILABLE={SUPERRES_AVAILABLE}, DENOISE_ADAPTER_AVAILABLE={DENOISE_ADAPTER_AVAILABLE}")
    except Exception as e:
        logger.warning(f"[音频修复初始化] 无法获取模块可用性状态: {e}")
        import traceback
        logger.warning(f"[音频修复初始化] 错误详情: {traceback.format_exc()}")

    # 预加载配置：轻量级算法在启动时预加载
    # 生产环境4090显存充足，可根据需要调整
    preload_list = [
        "dereverberation",              # 去混响 - 常用
        "clearvoice_frcrn_se_16k",      # FRCRN降噪 - 轻量快速
        "clearvoice_mossformer_gan_se_16k",  # MossFormerGAN - 轻量
        "spectral_subtraction",         # 谱减法 - 无模型
        "wiener_filtering",             # 维纳滤波 - 无模型
    ]

    device = "cuda" if (settings.cuda.enabled and torch.cuda.is_available()) else "cpu"
    logger.info(f"[音频修复初始化] 使用设备: {device}")
    logger.info(f"[音频修复初始化] 预加载列表: {preload_list}")

    init_stats = {'success': [], 'failed': [], 'skipped': []}

    # 预加载配置的算法
    for idx, name in enumerate(preload_list, 1):
        if name not in restorers:
            logger.warning(f"[音频修复初始化] [{idx}/{len(preload_list)}] 算法 '{name}' 不在可用列表中，跳过")
            init_stats['skipped'].append((name, '不在可用列表'))
            continue

        try:
            logger.info(f"[音频修复初始化] [{idx}/{len(preload_list)}] 正在预加载 '{name}'...")
            model_start = time.time()

            restorer_cls = restorers[name]["class"]
            logger.info(f"[音频修复初始化]   使用类: {restorer_cls.__name__}")

            restorer = restorer_cls(device=device)
            logger.info(f"[音频修复初始化]   实例创建完成，开始初始化...")

            success = restorer.initialize()
            model_time = time.time() - model_start

            if success:
                _restorer_instances[name] = restorer
                init_stats['success'].append((name, model_time))
                logger.info(f"[音频修复初始化] ✓ '{name}' 预加载成功 (耗时: {model_time:.2f}s)")
            else:
                init_stats['failed'].append((name, '初始化返回False'))
                logger.warning(f"[音频修复初始化] ✗ '{name}' 初始化失败 (耗时: {model_time:.2f}s)")
        except Exception as e:
            init_stats['failed'].append((name, str(e)))
            logger.error(f"[音频修复初始化] ✗ '{name}' 预加载异常: {e}")
            import traceback
            logger.error(f"[音频修复初始化] 错误详情: {traceback.format_exc()}")

    # 为未加载的算法创建占位条目（延迟加载标记）
    lazy_load_list = []
    for name in restorers.keys():
        if name not in _restorer_instances:
            _restorer_instances[name] = None  # 延迟加载标记
            lazy_load_list.append(name)

    if lazy_load_list:
        logger.info(f"[音频修复初始化] 以下算法将延迟加载: {lazy_load_list}")

    # 汇总统计
    total_time = time.time() - init_start
    loaded_count = len(init_stats['success'])
    lazy_count = len(lazy_load_list)

    logger.info("=" * 60)
    logger.info("[音频修复初始化] 初始化完成统计")
    logger.info(f"[音频修复初始化] 总耗时: {total_time:.2f}s")
    logger.info(f"[音频修复初始化] 预加载成功: {loaded_count}个 - {[m[0] for m in init_stats['success']]}")
    if init_stats['failed']:
        logger.warning(f"[音频修复初始化] 预加载失败: {len(init_stats['failed'])}个 - {[m[0] for m in init_stats['failed']]}")
    if init_stats['skipped']:
        logger.info(f"[音频修复初始化] 跳过: {len(init_stats['skipped'])}个 - {[m[0] for m in init_stats['skipped']]}")
    logger.info(f"[音频修复初始化] 延迟加载: {lazy_count}个")
    logger.info(f"[音频修复初始化] 当前已加载实例: {[k for k, v in _restorer_instances.items() if v is not None]}")
    logger.info("=" * 60)


def _get_restorer(name: str):
    """
    获取修复算法实例（带缓存）
    如果实例已预加载，直接返回；否则延迟加载并缓存
    """
    global _restorer_instances
    import time

    # 如果 _restorer_instances 为空，尝试重新初始化
    if not _restorer_instances:
        logger.warning(f"[音频修复] _restorer_instances 为空，尝试重新初始化...")
        try:
            # 重新获取可用算法并填充 _restorer_instances
            restorers = get_available_restorers()
            for key in restorers.keys():
                if key not in _restorer_instances:
                    _restorer_instances[key] = None  # 标记为延迟加载
            logger.info(f"[音频修复] 重新初始化完成，可用算法: {list(_restorer_instances.keys())}")
        except Exception as e:
            logger.error(f"[音频修复] 重新初始化失败: {e}")

    if name not in _restorer_instances:
        logger.error(f"[音频修复] 获取实例失败: 未知的修复算法 '{name}'")
        logger.error(f"[音频修复] 当前可用算法列表: {list(_restorer_instances.keys())}")
        logger.error(f"[音频修复] RESTORATION_AVAILABLE: {RESTORATION_AVAILABLE}")
        raise ValueError(f"未知的修复算法: {name}")

    # 如果已预加载，直接返回
    if _restorer_instances[name] is not None:
        logger.debug(f"[音频修复] 使用预加载实例: '{name}'")
        return _restorer_instances[name]

    # 延迟加载
    logger.info("=" * 60)
    logger.info(f"[音频修复延迟加载] 开始延迟加载算法: '{name}'")
    lazy_start = time.time()

    restorers = get_available_restorers()

    if name not in restorers:
        logger.error(f"[音频修复延迟加载] 算法不可用: '{name}'")
        raise ValueError(f"算法不可用: {name}")

    restorer_info = restorers[name]
    restorer_cls = restorer_info["class"]
    device = "cuda" if (settings.cuda.enabled and torch.cuda.is_available()) else "cpu"

    logger.info(f"[音频修复延迟加载] 使用类: {restorer_cls.__name__}")
    logger.info(f"[音频修复延迟加载] 使用设备: {device}")

    try:
        logger.info(f"[音频修复延迟加载] 创建实例...")
        instance_start = time.time()
        restorer = restorer_cls(device=device)
        instance_time = time.time() - instance_start
        logger.info(f"[音频修复延迟加载] 实例创建完成 (耗时: {instance_time:.2f}s)")

        logger.info(f"[音频修复延迟加载] 开始初始化...")
        init_start = time.time()
        success = restorer.initialize()
        init_time = time.time() - init_start

        if not success:
            logger.error(f"[音频修复延迟加载] ✗ 算法初始化失败: '{name}' (初始化耗时: {init_time:.2f}s)")
            raise RuntimeError(f"算法初始化失败: {name}")

        # 缓存实例
        _restorer_instances[name] = restorer
        total_time = time.time() - lazy_start
        logger.info(f"[音频修复延迟加载] ✓ '{name}' 延迟加载成功")
        logger.info(f"[音频修复延迟加载]   - 实例创建: {instance_time:.2f}s")
        logger.info(f"[音频修复延迟加载]   - 模型初始化: {init_time:.2f}s")
        logger.info(f"[音频修复延迟加载]   - 总耗时: {total_time:.2f}s")
        logger.info("=" * 60)

        return restorer
    except Exception as e:
        logger.error(f"[音频修复延迟加载] ✗ '{name}' 延迟加载异常: {e}")
        import traceback
        logger.error(f"[音频修复延迟加载] 错误详情: {traceback.format_exc()}")
        raise


# ===========================
# 数据模型
# ===========================


class RestorationAlgorithmInfo(BaseModel):
    """算法信息"""

    name: str
    display_name: str
    description: str
    type: str
    advantages: List[str] = []
    limitations: List[str] = []
    initialized: bool = False


class RestorationTaskInfo(BaseModel):
    """任务信息"""

    task_id: str
    algorithm: str
    filename: str
    status: str  # pending, processing, completed, failed
    created_at: str
    progress: float = 0.0
    message: str = ""
    result_file: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict] = None


# ===========================
# API端点
# ===========================


@router.get("/algorithms")
async def list_algorithms(current_user: User = Depends(get_current_active_user)) -> List[RestorationAlgorithmInfo]:
    """获取所有可用的音频修复算法（修复 + 降噪）"""
    algorithms = []

    restorers = get_available_restorers()
    for key, info in restorers.items():
        desc = get_restoration_description(key) or {}

        # 检查模型是否已下载
        downloaded = False
        try:
            import os as _os
            ckpt_map = {
                "clearvoice_frcrn_se_16k": "FRCRN_SE_16K",
                "clearvoice_mossformer2_se_48k": "MossFormer2_SE_48K",
                "clearvoice_mossformer_gan_se_16k": "MossFormerGAN_SE_16K",
                "clearvoice_mossformer2_ss_16k": "MossFormer2_SS_16K",
                "clearvoice_mossformer2_sr_48k": "MossFormer2_SR_48K",
            }
            ckpt = ckpt_map.get(key)
            if ckpt:
                best_path = _os.path.join(
                    project_root, "models/clearvoice", ckpt, "last_best_checkpoint"
                )
                downloaded = _os.path.isfile(best_path)
        except Exception:
            pass

        algorithms.append(
            RestorationAlgorithmInfo(
                name=key,
                display_name=desc.get("name", info.get("name", key)),
                description=desc.get("description", info.get("description", "")),
                type=desc.get("type", "未知"),
                advantages=desc.get("advantages", []),
                limitations=desc.get("limitations", []),
                initialized=downloaded,  # 复用 initialized 字段表示模型已下载
            )
        )

    return algorithms


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    algorithm: str = Form("dereverberation"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """上传音频文件"""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件名为空")

    # 验证文件格式
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.audio.supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的音频格式: {ext}，请使用 {settings.audio.supported_formats}",
        )

    # 验证算法
    restorers = get_available_restorers()
    if algorithm not in restorers:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的算法: {algorithm}")

    # 保存文件
    task_id = str(uuid.uuid4())
    upload_dir = os.path.join(settings.paths.upload_dir, task_id)
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 创建任务
    task_info = RestorationTaskInfo(
        task_id=task_id,
        algorithm=algorithm,
        filename=file.filename,
        status="pending",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    task_dict = task_info.dict()
    task_dict["user"] = current_user.username
    restoration_tasks[task_id] = task_dict

    return {
        "task_id": task_id,
        "filename": file.filename,
        "algorithm": algorithm,
        "message": "文件上传成功，请调用处理接口开始修复",
    }


@router.post("/upload-batch")
async def upload_audio_batch(
    files: List[UploadFile] = File(...),
    algorithm: str = Form(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """批量上传音频文件"""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请选择文件")

    restorers = get_available_restorers()
    if algorithm not in restorers:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的算法: {algorithm}")

    task_ids = []
    filenames = []
    for file in files:
        if not file.filename:
            continue
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.audio.supported_formats:
            continue

        task_id = str(uuid.uuid4())
        upload_dir = os.path.join(settings.paths.upload_dir, task_id)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        task_info = RestorationTaskInfo(
            task_id=task_id,
            algorithm=algorithm,
            filename=file.filename,
            status="pending",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        task_dict = task_info.dict()
        task_dict["user"] = current_user.username
        restoration_tasks[task_id] = task_dict
        task_ids.append(task_id)
        filenames.append(file.filename)

    return {
        "task_ids": task_ids,
        "filenames": filenames,
        "algorithm": algorithm,
        "count": len(task_ids),
        "message": f"成功上传 {len(task_ids)} 个文件",
    }


@router.post("/process/{task_id}")
async def process_restoration(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """提交音频修复任务到队列"""
    logger.info(f"[音频修复] 收到处理请求 - task_id: {task_id}, user: {current_user.username}")

    if task_id not in restoration_tasks:
        logger.error(f"[音频修复] 任务不存在: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task_info = restoration_tasks[task_id]

    if task_info["status"] in ("processing", "queued"):
        logger.warning(f"[音频修复] 任务已在队列或处理中: {task_id}, 状态: {task_info['status']}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务正在处理中")

    algorithm = task_info["algorithm"]
    upload_dir = os.path.join(settings.paths.upload_dir, task_id)
    filename = task_info["filename"]
    file_path = os.path.join(upload_dir, filename)

    logger.info(f"[音频修复] 算法: {algorithm}, 文件名: {filename}, 文件路径: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"[音频修复] 文件不存在: {file_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文件不存在")

    # 更新本地状态为 queued
    task_info["status"] = "queued"
    task_info["progress"] = 0
    task_info["message"] = "[queued]等待处理..."
    task_info["updated_at"] = datetime.now().isoformat()

    # 构造队列任务并提交
    queue_task = Task(
        task_id=task_id,
        user=current_user.username,
        data={
            "algorithm": algorithm,
            "filename": filename,
            "file_path": file_path,
        },
    )

    submitted = await restoration_task_queue.submit(queue_task)
    if not submitted:
        # 任务已存在（可能是重复提交）
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务已存在于队列中")

    logger.info(f"[音频修复] 任务已提交到队列: {task_id}")
    return {"task_id": task_id, "message": "任务已提交到队列"}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    """获取任务状态"""
    if task_id not in restoration_tasks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return restoration_tasks[task_id]


@router.get("/tasks")
async def list_tasks(current_user: User = Depends(get_current_active_user)) -> List[Dict[str, Any]]:
    """获取当前用户的所有任务"""
    return [t for t in restoration_tasks.values() if t.get("user") == current_user.username]


@router.get("/source/{task_id}")
async def get_source_audio(
    task_id: str,
    current_user: User = Depends(get_current_user_optional),
):
    """获取上传的原始音频文件（用于试听对比，支持 ?token= 查询参数）"""
    logger.info(f"[音频修复] 获取原始音频 - task_id: {task_id}")

    if task_id not in restoration_tasks:
        logger.error(f"[音频修复] 任务不存在: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task_info = restoration_tasks[task_id]
    upload_dir = os.path.join(settings.paths.upload_dir, task_id)
    filename = task_info["filename"]
    file_path = os.path.join(upload_dir, filename)

    logger.info(f"[音频修复] 原始音频路径: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"[音频修复] 原始文件不存在: {file_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="原始文件不存在")

    # 验证音频文件
    try:
        import soundfile as sf
        audio, sr = sf.read(file_path)
        logger.info(f"[音频修复] 原始音频验证 - 形状: {audio.shape}, 采样率: {sr}, 范围: [{audio.min():.4f}, {audio.max():.4f}]")
    except Exception as e:
        logger.warning(f"[音频修复] 无法验证原始音频: {e}")

    logger.info(f"[音频修复] 返回原始音频文件: {file_path}")
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"original_{filename}",
        headers={
            "X-Task-Id": task_id,
            "Accept-Ranges": "bytes",
        },
    )


@router.get("/download/{task_id}")
async def download_result(
    task_id: str,
    current_user: User = Depends(get_current_user_optional),
):
    """下载/播放修复结果（支持 ?token= 查询参数）"""
    logger.info(f"[音频修复] 下载修复结果 - task_id: {task_id}")

    if task_id not in restoration_tasks:
        logger.error(f"[音频修复] 任务不存在: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task_info = restoration_tasks[task_id]
    logger.info(f"[音频修复] 任务状态: {task_info.get('status')}, 结果文件: {task_info.get('result_file')}")

    if task_info["status"] != "completed":
        logger.warning(f"[音频修复] 任务未完成: {task_id}, 状态: {task_info['status']}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务未完成")

    result_file = task_info.get("result_file")
    if not result_file:
        logger.error(f"[音频修复] 结果文件路径为空: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="结果文件不存在")

    if not os.path.exists(result_file):
        logger.error(f"[音频修复] 结果文件不存在: {result_file}")
        # 列出结果目录内容
        result_dir = os.path.dirname(result_file)
        if os.path.exists(result_dir):
            files = os.listdir(result_dir)
            logger.info(f"[音频修复] 结果目录内容: {files}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="结果文件不存在")

    # 验证结果音频文件
    try:
        import soundfile as sf
        audio, sr = sf.read(result_file)
        logger.info(f"[音频修复] 结果音频验证 - 形状: {audio.shape}, 采样率: {sr}, 范围: [{audio.min():.4f}, {audio.max():.4f}]")
    except Exception as e:
        logger.warning(f"[音频修复] 无法验证结果音频: {e}")

    logger.info(f"[音频修复] 返回修复结果文件: {result_file}")
    return FileResponse(
        result_file,
        filename=os.path.basename(result_file),
        headers={"Accept-Ranges": "bytes"},
    )


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    """删除任务及文件"""
    if task_id not in restoration_tasks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    # 用户隔离
    if restoration_tasks[task_id].get("user") != current_user.username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权删除此任务")

    # 删除上传文件
    upload_dir = os.path.join(settings.paths.upload_dir, task_id)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)

    # 删除结果文件
    result_dir = os.path.join(settings.paths.result_dir, task_id)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    del restoration_tasks[task_id]

    return {"message": "任务已删除"}


@router.get("/gpu-status")
async def get_gpu_status(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """获取GPU显存状态"""
    try:
        from app.core.gpu_monitor import get_gpu_monitor
        
        monitor = get_gpu_monitor()
        if monitor is None:
            return {
                "gpu_monitor": False,
                "message": "GPU监控未启动"
            }
        
        status = monitor.get_status()
        return {
            "gpu_monitor": True,
            **status
        }
        
    except Exception as e:
        logger.error(f"[GPU状态] 获取失败: {e}")
        return {
            "gpu_monitor": False,
            "error": str(e)
        }


@router.websocket("/ws/{task_id}")
async def restoration_ws_endpoint(websocket: WebSocket, task_id: str):
    """音频修复任务实时进度推送

    连接时立即推送一次当前状态（解决前端 WS 连接晚于后端首次推送的时序问题）；
    之后每 2s 心跳推送；任务进入终态（completed/failed）后主动断开。
    """
    await restoration_manager.connect(websocket, task_id)
    try:
        # 立即推送一次当前状态
        if task_id in restoration_tasks:
            t = restoration_tasks[task_id]
            await websocket.send_json({
                "status": t.get("status"),
                "progress": t.get("progress", 0),
                "message": t.get("message", ""),
                "step": _get_step_name(t.get("progress", 0)),
            })
        while True:
            if task_id in restoration_tasks:
                t = restoration_tasks[task_id]
                await websocket.send_json({
                    "status": t.get("status"),
                    "progress": t.get("progress", 0),
                    "message": t.get("message", ""),
                    "step": _get_step_name(t.get("progress", 0)),
                })
                # 终态 → 推送后断开
                if t.get("status") in ("completed", "failed"):
                    break
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        restoration_manager.disconnect(task_id)
