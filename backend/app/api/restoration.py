"""
音频修复API路由
提供去混响、超分辨率等音频修复功能的RESTful API接口
"""

import os
import shutil
import uuid
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.auth import get_current_active_user, get_current_user_optional
from app.core.security import User
from app.core.config import settings
from app.core.logging_config import logger

# 创建线程池
executor = ThreadPoolExecutor(max_workers=2)

# 导入修复模块
import sys

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

# 任务存储
restoration_tasks: Dict[str, Dict[str, Any]] = {}


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
    restoration_tasks[task_id] = task_info.dict()

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
        restoration_tasks[task_id] = task_info.dict()
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
    """提交音频修复任务"""
    if task_id not in restoration_tasks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task_info = restoration_tasks[task_id]
    if task_info["status"] == "processing":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务正在处理中")

    algorithm = task_info["algorithm"]
    upload_dir = os.path.join(settings.paths.upload_dir, task_id)
    filename = task_info["filename"]
    file_path = os.path.join(upload_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文件不存在")

    # 更新状态
    task_info["status"] = "processing"
    task_info["message"] = "正在处理..."

    # 在子线程中处理
    import threading

    def process():
        try:
            task_info["progress"] = 0.1

            # 获取算法实例
            restorers = get_available_restorers()
            if algorithm not in restorers:
                raise ValueError(f"算法不可用: {algorithm}")

            restorer_cls = restorers[algorithm]["class"]
            restorer = restorer_cls(device="cuda" if settings.cuda.enabled else "cpu")
            success = restorer.initialize()

            if not success:
                raise RuntimeError("算法初始化失败")

            task_info["progress"] = 0.3

            # 读取音频
            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(file_path)
            task_info["progress"] = 0.5

            # 执行修复
            result = restorer.restore(audio, sr)
            task_info["progress"] = 0.8

            # 保存结果
            result_dir = os.path.join(settings.paths.result_dir, task_id)
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, f"restored_{filename}")
            sf.write(result_path, result.audio, result.sample_rate)

            task_info["status"] = "completed"
            task_info["progress"] = 1.0
            task_info["message"] = "处理完成"
            task_info["result_file"] = result_path
            task_info["processing_time"] = result.processing_time
            task_info["metadata"] = result.metadata

        except Exception as e:
            task_info["status"] = "failed"
            task_info["message"] = str(e)
            logger.error(f"音频修复失败: {e}")

    threading.Thread(target=process, daemon=True).start()

    return {"task_id": task_id, "message": "任务已提交"}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    """获取任务状态"""
    if task_id not in restoration_tasks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return restoration_tasks[task_id]


@router.get("/tasks")
async def list_tasks(current_user: User = Depends(get_current_active_user)) -> List[Dict[str, Any]]:
    """获取所有任务"""
    return list(restoration_tasks.values())


@router.get("/source/{task_id}")
async def get_source_audio(
    task_id: str,
    current_user: User = Depends(get_current_user_optional),
):
    """获取上传的原始音频文件（用于试听对比，支持 ?token= 查询参数）"""
    if task_id not in restoration_tasks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task_info = restoration_tasks[task_id]
    upload_dir = os.path.join(settings.paths.upload_dir, task_id)
    filename = task_info["filename"]
    file_path = os.path.join(upload_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="原始文件不存在")

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
    if task_id not in restoration_tasks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task_info = restoration_tasks[task_id]
    if task_info["status"] != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="任务未完成")

    result_file = task_info.get("result_file")
    if not result_file or not os.path.exists(result_file):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="结果文件不存在")

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


def init_restoration():
    """初始化音频修复模块（启动时调用）"""
    if not RESTORATION_AVAILABLE:
        logger.warning("音频修复模块不可用")
        return

    logger.info("=" * 60)
    logger.info("[音频修复算法初始化]")
    logger.info("=" * 60)

    restorers = get_available_restorers()
    for key, info in restorers.items():
        logger.info(f"  {key}: {info.get('name', 'N/A')} - {info.get('description', 'N/A')}")

    logger.info(f"共 {len(restorers)} 个音频修复算法可用")
    logger.info("=" * 60)
