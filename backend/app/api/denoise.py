"""
降噪算法测评API路由
提供降噪算法测评的RESTful API接口
"""

import os
import shutil
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Annotated
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.auth import get_current_active_user
from app.core.security import User
from app.core.config import settings
from app.core.logging_config import logger
from app.core.task_queue import task_queue, Task, TaskStatus

# 创建线程池
executor = ThreadPoolExecutor(max_workers=2)

# 导入降噪模块
import sys
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'app', 'algorithms'))

try:
    from denoise import DenoiserRegistry, get_available_denoisers
    from denoise.evaluator import BatchEvaluator, DenoiseEvaluation
    from denoise.report_generator import ReportGenerator
    DENOISE_AVAILABLE = True
    logger.info("✓ 降噪模块加载成功")
except ImportError as e:
    DENOISE_AVAILABLE = False
    logger.warning(f"降噪模块加载失败: {e}")

import pandas as pd


router = APIRouter(prefix="/denoise", tags=["降噪测评"])

# 确保目录存在
Path(settings.paths.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.paths.result_dir).mkdir(parents=True, exist_ok=True)
Path(settings.paths.temp_dir).mkdir(parents=True, exist_ok=True)

# 任务存储
denoise_tasks = {}

# 初始化降噪器
_denoisers = {}


def init_denoisers():
    """初始化所有降噪算法（按需延迟加载策略）"""
    global _denoisers

    if not DENOISE_AVAILABLE:
        logger.warning("降噪模块不可用，跳过初始化")
        return

    logger.info("=" * 60)
    logger.info("[降噪算法初始化] 开始注册降噪算法...")
    logger.info("=" * 60)

    available_denoisers = get_available_denoisers()
    logger.info(f"可用降噪算法: {[d['name'] for d in available_denoisers]}")

    # 预注册的算法列表（按优先级排序）
    # 轻量模型（<200MB）可以在启动时预加载
    lightweight_models = [
        "clearvoice_frcrn_se_16k",    # 154MB, 16kHz实时增强
        "clearvoice_mossformer_gan_se_16k",  # 131MB, 16kHz GAN增强
    ]

    # 大型模型（>500MB）仅在首次使用时加载
    # "clearvoice_mossformer2_se_48k"  # 212MB, 48kHz
    # "clearvoice_mossformer2_ss_16k"  # 640MB, 语音分离
    # "clearvoice_mossformer2_sr_48k"  # 2.1GB, 超分辨率

    # 只预加载轻量模型
    for name in lightweight_models:
        try:
            if name not in _denoisers:
                logger.info(f"  初始化 {name}...")
                denoiser = DenoiserRegistry.get(
                    name, device="cuda" if settings.cuda.enabled else "cpu"
                )
                if denoiser:
                    denoiser.initialize()
                    _denoisers[name] = denoiser
                    logger.info(f"  ✓ {name} 初始化成功")
        except Exception as e:
            logger.warning(f"  ✗ {name} 初始化失败: {e}")

    # 为未加载的算法创建占位条目（延迟加载标记）
    for denoiser_info in available_denoisers:
        name = denoiser_info["name"]
        if name not in _denoisers:
            _denoisers[name] = None  # 延迟加载标记

    logger.info("=" * 60)
    logger.info(
        f"[降噪算法初始化] 完成! "
        f"预加载: {sum(1 for v in _denoisers.values() if v is not None)}, "
        f"延迟加载: {sum(1 for v in _denoisers.values() if v is None)}"
    )
    logger.info("=" * 60)


# 数据模型
class DenoiseAlgorithmInfo(BaseModel):
    """降噪算法信息"""
    name: str
    description: str
    type: str
    pros: List[str] = []
    cons: List[str] = []
    initialized: bool = False
    sample_rate: int = 16000
    task: str = "denoise"
    downloaded: bool = False


class DenoiseTaskCreate(BaseModel):
    """创建降噪测评任务请求"""
    algorithms: List[str]
    has_reference: bool = False


class DenoiseTaskResponse(BaseModel):
    """降噪任务响应"""
    task_id: str
    status: str
    progress: int
    message: str
    result_file: Optional[str] = None
    created_at: str
    updated_at: str


class DenoiseResultItem(BaseModel):
    """单条降噪结果"""
    file_name: str
    algorithm_name: str
    pesq: Optional[float] = None
    stoi: Optional[float] = None
    sisdr: Optional[float] = None
    dnsmos_ovrl: Optional[float] = None
    processing_time: float = 0.0
    rtf: Optional[float] = None


@router.get("/algorithms", response_model=List[DenoiseAlgorithmInfo])
async def list_algorithms(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> List[DenoiseAlgorithmInfo]:
    """
    获取所有可用的降噪算法列表

    Returns:
        降噪算法信息列表
    """
    if not DENOISE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="降噪模块不可用"
        )

    from denoise.registry import DENOISER_DESCRIPTIONS

    algorithms = []
    for name, info in DENOISER_DESCRIPTIONS.items():
        # 确定任务类型和采样率
        task = "denoise"
        sample_rate = 16000

        # 根据算法名称判断任务类型
        if "ss_16k" in name or "separation" in name.lower():
            task = "separation"
        elif "sr_48k" in name or "super_resolution" in name.lower():
            task = "super_resolution"
        elif "enhancement" in name.lower() or "_se_" in name:
            task = "denoise"

        # 根据算法名称判断采样率
        if "48k" in name.lower():
            sample_rate = 48000
        elif "16k" in name.lower():
            sample_rate = 16000

        # 检查模型是否已下载（直接检查文件系统）
        downloaded = False
        try:
            import os as _os
            # 根据算法名称确定checkpoint目录
            model_name_map = {
                "clearvoice_frcrn_se_16k": "FRCRN_SE_16K",
                "clearvoice_mossformer2_se_48k": "MossFormer2_SE_48K",
                "clearvoice_mossformer_gan_se_16k": "MossFormerGAN_SE_16K",
                "clearvoice_mossformer2_ss_16k": "MossFormer2_SS_16K",
                "clearvoice_mossformer2_sr_48k": "MossFormer2_SR_48K",
                "clearervoice_frcrn": "FRCRN_SE_16K",
                "clearervoice_mossformer": "MossFormer2_SE_48K",
                "clearervoice_mossformer2": "MossFormer2_SE_48K",
            }
            ckpt_name = model_name_map.get(name)
            if ckpt_name:
                best_path = _os.path.join(
                    project_root, "models/clearvoice", ckpt_name, "last_best_checkpoint"
                )
                downloaded = _os.path.isfile(best_path)
        except Exception:
            pass

        # 确定初始化状态
        initialized = False
        denoiser_obj = _denoisers.get(name)
        if denoiser_obj is not None and hasattr(denoiser_obj, 'is_initialized'):
            initialized = denoiser_obj.is_initialized()

        algorithms.append(DenoiseAlgorithmInfo(
            name=name,
            description=info.get("description", ""),
            type=info.get("type", "未知"),
            pros=info.get("pros", []),
            cons=info.get("cons", []),
            initialized=initialized,
            sample_rate=sample_rate,
            task=task,
            downloaded=downloaded,
        ))

    return algorithms


@router.post("/upload", response_model=dict)
async def upload_files(
    current_user: Annotated[User, Depends(get_current_active_user)],
    files: List[UploadFile] = File(...),
    reference_files: Optional[List[UploadFile]] = File(None),
    algorithms: str = Form(...)
) -> dict:
    """
    上传音频文件进行降噪测评
    
    Args:
        files: 带噪音频文件列表
        reference_files: 参考音频文件列表(可选)
        algorithms: 选择的算法列表(JSON字符串)
        
    Returns:
        任务创建信息
    """
    if not DENOISE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="降噪模块不可用"
        )
    
    task_id = str(uuid.uuid4())
    task_upload_dir = Path(settings.paths.upload_dir) / "denoise" / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析算法列表
    import json
    try:
        selected_algorithms = json.loads(algorithms)
    except:
        raise HTTPException(
            status_code=400,
            detail="算法参数格式错误"
        )
    
    # 保存带噪音频
    uploaded_files = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.audio.supported_formats:
            continue
        
        file_path = task_upload_dir / "noisy" / file.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        uploaded_files.append(str(file_path))
    
    # 保存参考音频
    ref_files = []
    if reference_files:
        for file in reference_files:
            ext = Path(file.filename).suffix.lower()
            if ext not in settings.audio.supported_formats:
                continue
            
            file_path = task_upload_dir / "ref" / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            ref_files.append(str(file_path))
    
    if not uploaded_files:
        raise HTTPException(
            status_code=400,
            detail="没有有效的音频文件"
        )
    
    # 初始化任务状态
    now = datetime.now().isoformat()
    denoise_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "文件上传完成，等待处理",
        "uploaded_files": uploaded_files,
        "reference_files": ref_files,
        "algorithms": selected_algorithms,
        "result_file": None,
        "created_at": now,
        "updated_at": now,
        "user": current_user.username
    }
    
    logger.info(f"降噪任务创建: {task_id}, 用户: {current_user.username}, 算法: {selected_algorithms}")
    
    return {
        "task_id": task_id,
        "files": [Path(f).name for f in uploaded_files],
        "reference_files": [Path(f).name for f in ref_files],
        "algorithms": selected_algorithms,
        "message": f"成功上传 {len(uploaded_files)} 个文件"
    }


@router.post("/process/{task_id}")
async def start_process(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    提交降噪测评任务到队列
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务提交信息
    """
    if task_id not in denoise_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 检查任务是否已在队列中
    existing_task = await task_queue.get_task(task_id)
    if existing_task and existing_task.status in [TaskStatus.QUEUED, TaskStatus.PROCESSING]:
        return {
            "task_id": task_id,
            "status": existing_task.status.value,
            "message": "任务已在队列中" if existing_task.status == TaskStatus.QUEUED else "任务正在处理中"
        }
    
    # 创建任务对象
    task_data = denoise_tasks[task_id]
    task = Task(
        task_id=task_id,
        user=current_user.username,
        status=TaskStatus.PENDING,
        progress=0,
        message="等待处理...",
        data=task_data
    )
    
    # 提交到队列
    success = await task_queue.submit(task)
    if not success:
        raise HTTPException(status_code=400, detail="任务提交失败")
    
    # 更新本地任务状态
    denoise_tasks[task_id]["status"] = "queued"
    denoise_tasks[task_id]["message"] = f"已加入队列，当前排队位置: {task_queue.get_queue_size()}"
    denoise_tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    return {
        "task_id": task_id,
        "status": "queued",
        "queue_position": task_queue.get_queue_size(),
        "message": f"任务已加入队列"
    }


async def process_denoise_task(queue_task):
    """
    处理降噪测评任务的核心逻辑
    
    Args:
        queue_task: 队列任务对象
    """
    task_id = queue_task.task_id
    
    try:
        task_data = queue_task.data
        noisy_files = task_data.get("uploaded_files", [])
        ref_files = task_data.get("reference_files", [])
        algorithms = task_data.get("algorithms", [])
        
        # 创建输出目录
        output_dir = Path(settings.paths.result_dir) / "denoise" / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化批量测评器
        evaluator = BatchEvaluator(output_dir=str(output_dir))
        
        all_results = {}
        total_algorithms = len(algorithms)
        
        for idx, algo_name in enumerate(algorithms):
            progress = int((idx / total_algorithms) * 80)
            await update_denoise_task_progress(
                task_id, progress, 
                f"正在测评算法 {idx+1}/{total_algorithms}: {algo_name}..."
            )
            
            try:
                results = evaluator.evaluate_algorithm(
                    algo_name,
                    noisy_files,
                    ref_files if ref_files else None,
                    output_subdir=task_id
                )
                all_results[algo_name] = results
            except Exception as e:
                logger.error(f"算法 {algo_name} 测评失败: {e}")
                all_results[algo_name] = []
        
        # 生成报告
        await update_denoise_task_progress(task_id, 90, "正在生成报告...")
        
        report_gen = ReportGenerator(output_dir=str(output_dir))
        
        # 生成Excel报告
        excel_file = report_gen.generate_excel_report(all_results)
        
        # 生成HTML报告
        html_file = report_gen.generate_html_report(all_results)
        
        # 生成Markdown报告
        md_file = report_gen.generate_markdown_report(all_results)
        
        # 更新任务状态
        denoise_tasks[task_id]["status"] = "completed"
        denoise_tasks[task_id]["progress"] = 100
        denoise_tasks[task_id]["message"] = "处理完成"
        denoise_tasks[task_id]["result_file"] = excel_file
        denoise_tasks[task_id]["reports"] = {
            "excel": excel_file,
            "html": html_file,
            "markdown": md_file
        }
        denoise_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"降噪任务完成: {task_id}")
        
    except Exception as e:
        error_msg = str(e)
        denoise_tasks[task_id]["status"] = "failed"
        denoise_tasks[task_id]["message"] = f"处理失败: {error_msg}"
        denoise_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        logger.error(f"降噪任务失败 {task_id}: {e}")


async def update_denoise_task_progress(task_id: str, progress: int, message: str):
    """更新降噪任务进度"""
    if task_id in denoise_tasks:
        denoise_tasks[task_id]["progress"] = progress
        denoise_tasks[task_id]["message"] = message
        denoise_tasks[task_id]["updated_at"] = datetime.now().isoformat()
        logger.info(f"降噪任务进度 {task_id}: {progress}% - {message}")


@router.get("/tasks/{task_id}", response_model=DenoiseTaskResponse)
async def get_task_status(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> DenoiseTaskResponse:
    """
    获取降噪任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务状态信息
    """
    if task_id not in denoise_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = denoise_tasks[task_id]
    return DenoiseTaskResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        result_file=task.get("result_file"),
        created_at=task["created_at"],
        updated_at=task["updated_at"]
    )


@router.get("/tasks")
async def list_tasks(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> List[dict]:
    """
    获取所有降噪任务列表
    
    Returns:
        任务列表
    """
    user_tasks = [
        task for task in denoise_tasks.values()
        if task.get("user") == current_user.username
    ]
    return user_tasks


@router.get("/download/{task_id}")
async def download_result(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    format: str = "excel",
) -> FileResponse:
    """
    下载降噪测评报告
    
    Args:
        task_id: 任务ID
        format: 报告格式 (excel/html/markdown)
        
    Returns:
        报告文件
    """
    if task_id not in denoise_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = denoise_tasks[task_id]
    if task.get("user") != current_user.username:
        raise HTTPException(status_code=403, detail="无权访问此任务")
    
    reports = task.get("reports", {})
    
    if format == "html":
        file_path = reports.get("html")
        filename = f"降噪测评报告_{task_id[:8]}.html"
        media_type = "text/html"
    elif format == "markdown":
        file_path = reports.get("markdown")
        filename = f"降噪测评报告_{task_id[:8]}.md"
        media_type = "text/markdown"
    else:
        file_path = reports.get("excel")
        filename = f"降噪测评报告_{task_id[:8]}.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="报告文件不存在")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type=media_type
    )


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    删除降噪任务及其相关文件
    
    Args:
        task_id: 任务ID
        
    Returns:
        删除结果
    """
    if task_id not in denoise_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = denoise_tasks[task_id]
    if task.get("user") != current_user.username:
        raise HTTPException(status_code=403, detail="无权删除此任务")
    
    # 清理文件
    task_upload_dir = Path(settings.paths.upload_dir) / "denoise" / task_id
    if task_upload_dir.exists():
        shutil.rmtree(task_upload_dir)
    
    result_dir = Path(settings.paths.result_dir) / "denoise" / task_id
    if result_dir.exists():
        shutil.rmtree(result_dir)
    
    del denoise_tasks[task_id]
    
    logger.info(f"降噪任务删除: {task_id}, 用户: {current_user.username}")
    
    return {"message": "任务已删除", "task_id": task_id}


@router.post("/demo")
async def run_demo(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    运行降噪算法演示
    使用系统内置的测试音频进行对比

    Returns:
        演示结果
    """
    if not DENOISE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="降噪模块不可用"
        )

    # 这里可以实现演示逻辑
    # 例如使用预设的噪声音频进行对比测试

    return {
        "message": "演示功能开发中",
        "available_algorithms": list(_denoisers.keys())
    }


@router.post("/denoise-single")
async def denoise_single_file(
    current_user: Annotated[User, Depends(get_current_active_user)],
    file: UploadFile = File(...),
    algorithm: str = Form(...),
) -> FileResponse:
    """
    单文件降噪处理接口
    选择一个算法对单个音频文件进行降噪，直接返回降噪后的音频文件

    Args:
        file: 带噪音频文件
        algorithm: 降噪算法名称

    Returns:
        降噪后的WAV音频文件
    """
    if not DENOISE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="降噪模块不可用"
        )

    # 验证文件格式
    ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if ext not in settings.audio.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {ext}，支持: {settings.audio.supported_formats}"
        )

    # 验证算法是否存在，支持延迟加载
    denoiser = _denoisers.get(algorithm)
    if denoiser is None:
        # 延迟加载：首次使用时才创建和初始化
        denoiser = DenoiserRegistry.get(
            algorithm, device="cuda" if settings.cuda.enabled else "cpu"
        )
        if denoiser is None:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的降噪算法: {algorithm}"
            )

    # 初始化算法（如果尚未初始化）
    if not denoiser.is_initialized():
        if not denoiser.initialize():
            raise HTTPException(
                status_code=500,
                detail=f"降噪算法 {algorithm} 初始化失败"
            )
    _denoisers[algorithm] = denoiser

    import tempfile
    import soundfile as sf
    import numpy as np

    task_id = str(uuid.uuid4())
    temp_dir = Path(settings.paths.temp_dir) / "denoise_single" / task_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 保存上传文件
        input_path = temp_dir / f"input{ext}"
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # 读取音频
        audio, sr = sf.read(str(input_path))

        # 执行降噪
        result = denoiser.denoise(audio, sr)

        # 保存降噪结果
        output_path = temp_dir / "output.wav"
        sf.write(str(output_path), result.audio, result.sample_rate)

        logger.info(
            f"单文件降噪完成: algorithm={algorithm}, "
            f"file={file.filename}, time={result.processing_time:.3f}s"
        )

        # 返回降噪后的音频文件（使用 background 清理临时文件）
        from fastapi import BackgroundTasks

        # 返回降噪后的音频文件
        return FileResponse(
            str(output_path),
            media_type="audio/wav",
            filename=f"denoised_{algorithm}_{file.filename or 'audio'}.wav",
            headers={
                "X-Processing-Time": str(round(result.processing_time, 3)),
                "X-Algorithm": algorithm,
            },
        )

    except Exception as e:
        logger.error(f"单文件降噪失败: {e}", exc_info=True)
        # 清理临时文件
        import shutil as _shutil
        try:
            _shutil.rmtree(str(temp_dir), ignore_errors=True)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"降噪处理失败: {str(e)}"
        )
