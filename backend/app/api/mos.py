"""
MOS评分API路由 (优化版)
提供音频上传、处理和结果下载接口
优化内容:
1. 使用并行计算优化MOS评分速度
2. 增加性能监控和耗时分析
3. 优化音频预处理流程
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

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.auth import get_current_active_user
from app.core.security import User
from app.core.config import settings
from app.core.logging_config import logger
from app.core.task_queue import task_queue, Task, TaskStatus

# 创建线程池用于执行同步任务
executor = ThreadPoolExecutor(max_workers=4)

# 导入优化版MOS计算模块
import sys
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'app', 'algorithms'))
sys.path.insert(0, os.path.join(project_root, 'app', 'core'))

# 优先使用优化版,如果失败则回退到原版
try:
    from tcf.tcf_calculator import (
        compute_mos_scores_optimized,
        get_performance_report,
        reset_performance_tracking,
        parallel_compute
    )
    USE_OPTIMIZED = True
    logger.info("✓ 使用优化版MOS计算模块")
except ImportError as e:
    logger.warning(f"优化版MOS模块加载失败,使用原版: {e}")
    from mos_calculator import (
        ToneColorFidelityScore, DNSMOScore, NisqaMosScore,
        RefScore, WerScore, ScoreqScore, can_convert_to_float
    )
    USE_OPTIMIZED = False

try:
    from audio_cut import cut_all_audio_files_from_list
    from audio_align import align_splited_wav_from_list
except ImportError:
    try:
        from app.core.audio_cut import cut_all_audio_files_from_list
        from app.core.audio_processor import align_splited_wav_from_list
    except ImportError:
        from audio_cut import cut_all_audio_files_from_list
        from audio_processor import align_splited_wav_from_list
import numpy as np
import pandas as pd


router = APIRouter(prefix="/mos", tags=["MOS评分"])

# 确保目录存在
Path(settings.paths.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.paths.result_dir).mkdir(parents=True, exist_ok=True)
Path(settings.paths.temp_dir).mkdir(parents=True, exist_ok=True)

# 任务存储
tasks = {}

# 模型实例 (仅原版使用)
models = {}

# 性能统计
performance_stats = {}


def init_models():
    """初始化所有评分模型"""
    global models
    
    logger.info("=" * 60)
    logger.info("[模型初始化] 开始加载MOS评分模型...")
    logger.info("=" * 60)
    
    if USE_OPTIMIZED:
        logger.info("[模型初始化] 使用优化版MOS计算模块")
        logger.info("[模型初始化] 正在初始化并行计算模型...")
        try:
            start_time = time.time()
            parallel_compute.init_models()
            elapsed = time.time() - start_time
            logger.info(f"✅ 优化版模型初始化完成 (耗时: {elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"❌ 优化版模型初始化失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
        logger.info("=" * 60)
        return
    
    logger.info("[模型初始化] 使用标准版MOS计算模块")
    
    # 核心模型（必须）
    logger.info("[模型初始化] 加载核心模型...")
    
    try:
        logger.info("  [1/5] 正在加载 RefScore 模型...")
        start_time = time.time()
        models["ref_score"] = RefScore()
        elapsed = time.time() - start_time
        logger.info(f"  ✅ RefScore 加载完成 (耗时: {elapsed:.2f}s)")
    except Exception as e:
        logger.warning(f"  ⚠️ RefScore 加载失败: {e}")
    
    try:
        logger.info("  [2/5] 正在加载 DNSMOScore 模型...")
        start_time = time.time()
        models["dnsmos"] = DNSMOScore()
        elapsed = time.time() - start_time
        logger.info(f"  ✅ DNSMOScore 加载完成 (耗时: {elapsed:.2f}s)")
    except Exception as e:
        logger.warning(f"  ⚠️ DNSMOScore 加载失败: {e}")
    
    # 可选模型
    logger.info("[模型初始化] 加载可选模型...")
    
    try:
        logger.info("  [3/5] 正在加载 WerScore 模型...")
        start_time = time.time()
        models["wer"] = WerScore()
        elapsed = time.time() - start_time
        logger.info(f"  ✅ WerScore 加载完成 (耗时: {elapsed:.2f}s)")
    except Exception as e:
        logger.warning(f"  ⚠️ WerScore 加载失败: {e} (WER功能将不可用)")
    
    try:
        logger.info("  [4/5] 正在加载 NisqaMosScore 模型...")
        start_time = time.time()
        models["nisqa"] = NisqaMosScore()
        elapsed = time.time() - start_time
        logger.info(f"  ✅ NisqaMosScore 加载完成 (耗时: {elapsed:.2f}s)")
    except Exception as e:
        logger.warning(f"  ⚠️ NisqaMosScore 加载失败: {e} (NISQA功能将不可用)")
    
    try:
        logger.info("  [5/5] 正在加载 ToneColorFidelityScore 模型...")
        start_time = time.time()
        models["tcf"] = ToneColorFidelityScore()
        elapsed = time.time() - start_time
        logger.info(f"  ✅ ToneColorFidelityScore 加载完成 (耗时: {elapsed:.2f}s)")
    except Exception as e:
        logger.warning(f"  ⚠️ ToneColorFidelityScore 加载失败: {e} (音色还原度功能将不可用)")
    
    try:
        logger.info("  [6/6] 正在加载 ScoreqScore 模型...")
        start_time = time.time()
        models["scoreq"] = ScoreqScore()
        elapsed = time.time() - start_time
        logger.info(f"  ✅ ScoreqScore 加载完成 (耗时: {elapsed:.2f}s)")
    except Exception as e:
        logger.warning(f"  ⚠️ ScoreqScore 加载失败: {e} (Scoreq功能将不可用)")
    
    loaded_count = len(models)
    logger.info("=" * 60)
    logger.info(f"[模型初始化] 完成! 成功加载 {loaded_count} 个模型")
    logger.info("=" * 60)


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: str
    progress: int
    message: str
    result_file: Optional[str] = None
    created_at: str
    updated_at: str


class TaskCreateResponse(BaseModel):
    """任务创建响应模型"""
    task_id: str
    files: List[str]
    message: str


@router.post("/upload", response_model=TaskCreateResponse)
async def upload_files(
    current_user: Annotated[User, Depends(get_current_active_user)],
    files: List[UploadFile] = File(...),
    metrics: Optional[str] = Form(None)
) -> TaskCreateResponse:
    """
    上传音频文件

    Args:
        current_user: 当前登录用户
        files: 音频文件列表
        metrics: 计算项目配置(JSON字符串)

    Returns:
        任务创建信息
    """
    task_id = str(uuid.uuid4())
    task_upload_dir = Path(settings.paths.upload_dir) / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    for file in files:
        # 检查文件格式
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.audio.supported_formats:
            continue

        file_path = task_upload_dir / file.filename
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            uploaded_files.append(file.filename)
            logger.info(f"文件上传: {file.filename} (任务: {task_id})")
        except Exception as e:
            logger.error(f"文件上传失败 {file.filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"文件上传失败: {file.filename}"
            )

    if not uploaded_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="没有有效的音频文件(仅支持.wav和.mp3)"
        )

    # 解析计算项目配置
    selected_metrics = None
    if metrics:
        try:
            import json
            selected_metrics = json.loads(metrics)
            logger.info(f"任务 {task_id} 计算项目配置: {selected_metrics}")
        except Exception as e:
            logger.warning(f"解析计算项目配置失败: {e}")

    # 初始化任务状态
    now = datetime.now().isoformat()
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "文件上传完成,等待处理",
        "uploaded_files": uploaded_files,
        "result_file": None,
        "created_at": now,
        "updated_at": now,
        "user": current_user.username,
        "selected_metrics": selected_metrics  # 保存计算项目配置
    }

    logger.info(f"任务创建: {task_id}, 用户: {current_user.username}, 文件数: {len(uploaded_files)}")

    return TaskCreateResponse(
        task_id=task_id,
        files=uploaded_files,
        message=f"成功上传 {len(uploaded_files)} 个文件"
    )


@router.post("/process/{task_id}")
async def start_process(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    提交音频处理任务到队列
    
    Args:
        task_id: 任务ID
        current_user: 当前登录用户
        
    Returns:
        任务提交信息
    """
    if task_id not in tasks:
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
    task_data = tasks[task_id]
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
        raise HTTPException(status_code=400, detail="任务提交失败，可能已存在")
    
    # 更新本地任务状态
    tasks[task_id]["status"] = "queued"
    tasks[task_id]["message"] = f"已加入队列，当前排队位置: {task_queue.get_queue_size()}"
    tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    logger.info(f"任务已提交到队列: {task_id}, 用户: {current_user.username}, 队列大小: {task_queue.get_queue_size()}")
    
    return {
        "task_id": task_id,
        "status": "queued",
        "queue_position": task_queue.get_queue_size(),
        "message": f"任务已加入队列，当前排队位置: {task_queue.get_queue_size()}"
    }


async def process_audio_task(queue_task):
    """
    处理音频任务的核心逻辑 (队列版本)

    Args:
        queue_task: 队列任务对象
    """
    task_id = queue_task.task_id
    
    try:
        # 从队列任务数据中获取任务信息
        task_data = queue_task.data
        task_upload_dir = Path(settings.paths.upload_dir) / task_id

        # 获取上传的文件列表
        input_files = [str(task_upload_dir / f) for f in task_data.get("uploaded_files", [])]

        # 获取用户选择的计算项目
        selected_metrics = task_data.get("selected_metrics", None)
        if selected_metrics:
            logger.info(f"任务 {task_id} 使用自定义计算项目: {selected_metrics}")
        else:
            logger.info(f"任务 {task_id} 使用默认计算项目")

        ref_dir = Path(settings.paths.ref_dir)
        logger.info(f"检查参考音频目录: {ref_dir}")

        # 详细检查参考音频目录状态
        if not ref_dir.exists():
            logger.info(f"参考音频目录不存在: {ref_dir}")
            has_reference = False
        else:
            ref_files = list(ref_dir.iterdir())
            wav_files = [f for f in ref_files if f.suffix.lower() in ['.wav', '.mp3', '.flac']]
            logger.info(f"参考音频目录存在，总文件数: {len(ref_files)}, 音频文件数: {len(wav_files)}")
            if wav_files:
                logger.info(f"参考音频文件列表: {[f.name for f in wav_files[:5]]}")  # 只显示前5个
            has_reference = len(wav_files) > 0

        # 根据用户选择判断是否需要参考音频
        need_ref_metrics = selected_metrics and any(m in ['pesq', 'stoi', 'sisdr', 'wer', 'tcf'] for m in selected_metrics)

        if has_reference and need_ref_metrics:
            # 有参考音频且用户选择了有参考指标：执行切分、对齐流程
            # Step 1: 音频切分
            await update_task_progress(task_id, 10, "正在切分音频...")
            split_output_dir = Path(settings.paths.temp_dir) / f"{task_id}_split"
            split_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                split_files = cut_all_audio_files_from_list(
                    input_files,
                    output_dir=str(split_output_dir),
                    ref_dir=str(ref_dir)
                )
            except Exception as e:
                logger.error(f"音频切分失败: {e}")
                split_files = input_files  # 如果切分失败,使用原始文件

            # Step 2: 音频对齐
            await update_task_progress(task_id, 30, "正在进行音频对齐...")
            align_output_dir = Path(settings.paths.temp_dir) / f"{task_id}_aligned"
            align_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                logger.info(f"【对齐流程】切分文件数: {len(split_files)}, 参考目录: {ref_dir}")
                for i, f in enumerate(split_files[:3]):  # 只打印前3个
                    logger.info(f"【对齐流程】切分文件 {i+1}: {os.path.basename(f)}")

                aligned_files = align_splited_wav_from_list(
                    split_files,
                    ref_dir=str(ref_dir),
                    output_dir=str(align_output_dir)
                )

                logger.info(f"【对齐流程】对齐完成，对齐文件数: {len(aligned_files)}")
                for i, f in enumerate(aligned_files[:3]):  # 只打印前3个
                    logger.info(f"【对齐流程】对齐文件 {i+1}: {os.path.basename(f)}")
            except Exception as e:
                logger.error(f"音频对齐失败: {e}")
                aligned_files = split_files  # 如果对齐失败,使用切分后的文件

            # Step 3: MOS评分计算（有参考）
            await update_task_progress(task_id, 50, "正在计算MOS得分...")
            # 使用线程池执行同步计算，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor,
                compute_mos_scores_sync,
                aligned_files,
                str(ref_dir),
                True,
                selected_metrics  # 传递计算项目配置
            )
        else:
            # 无参考音频或用户未选择有参考指标：直接处理原始文件
            if not ref_dir.exists():
                logger.info(f"未找到参考音频原因: 目录 {ref_dir} 不存在")
            elif not has_reference:
                logger.info(f"未找到参考音频原因: 目录 {ref_dir} 存在但没有音频文件(.wav/.mp3/.flac)")
            else:
                logger.info(f"用户未选择有参考指标，跳过切分对齐")
            logger.info("执行无参考MOS计算")
            await update_task_progress(task_id, 20, "执行无参考MOS计算...")

            # 直接使用原始文件，不进行切分和对齐
            aligned_files = input_files

            # Step 3: MOS评分计算（无参考）
            await update_task_progress(task_id, 50, "正在计算MOS得分...")
            # 使用线程池执行同步计算，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor,
                compute_mos_scores_sync,
                aligned_files,
                str(ref_dir) if ref_dir.exists() else "",
                False,
                selected_metrics  # 传递计算项目配置
            )
        
        # Step 4: 生成Excel报告
        await update_task_progress(task_id, 90, "正在生成报告...")

        file_names = [os.path.basename(f) for f in aligned_files]
        df_data = {"文件名": file_names}

        logger.info(f"生成报告 - 文件数量: {len(file_names)}, 结果项数量: {len(results)}, 选择指标: {selected_metrics}")

        # 定义指标名称映射（结果键名 -> 显示名称）
        metric_name_map = {
            'pesq': 'PESQ',
            'stoi': 'STOI',
            'sisdr': 'SISDR',
            'wer': 'WER',
            'wcorr': 'WCORR',
            'tcf': '音色还原度',
            'dnsmos': 'DNSMOS',
            'nisqa': 'NISQA',
            'scoreq': 'Scoreq',
            'utmos': 'UTMOS',
            'final_scores': '综合得分'
        }

        # 定义指标分组（选择键名 -> 结果键名列表）
        metric_groups = {
            'dnsmos': ['OVRL', 'SIG', 'BAK', 'P808_MOS'],
            'nisqa': ['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred'],
            'scoreq': ['scoreq'],
            'utmos': ['utmos'],
            'pesq': ['pesq'],
            'stoi': ['STOI'],
            'sisdr': ['SISDR'],
            'wer': ['wer', 'wcorr'],
            'tcf': ['tcf'],
        }

        # 只包含用户选择的指标
        for method, scores in results.items():
            # 跳过未选择的指标（除了文件名和综合得分）
            if method != 'final_scores':
                # 检查是否在选择列表中
                is_selected = False
                if selected_metrics:
                    for selected_key in selected_metrics:
                        # 检查是否是直接匹配
                        if selected_key == method or metric_name_map.get(selected_key) == method:
                            is_selected = True
                            break
                        # 检查组内成员
                        if selected_key in metric_groups and method in metric_groups[selected_key]:
                            is_selected = True
                            break
                else:
                    # 如果没有指定选择列表，包含所有
                    is_selected = True

                if not is_selected:
                    logger.info(f"跳过未选择的指标: {method}")
                    continue

            if isinstance(scores, list):
                if len(scores) != len(file_names):
                    logger.warning(f"报告生成时长度不匹配 - {method}: 期望 {len(file_names)} 个, 实际 {len(scores)} 个")
                    # 补齐或截断
                    if len(scores) < len(file_names):
                        scores = scores + [0.0] * (len(file_names) - len(scores))
                    else:
                        scores = scores[:len(file_names)]
                df_data[method] = scores
            else:
                logger.warning(f"结果项 {method} 不是列表类型: {type(scores)}")
                df_data[method] = [scores] * len(file_names)
        
        df = pd.DataFrame(df_data)
        logger.info(f"DataFrame 形状: {df.shape}, 列: {df.columns.tolist()}")
        
        result_file = Path(settings.paths.result_dir) / f"{task_id}_results.xlsx"
        df.to_excel(result_file, index=False)
        logger.info(f"报告已保存: {result_file}")
        
        # 更新任务状态为完成
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "处理完成"
        tasks[task_id]["result_file"] = str(result_file)
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"任务完成: {task_id}")
        
    except Exception as e:
        error_msg = str(e)
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"处理失败: {error_msg}"
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        logger.error(f"任务失败 {task_id}: {e}")


async def update_task_progress(task_id: str, progress: int, message: str):
    """更新任务进度"""
    if task_id in tasks:
        tasks[task_id]["progress"] = progress
        tasks[task_id]["message"] = message
        tasks[task_id]["updated_at"] = datetime.now().isoformat()
        logger.info(f"任务进度 {task_id}: {progress}% - {message}")


def compute_mos_scores_sync(audio_files: List[str], ref_dir: str, has_reference: bool = True, selected_metrics: Optional[List[str]] = None) -> dict:
    """
    同步计算MOS分数 (支持优化版)

    Args:
        audio_files: 音频文件路径列表
        ref_dir: 参考音频目录
        has_reference: 是否有参考音频
        selected_metrics: 用户选择的计算项目列表

    Returns:
        评分结果字典
    """
    file_num = len(audio_files)
    total_start_time = time.time()

    logger.info("=" * 60)
    logger.info("[MOS计算] 开始计算MOS分数")
    logger.info("=" * 60)
    logger.info(f"[MOS计算] 文件数量: {file_num}")
    logger.info(f"[MOS计算] 优化模式: {USE_OPTIMIZED}")
    logger.info(f"[MOS计算] 参考音频: {'有' if has_reference else '无'}")
    logger.info(f"[MOS计算] 计算项目: {selected_metrics}")

    # 如果没有指定计算项目，使用默认全部
    if selected_metrics is None:
        selected_metrics = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf', 'dnsmos', 'nisqa', 'scoreq', 'utmos']
        logger.info(f"[MOS计算] 使用默认计算项目: {selected_metrics}")

    # 使用优化版计算
    if USE_OPTIMIZED:
        logger.info("[MOS计算] 使用优化版并行计算...")
        try:
            start_time = time.time()
            result = compute_mos_scores_optimized(audio_files, ref_dir, has_reference, selected_metrics)
            elapsed = time.time() - start_time

            # 记录性能统计
            perf_report = get_performance_report()
            logger.info("=" * 60)
            logger.info("[MOS计算] ✅ 优化版计算完成")
            logger.info(f"[MOS计算] 计算耗时: {elapsed:.2f}s")
            logger.info(f"[MOS计算] 平均每文件: {elapsed/file_num:.2f}s")
            logger.info(f"[MOS计算] 性能报告: {perf_report}")
            logger.info("=" * 60)

            # 保存性能统计到全局
            performance_stats['last_report'] = perf_report
            performance_stats['total_time'] = elapsed

            return result
        except Exception as e:
            logger.error(f"[MOS计算] ❌ 优化版计算失败: {e}")
            logger.info("[MOS计算] 回退到原版串行计算...")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 回退到原版计算

    # 原版串行计算逻辑
    logger.info("[MOS计算] 使用原版串行计算...")
    result = {}
    calc_start_time = time.time()

    # 有参考指标
    if has_reference:
        logger.info("[MOS计算] 计算有参考指标...")
        
        # PESQ, STOI, SISDR
        if any(m in selected_metrics for m in ['pesq', 'stoi', 'sisdr']):
            if "ref_score" in models:
                logger.info("  [1/6] 计算参考相关指标(STOI, SISDR, PESQ)...")
                try:
                    start_time = time.time()
                    ref_scores = models["ref_score"].get_mos(audio_files, ref_dir)
                    elapsed = time.time() - start_time
                    # 只保留用户选择的指标
                    if 'pesq' not in selected_metrics:
                        ref_scores.pop('pesq', None)
                    if 'stoi' not in selected_metrics:
                        ref_scores.pop('STOI', None)
                    if 'sisdr' not in selected_metrics:
                        ref_scores.pop('SISDR', None)
                    result.update(ref_scores)
                    logger.info(f"    ✅ 参考指标计算完成 (耗时: {elapsed:.2f}s)")
                except Exception as e:
                    logger.warning(f"    ⚠️ RefScore计算失败: {e}")
                    result.update({"STOI": [0.0]*file_num, "SISDR": [0.0]*file_num, "pesq": [0.0]*file_num})
            else:
                logger.warning("  [1/6] RefScore模型未加载，跳过")
                result.update({"STOI": [0.0]*file_num, "SISDR": [0.0]*file_num, "pesq": [0.0]*file_num})
        else:
            result.update({"STOI": [0.0]*file_num, "SISDR": [0.0]*file_num, "pesq": [0.0]*file_num})

        # WER
        if 'wer' in selected_metrics:
            if "wer" in models:
                logger.info("  [2/6] 计算WER...")
                try:
                    start_time = time.time()
                    wer_scores = models["wer"].get_wer(audio_files)
                    elapsed = time.time() - start_time
                    result.update(wer_scores)
                    logger.info(f"    ✅ WER计算完成 (耗时: {elapsed:.2f}s)")
                except Exception as e:
                    logger.warning(f"    ⚠️ WER计算失败: {e}")
                    result.update({"wer": [0.0]*file_num, "wcorr": [0.0]*file_num})
            else:
                logger.warning("  [2/6] WER模型未加载，跳过")
                result.update({"wer": [0.0]*file_num, "wcorr": [0.0]*file_num})
        else:
            result.update({"wer": [0.0]*file_num, "wcorr": [0.0]*file_num})

        # TCF
        if 'tcf' in selected_metrics:
            if "tcf" in models:
                logger.info("  [3/6] 计算音色还原度(TCF)...")
                try:
                    start_time = time.time()
                    tcf_scores = models["tcf"].get_mos(audio_files, ref_dir)
                    elapsed = time.time() - start_time
                    result.update(tcf_scores)
                    logger.info(f"    ✅ TCF计算完成 (耗时: {elapsed:.2f}s)")
                except Exception as e:
                    logger.warning(f"    ⚠️ TCF计算失败: {e}")
                    result.update({"tcf": [0.0]*file_num})
            else:
                logger.warning("  [3/6] TCF模型未加载，跳过")
                result.update({"tcf": [0.0]*file_num})
        else:
            result.update({"tcf": [0.0]*file_num})
    else:
        logger.info("[MOS计算] 无参考音频，跳过参考相关指标")
        result.update({
            "STOI": [0.0]*file_num, "SISDR": [0.0]*file_num, "pesq": [0.0]*file_num,
            "wer": [0.0]*file_num, "wcorr": [0.0]*file_num, "tcf": [0.0]*file_num
        })

    # 无参考指标
    logger.info("[MOS计算] 计算无参考指标...")
    
    # DNSMOS
    if 'dnsmos' in selected_metrics:
        if "dnsmos" in models:
            logger.info("  [4/6] 计算DNSMOS...")
            try:
                start_time = time.time()
                dnsmos_scores = models["dnsmos"].get_mos(audio_files)
                elapsed = time.time() - start_time
                result.update(dnsmos_scores)
                logger.info(f"    ✅ DNSMOS计算完成 (耗时: {elapsed:.2f}s)")
            except Exception as e:
                logger.warning(f"    ⚠️ DNSMOS计算失败: {e}")
                result.update({"OVRL": [0.0]*file_num, "SIG": [0.0]*file_num, "BAK": [0.0]*file_num, "P808_MOS": [0.0]*file_num})
        else:
            logger.warning("  [4/6] DNSMOS模型未加载，跳过")
            result.update({"OVRL": [0.0]*file_num, "SIG": [0.0]*file_num, "BAK": [0.0]*file_num, "P808_MOS": [0.0]*file_num})
    else:
        result.update({"OVRL": [0.0]*file_num, "SIG": [0.0]*file_num, "BAK": [0.0]*file_num, "P808_MOS": [0.0]*file_num})

    # NISQA
    if 'nisqa' in selected_metrics:
        if "nisqa" in models:
            logger.info("  [5/6] 计算NISQA...")
            try:
                start_time = time.time()
                nisqa_scores = models["nisqa"].get_mos(audio_files)
                elapsed = time.time() - start_time
                result.update(nisqa_scores)
                logger.info(f"    ✅ NISQA计算完成 (耗时: {elapsed:.2f}s)")
            except Exception as e:
                logger.warning(f"    ⚠️ NISQA计算失败: {e}")
                result.update({"mos_pred": [0.0]*file_num, "noi_pred": [0.0]*file_num, "dis_pred": [0.0]*file_num,
                              "col_pred": [0.0]*file_num, "loud_pred": [0.0]*file_num})
        else:
            logger.warning("  [5/6] NISQA模型未加载，跳过")
            result.update({"mos_pred": [0.0]*file_num, "noi_pred": [0.0]*file_num, "dis_pred": [0.0]*file_num,
                          "col_pred": [0.0]*file_num, "loud_pred": [0.0]*file_num})
    else:
        result.update({"mos_pred": [0.0]*file_num, "noi_pred": [0.0]*file_num, "dis_pred": [0.0]*file_num,
                      "col_pred": [0.0]*file_num, "loud_pred": [0.0]*file_num})

    # Scoreq
    if 'scoreq' in selected_metrics:
        if "scoreq" in models:
            logger.info("  [6/6] 计算Scoreq...")
            try:
                start_time = time.time()
                scoreq_scores = models["scoreq"].get_mos(audio_files)
                elapsed = time.time() - start_time
                result.update(scoreq_scores)
                logger.info(f"    ✅ Scoreq计算完成 (耗时: {elapsed:.2f}s)")
            except Exception as e:
                logger.warning(f"    ⚠️ Scoreq计算失败: {e}")
                result.update({"scoreq": [0.0]*file_num})
        else:
            logger.warning("  [6/6] Scoreq模型未加载，跳过")
            result.update({"scoreq": [0.0]*file_num})
    else:
        result.update({"scoreq": [0.0]*file_num})

    # UTMOS
    if 'utmos' in selected_metrics:
        logger.info("  [UTMOS] UTMOS在优化版中处理，原版暂不实现")
        result.update({"utmos": [0.0]*file_num})
    else:
        result.update({"utmos": [0.0]*file_num})

    # 计算总耗时
    total_elapsed = time.time() - calc_start_time
    logger.info("=" * 60)
    logger.info("[MOS计算] ✅ 所有指标计算完成")
    logger.info(f"[MOS计算] 总耗时: {total_elapsed:.2f}s")
    logger.info(f"[MOS计算] 平均每文件: {total_elapsed/file_num:.2f}s")
    logger.info("=" * 60)

    # 验证结果长度
    logger.info("[MOS计算] 验证结果数据...")
    for key, value in result.items():
        if isinstance(value, list) and len(value) != file_num:
            logger.warning(f"  ⚠️ 结果列表长度不匹配: {key} 期望 {file_num} 个, 实际 {len(value)} 个")
            if len(value) < file_num:
                value.extend([0.0] * (file_num - len(value)))
            else:
                value = value[:file_num]
            result[key] = value
    logger.info("  ✅ 结果数据验证完成")

    # 计算最终得分
    logger.info("[MOS计算] 计算最终MOS得分...")
    values = np.array(list(result.values())).T
    final_scores = []
    for i in range(len(values)):
        val_tmp = [float(s) if isinstance(s, (int, float)) else 0.0 for s in values[i]]
        if has_reference:
            tmp = np.mean([
                val_tmp[0] * 5, (1 / (1 + np.exp(-val_tmp[1]))) * 5, val_tmp[2],
                val_tmp[3], val_tmp[4], val_tmp[6], (1 - val_tmp[7]) * 5, val_tmp[8] * 5,
                val_tmp[9], val_tmp[10], val_tmp[11], val_tmp[12], val_tmp[13],
                val_tmp[14] * 5, val_tmp[15]
            ])
        else:
            tmp = np.mean([
                val_tmp[6], val_tmp[7], val_tmp[8], val_tmp[9],
                val_tmp[10], val_tmp[11], val_tmp[12], val_tmp[13],
                val_tmp[14], val_tmp[15]
            ])
        final_scores.append(tmp)

    result['final_scores'] = final_scores
    logger.info(f"MOS分数计算完成，最终得分数量: {len(final_scores)}")
    return result


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> TaskResponse:
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        current_user: 当前登录用户
        
    Returns:
        任务状态信息
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    return TaskResponse(
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
    获取所有任务列表
    
    Args:
        current_user: 当前登录用户
        
    Returns:
        任务列表
    """
    # 只返回当前用户的任务
    user_tasks = [
        task for task in tasks.values()
        if task.get("user") == current_user.username
    ]
    return user_tasks


@router.get("/download/{task_id}")
async def download_result(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> FileResponse:
    """
    下载处理结果

    Args:
        task_id: 任务ID
        current_user: 当前登录用户

    Returns:
        Excel文件
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    if task.get("user") != current_user.username:
        raise HTTPException(status_code=403, detail="无权访问此任务")

    if not task.get("result_file"):
        raise HTTPException(status_code=400, detail="结果文件尚未生成")

    result_file = Path(task["result_file"])
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")

    return FileResponse(
        result_file,
        filename=f"MOS评分结果_{task_id[:8]}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@router.get("/results/{task_id}")
async def get_task_results(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    获取任务详细结果数据（用于前端展示）

    Args:
        task_id: 任务ID
        current_user: 当前登录用户

    Returns:
        详细结果数据
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    if task.get("user") != current_user.username:
        raise HTTPException(status_code=403, detail="无权访问此任务")

    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    if not task.get("result_file"):
        raise HTTPException(status_code=400, detail="结果文件尚未生成")

    result_file = Path(task["result_file"])
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")

    try:
        # 读取Excel文件
        import pandas as pd
        df = pd.read_excel(result_file)

        # 转换为JSON格式
        results = df.to_dict(orient='records')

        return {
            "task_id": task_id,
            "status": "completed",
            "results": results,
            "columns": df.columns.tolist(),
            "total_files": len(results)
        }
    except Exception as e:
        logger.error(f"读取结果文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"读取结果文件失败: {str(e)}")


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    删除任务及其相关文件
    
    Args:
        task_id: 任务ID
        current_user: 当前登录用户
        
    Returns:
        删除结果
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    if task.get("user") != current_user.username:
        raise HTTPException(status_code=403, detail="无权删除此任务")
    
    # 清理文件
    task_upload_dir = Path(settings.paths.upload_dir) / task_id
    if task_upload_dir.exists():
        shutil.rmtree(task_upload_dir)
    
    split_dir = Path(settings.paths.temp_dir) / f"{task_id}_split"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    
    align_dir = Path(settings.paths.temp_dir) / f"{task_id}_aligned"
    if align_dir.exists():
        shutil.rmtree(align_dir)
    
    if task.get("result_file"):
        result_file = Path(task["result_file"])
        if result_file.exists():
            result_file.unlink()
    
    del tasks[task_id]
    
    logger.info(f"任务删除: {task_id}, 用户: {current_user.username}")
    
    return {"message": "任务已删除", "task_id": task_id}


# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket
    
    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
    
    async def send_progress(self, task_id: str, data: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(data)
            except Exception:
                # 连接已关闭，移除连接
                self.disconnect(task_id)


manager = ConnectionManager()


@router.websocket("/ws/{task_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    task_id: str
):
    """
    WebSocket实时进度推送
    
    Args:
        websocket: WebSocket连接
        task_id: 任务ID
    """
    await manager.connect(websocket, task_id)
    try:
        while True:
            if task_id in tasks:
                task = tasks[task_id]
                try:
                    await websocket.send_json({
                        "status": task["status"],
                        "progress": task["progress"],
                        "message": task["message"]
                    })
                except Exception:
                    # 连接异常，退出循环
                    break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        manager.disconnect(task_id)


@router.get("/performance")
async def get_performance_metrics(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    获取MOS计算性能统计
    
    Args:
        current_user: 当前登录用户
        
    Returns:
        性能统计信息
    """
    # 获取优化版性能报告
    perf_report = {}
    if USE_OPTIMIZED:
        try:
            perf_report = get_performance_report()
        except Exception as e:
            logger.warning(f"获取性能报告失败: {e}")
    
    return {
        "optimized_mode": USE_OPTIMIZED,
        "performance_report": perf_report,
        "global_stats": performance_stats,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/performance/reset")
async def reset_performance_metrics(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    重置性能统计
    
    Args:
        current_user: 当前登录用户
        
    Returns:
        操作结果
    """
    global performance_stats
    performance_stats = {}
    
    if USE_OPTIMIZED:
        try:
            reset_performance_tracking()
        except Exception as e:
            logger.warning(f"重置性能跟踪失败: {e}")
    
    return {
        "message": "性能统计已重置",
        "timestamp": datetime.now().isoformat()
    }
