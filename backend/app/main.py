"""
AudioMOS FastAPI 主应用入口
提供音频质量评估的RESTful API服务
"""
import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging_config import logger, log_request
from app.api import auth, mos, restoration, reference_audio, restoration_batch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    import time
    start_time = time.time()
    
    # 启动时执行
    logger.info("=" * 60)
    logger.info("AudioMOS 系统启动")
    logger.info("=" * 60)
    
    # 系统信息日志
    logger.info("[系统配置]")
    logger.info(f"  服务地址: {settings.server.host}:{settings.server.port}")
    logger.info(f"  调试模式: {settings.server.debug}")
    logger.info(f"  CUDA启用: {settings.cuda.enabled}")
    if settings.cuda.enabled:
        logger.info(f"  GPU设备ID: {settings.cuda.device_id}")
    logger.info(f"  参考音频目录: {settings.paths.ref_dir}")
    logger.info(f"  上传目录: {settings.paths.upload_dir}")
    logger.info(f"  结果目录: {settings.paths.result_dir}")
    
    # 检查目录是否存在
    logger.info("[目录检查]")
    for dir_name, dir_path in [
        ("参考音频", settings.paths.ref_dir),
        ("上传文件", settings.paths.upload_dir),
        ("结果文件", settings.paths.result_dir),
    ]:
        if os.path.exists(dir_path):
            logger.info(f"  ✅ {dir_name}: {dir_path}")
        else:
            logger.warning(f"  ⚠️  {dir_name}: {dir_path} (不存在)")
    
    # 初始化MOS模型
    logger.info("[模型初始化]")
    try:
        logger.info("  正在初始化MOS模型...")
        mos.init_models()
        logger.info("  ✅ MOS模型初始化成功")
    except Exception as e:
        logger.error(f"  ❌ MOS模型初始化失败: {e}")
        import traceback
        logger.error(f"  错误详情: {traceback.format_exc()}")
    
    # 初始化音频修复算法（合并降噪后统一初始化）
    logger.info("[音频修复算法初始化]")
    try:
        logger.info("  正在初始化音频修复算法...")
        restoration.init_restoration()
        logger.info("  ✅ 音频修复算法初始化成功")
    except Exception as e:
        logger.error(f"  ❌ 音频修复算法初始化失败: {e}")
        import traceback
        logger.error(f"  错误详情: {traceback.format_exc()}")
    
    # 启动任务队列
    logger.info("[任务队列]")
    try:
        logger.info("  正在启动任务队列...")
        from app.core.task_queue import task_queue
        from app.api.mos import process_audio_task
        await task_queue.start(process_audio_task)
        logger.info("  ✅ 任务队列启动成功")
    except Exception as e:
        logger.error(f"  ❌ 任务队列启动失败: {e}")
        import traceback
        logger.error(f"  错误详情: {traceback.format_exc()}")

    # GPU设备初始化(多卡部署支持)
    logger.info("[GPU设备配置]")
    try:
        import torch
        import os
        
        # 从配置读取GPU ID
        gpu_id = settings.cuda.device_id
        
        # 验证CUDA可用性
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            
            logger.info(f"  检测到 {device_count} 张GPU")
            
            # 验证GPU ID有效性
            if gpu_id >= device_count:
                logger.warning(
                    f"  ⚠️ 配置的GPU ID({gpu_id})超出范围(0-{device_count-1}), "
                    f"自动调整为GPU 0"
                )
                gpu_id = 0
                settings.cuda.device_id = 0
            
            # 设置默认设备
            torch.cuda.set_device(gpu_id)
            
            # 获取设备信息
            device_name = torch.cuda.get_device_name(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            compute_capability = torch.cuda.get_device_properties(gpu_id).major
            
            logger.info(f"  ✅ GPU初始化成功:")
            logger.info(f"     配置GPU ID: {gpu_id}")
            logger.info(f"     设备名称: {device_name}")
            logger.info(f"     显存容量: {total_memory:.1f} GB")
            logger.info(f"     计算能力: {compute_capability.x}")
            
            # 显存限制(可选)
            if settings.cuda.memory_fraction is not None:
                try:
                    torch.cuda.set_per_process_memory_fraction(
                        settings.cuda.memory_fraction, 
                        gpu_id
                    )
                    logger.info(
                        f"     显存限制: {settings.cuda.memory_fraction*100:.1f}% "
                        f"(约 {total_memory * settings.cuda.memory_fraction:.1f} GB)"
                    )
                except Exception as e:
                    logger.warning(f"     ⚠️ 显存限制设置失败: {e}")
            
            # 当前显存占用验证
            free_memory = torch.cuda.memory_reserved(gpu_id) / 1024**3
            logger.info(f"     当前占用: {free_memory:.3f} GB")
            
        else:
            logger.warning("  ⚠️ CUDA不可用,服务将使用CPU模式运行")
            logger.warning("     性能将大幅降低,建议检查CUDA安装")
            gpu_id = -1  # CPU模式标记
            
    except Exception as e:
        logger.error(f"  ❌ GPU初始化失败: {e}")
        import traceback
        logger.error(f"  错误详情: {traceback.format_exc()}")
        logger.warning("  将尝试使用CPU模式运行")

    # 启动音频修复任务队列（与 MOS 队列隔离，避免互相阻塞）
    logger.info("[音频修复任务队列]")
    try:
        logger.info("  正在启动音频修复任务队列...")
        from app.api.restoration import restoration_task_queue, process_restoration_task
        await restoration_task_queue.start(process_restoration_task)
        logger.info("  ✅ 音频修复任务队列启动成功")
    except Exception as e:
        logger.error(f"  ❌ 音频修复任务队列启动失败: {e}")
        import traceback
        logger.error(f"  错误详情: {traceback.format_exc()}")

    # 启动GPU显存监控守护线程
    logger.info("[GPU显存监控]")
    try:
        logger.info("  正在启动GPU显存监控守护线程...")
        from app.core.gpu_monitor import init_gpu_monitor
        
        # 传入GPU ID和阈值配置
        gpu_monitor = init_gpu_monitor(
            gpu_id=settings.cuda.device_id  # 使用配置的GPU ID
        )
        
        # 更新阈值配置(覆盖默认值)
        gpu_monitor.warning_threshold_mb = settings.cuda.warning_threshold_mb
        gpu_monitor.critical_threshold_mb = settings.cuda.critical_threshold_mb
        
        gpu_monitor.start()
        app.state.gpu_monitor = gpu_monitor  # 保存到app.state以便shutdown时停止
        
        logger.info(
            f"  ✅ GPU显存监控线程启动成功 "
            f"(GPU {settings.cuda.device_id}, "
            f"警告阈值={settings.cuda.warning_threshold_mb}MB, "
            f"严重阈值={settings.cuda.critical_threshold_mb}MB)"
        )
    except Exception as e:
        logger.error(f"  ❌ GPU显存监控启动失败: {e}")
        import traceback
        logger.error(f"  错误详情: {traceback.format_exc()}")

    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"系统启动完成 (耗时: {elapsed_time:.2f}s)")
    logger.info("=" * 60)
    
    yield
    
    # 关闭时执行
    logger.info("=" * 60)
    logger.info("AudioMOS 系统关闭中...")
    logger.info("=" * 60)
    
    try:
        from app.core.task_queue import task_queue
        await task_queue.stop()
        logger.info("  ✅ 任务队列已停止")
    except Exception as e:
        logger.error(f"  ❌ 任务队列停止失败: {e}")

    try:
        from app.api.restoration import restoration_task_queue
        await restoration_task_queue.stop()
        logger.info("  ✅ 音频修复任务队列已停止")
    except Exception as e:
        logger.error(f"  ❌ 音频修复任务队列停止失败: {e}")

    try:
        if hasattr(app.state, "gpu_monitor"):
            app.state.gpu_monitor.stop()
            logger.info("  ✅ GPU显存监控线程已停止")
    except Exception as e:
        logger.error(f"  ❌ GPU显存监控停止失败: {e}")

    logger.info("=" * 60)
    logger.info("AudioMOS 系统已关闭")
    logger.info("=" * 60)


# 创建FastAPI应用
app = FastAPI(
    title="AudioMOS API",
    description="音频质量评估系统后端API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS — 从环境变量读取允许的来源，默认为全部
# 生产环境应设置: export AUDIOMOS_CORS_ORIGINS="https://your-domain.com"
_cors_origins_str = os.getenv("AUDIOMOS_CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_str.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 请求耗时日志中间件 ==========
@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    """记录每个HTTP请求的耗时"""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    # 只记录API请求，跳过静态文件
    path = request.url.path
    if path.startswith("/api/") or path.startswith("/health"):
        log_request(
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
    return response


@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "service": "audiomos-api",
        "version": "1.0.0"
    }


# 注册路由
app.include_router(auth.router, prefix="/api")
app.include_router(mos.router, prefix="/api")
app.include_router(restoration.router, prefix="/api")
app.include_router(restoration_batch.router, prefix="/api")  # 批量处理路由
app.include_router(reference_audio.router, prefix="/api")


# ========== 前后端一体模式：托管前端静态文件 ==========
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(_PROJECT_ROOT, "frontend", "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

logger.info(f"静态文件目录: {STATIC_DIR}")

# 挂载静态资源
app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

# 根路径返回前端首页
@app.get("/")
async def serve_index():
    return FileResponse(INDEX_HTML)

# SPA 回退：非 API 路径返回前端首页
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
        return {"detail": "Not Found"}
    static_file = os.path.join(STATIC_DIR, full_path)
    if os.path.exists(static_file) and os.path.isfile(static_file):
        return FileResponse(static_file)
    return FileResponse(INDEX_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug
    )
