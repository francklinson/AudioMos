"""
AudioMOS FastAPI 主应用入口
提供音频质量评估的RESTful API服务
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging_config import logger
from app.api import auth, mos


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
    logger.info(f"  服务地址: {settings.server.backend.host}:{settings.server.backend.port}")
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

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    根路径 - 服务信息
    """
    return {
        "name": "AudioMOS API",
        "version": "1.0.0",
        "description": "音频质量评估系统",
        "docs": "/docs"
    }


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.server.backend.host,
        port=settings.server.backend.port,
        reload=settings.server.debug
    )
