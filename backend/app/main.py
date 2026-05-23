"""
AudioMOS FastAPI 主应用入口
提供音频质量评估的RESTful API服务
"""
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
    # 启动时执行
    logger.info("=" * 50)
    logger.info("AudioMOS 系统启动")
    logger.info("=" * 50)
    logger.info(f"服务地址: {settings.server.host}:{settings.server.port}")
    logger.info(f"CUDA启用: {settings.cuda.enabled}")
    logger.info(f"参考音频目录: {settings.paths.ref_dir}")
    
    # 初始化MOS模型
    try:
        mos.init_models()
        logger.info("MOS模型初始化成功")
    except Exception as e:
        logger.error(f"MOS模型初始化失败: {e}")
    
    # 启动任务队列
    try:
        from app.core.task_queue import task_queue
        from app.api.mos import process_audio_task
        await task_queue.start(process_audio_task)
        logger.info("任务队列启动成功")
    except Exception as e:
        logger.error(f"任务队列启动失败: {e}")
    
    yield
    
    # 关闭时执行
    try:
        from app.core.task_queue import task_queue
        await task_queue.stop()
        logger.info("任务队列已停止")
    except Exception as e:
        logger.error(f"任务队列停止失败: {e}")
    
    logger.info("AudioMOS 系统关闭")


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
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug
    )
