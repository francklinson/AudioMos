"""
AudioMOS FastAPI 主应用入口
提供音频质量评估的RESTful API服务
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging_config import logger
from app.api import auth, mos, denoise, restoration, reference_audio


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
    
    # 初始化降噪算法
    logger.info("[降噪算法初始化]")
    try:
        logger.info("  正在初始化降噪算法...")
        denoise.init_denoisers()
        logger.info("  ✅ 降噪算法初始化成功")
    except Exception as e:
        logger.error(f"  ❌ 降噪算法初始化失败: {e}")
        import traceback
        logger.error(f"  错误详情: {traceback.format_exc()}")

    # 初始化音频修复算法
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
app.include_router(denoise.router, prefix="/api")
app.include_router(restoration.router, prefix="/api")
app.include_router(reference_audio.router, prefix="/api")


# ========== 前后端一体模式：托管前端静态文件 ==========
# 检查是否存在前端构建文件
# 使用绝对路径，确保在任何工作目录下都能正确找到静态文件
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(_PROJECT_ROOT, "backend", "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

if os.path.exists(STATIC_DIR) and os.path.exists(INDEX_HTML):
    logger.info("检测到前端构建文件，启用前后端一体模式")
    
    # 挂载静态文件目录
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")
    
    # 其他静态资源目录
    for static_subdir in ["images", "fonts", "icons"]:
        subdir_path = os.path.join(STATIC_DIR, static_subdir)
        if os.path.exists(subdir_path):
            app.mount(f"/{static_subdir}", StaticFiles(directory=subdir_path), name=static_subdir)
    
    # 根路径返回前端首页
    @app.get("/")
    async def serve_index():
        return FileResponse(INDEX_HTML)
    
    # 所有其他路径也返回前端首页（支持前端路由）
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # 排除 API 路径
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            return {"detail": "Not Found"}
        
        # 检查是否是静态文件请求
        static_file = os.path.join(STATIC_DIR, full_path)
        if os.path.exists(static_file) and os.path.isfile(static_file):
            return FileResponse(static_file)
        
        # 否则返回 index.html（前端路由处理）
        return FileResponse(INDEX_HTML)
    
    logger.info(f"静态文件目录: {STATIC_DIR}")
else:
    logger.info("未检测到前端构建文件，仅 API 模式运行")
    
    # 原有的根路径响应
    @app.get("/")
    async def root():
        return {
            "name": "AudioMOS API",
            "version": "1.0.0",
            "description": "音频质量评估系统",
            "docs": "/docs",
            "mode": "api-only"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.server.backend.host,
        port=settings.server.backend.port,
        reload=settings.server.debug
    )
