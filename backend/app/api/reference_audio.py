"""
参考音频管理模块API
提供参考音频的上传、查询、删除等功能
用于带参考的MOS分计算
"""
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Annotated

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.auth import get_current_active_user, get_current_user_optional
from app.core.security import User
from app.core.config import settings
from app.core.logging_config import logger

router = APIRouter(prefix="/reference-audio", tags=["参考音频管理"])

# 确保参考音频目录存在
Path(settings.paths.ref_dir).mkdir(parents=True, exist_ok=True)

# 参考音频元数据存储文件
REF_METADATA_FILE = Path(settings.paths.ref_dir) / ".metadata.json"


class ReferenceAudioInfo(BaseModel):
    """参考音频信息模型"""
    id: str
    filename: str
    original_name: str
    file_size: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    description: Optional[str] = None
    created_at: str
    updated_at: str


class ReferenceAudioResponse(BaseModel):
    """参考音频响应模型"""
    id: str
    filename: str
    original_name: str
    file_size: int
    file_size_formatted: str
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    description: Optional[str] = None
    ground_truth_text: Optional[str] = None  # WeNet自动转录 / 预置文本
    created_at: str
    updated_at: str


class ReferenceAudioListResponse(BaseModel):
    """参考音频列表响应"""
    total: int
    items: List[ReferenceAudioResponse]


class ReferenceAudioUpdateRequest(BaseModel):
    """更新参考音频请求"""
    description: Optional[str] = None
    ground_truth_text: Optional[str] = None  # WER评估用的参考文本


def _load_metadata() -> dict:
    """加载参考音频元数据"""
    import json
    if REF_METADATA_FILE.exists():
        try:
            with open(REF_METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载参考音频元数据失败: {e}")
    return {"audios": {}}


def _save_metadata(metadata: dict):
    """保存参考音频元数据"""
    import json
    try:
        with open(REF_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存参考音频元数据失败: {e}")


def _format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _get_audio_info(file_path: Path) -> dict:
    """获取音频文件信息"""
    info = {
        "duration": None,
        "sample_rate": None,
        "channels": None
    }
    try:
        import librosa
        y, sr = librosa.load(str(file_path), sr=None, mono=False)
        info["duration"] = float(librosa.get_duration(y=y, sr=sr))
        info["sample_rate"] = int(sr)
        info["channels"] = 1 if y.ndim == 1 else y.shape[0]
    except Exception as e:
        logger.warning(f"获取音频信息失败 {file_path}: {e}")
    return info


def _to_response(audio_id: str, info: dict) -> ReferenceAudioResponse:
    """转换为响应模型"""
    return ReferenceAudioResponse(
        id=audio_id,
        filename=info.get("filename", ""),
        original_name=info.get("original_name", ""),
        file_size=info.get("file_size", 0),
        file_size_formatted=_format_file_size(info.get("file_size", 0)),
        duration=info.get("duration"),
        sample_rate=info.get("sample_rate"),
        channels=info.get("channels"),
        description=info.get("description"),
        ground_truth_text=info.get("ground_truth_text"),
        created_at=info.get("created_at", ""),
        updated_at=info.get("updated_at", "")
    )


@router.get("/list", response_model=ReferenceAudioListResponse)
async def list_reference_audios(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> ReferenceAudioListResponse:
    """
    获取所有参考音频列表
    """
    metadata = _load_metadata()
    audios = metadata.get("audios", {})

    items = []
    for audio_id, info in audios.items():
        file_path = Path(settings.paths.ref_dir) / info.get("filename", "")
        if file_path.exists():
            items.append(_to_response(audio_id, info))
        else:
            logger.warning(f"参考音频文件不存在: {file_path}")

    items.sort(key=lambda x: x.created_at, reverse=True)

    return ReferenceAudioListResponse(
        total=len(items),
        items=items
    )


@router.post("/upload", response_model=ReferenceAudioResponse)
async def upload_reference_audio(
    current_user: Annotated[User, Depends(get_current_active_user)],
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
) -> ReferenceAudioResponse:
    """
    上传参考音频文件
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.audio.supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式: {ext}，仅支持 {', '.join(settings.audio.supported_formats)}"
        )

    audio_id = str(uuid.uuid4())
    original_name = Path(file.filename).stem

    base_filename = f"{original_name}{ext}"
    file_path = Path(settings.paths.ref_dir) / base_filename
    counter = 1
    while file_path.exists():
        base_filename = f"{original_name}_{counter:03d}{ext}"
        file_path = Path(settings.paths.ref_dir) / base_filename
        counter += 1

    safe_filename = base_filename

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        file_size = len(content)
        logger.info(f"参考音频上传成功: {file.filename} -> {safe_filename} ({_format_file_size(file_size)})")

        audio_info = _get_audio_info(file_path)

        metadata = _load_metadata()
        now = datetime.now().isoformat()

        info = {
            "filename": safe_filename,
            "original_name": file.filename,
            "file_size": file_size,
            "duration": audio_info.get("duration"),
            "sample_rate": audio_info.get("sample_rate"),
            "channels": audio_info.get("channels"),
            "description": description,
            "created_at": now,
            "updated_at": now
        }

        metadata["audios"][audio_id] = info
        _save_metadata(metadata)

        # 自动转录（WeNet），填入 ground_truth_text
        _auto_transcribe_reference(audio_id, str(file_path), metadata)

        # 增量更新指纹数据库
        _incremental_add_to_fingerprint(audio_id, str(file_path),
                                        description=description)

        return _to_response(audio_id, info)

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        logger.error(f"上传参考音频失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传失败: {str(e)}"
        )


@router.post("/upload-batch")
async def upload_reference_audios_batch(
    current_user: Annotated[User, Depends(get_current_active_user)],
    files: List[UploadFile] = File(...)
) -> dict:
    """
    批量上传参考音频文件
    """
    results = {
        "success": [],
        "failed": [],
        "total": len(files)
    }

    metadata = _load_metadata()
    now = datetime.now().isoformat()

    for idx, file in enumerate(files):
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.audio.supported_formats:
            results["failed"].append({
                "filename": file.filename,
                "reason": f"不支持的文件格式: {ext}"
            })
            continue

        audio_id = str(uuid.uuid4())
        original_name = Path(file.filename).stem

        base_filename = f"{original_name}{ext}"
        file_path = Path(settings.paths.ref_dir) / base_filename
        counter = 1
        while file_path.exists():
            base_filename = f"{original_name}_{counter:03d}{ext}"
            file_path = Path(settings.paths.ref_dir) / base_filename
            counter += 1

        safe_filename = base_filename

        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            file_size = len(content)
            audio_info = _get_audio_info(file_path)

            info = {
                "filename": safe_filename,
                "original_name": file.filename,
                "file_size": file_size,
                "duration": audio_info.get("duration"),
                "sample_rate": audio_info.get("sample_rate"),
                "channels": audio_info.get("channels"),
                "description": None,
                "created_at": now,
                "updated_at": now
            }

            metadata["audios"][audio_id] = info
            results["success"].append({
                "id": audio_id,
                "filename": file.filename,
                "file_size": file_size
            })

        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            results["failed"].append({
                "filename": file.filename,
                "reason": str(e)
            })

    _save_metadata(metadata)

    # 自动转录 + 增量更新指纹数据库
    for item in results["success"]:
        audio_id = item["id"]
        file_path = str(Path(settings.paths.ref_dir) / metadata["audios"][audio_id]["filename"])
        _auto_transcribe_reference(audio_id, file_path, metadata)

    _incremental_batch_add_to_fingerprint(results["success"], metadata)

    logger.info(f"批量上传参考音频完成: 成功 {len(results['success'])}, 失败 {len(results['failed'])}")

    return {
        "message": f"上传完成: 成功 {len(results['success'])}, 失败 {len(results['failed'])}",
        "results": results
    }


@router.get("/detail/{audio_id}", response_model=ReferenceAudioResponse)
async def get_reference_audio(
    audio_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> ReferenceAudioResponse:
    """获取单个参考音频信息"""
    metadata = _load_metadata()
    audios = metadata.get("audios", {})

    if audio_id not in audios:
        raise HTTPException(status_code=404, detail="参考音频不存在")

    info = audios[audio_id]
    file_path = Path(settings.paths.ref_dir) / info.get("filename", "")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="参考音频文件不存在")

    return _to_response(audio_id, info)


@router.get("/download/{audio_id}")
async def download_reference_audio(
    audio_id: str,
    current_user: Annotated[User, Depends(get_current_user_optional)]
) -> FileResponse:
    """下载参考音频文件"""
    metadata = _load_metadata()
    audios = metadata.get("audios", {})

    if audio_id not in audios:
        raise HTTPException(status_code=404, detail="参考音频不存在")

    info = audios[audio_id]
    file_path = Path(settings.paths.ref_dir) / info.get("filename", "")
    original_name = info.get("original_name", f"reference_{audio_id}.wav")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="参考音频文件不存在")

    return FileResponse(
        file_path,
        filename=original_name,
        media_type="audio/wav"
    )


@router.put("/update/{audio_id}", response_model=ReferenceAudioResponse)
async def update_reference_audio(
    audio_id: str,
    request: ReferenceAudioUpdateRequest,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> ReferenceAudioResponse:
    """更新参考音频信息（描述、ground truth文本）"""
    metadata = _load_metadata()
    audios = metadata.get("audios", {})

    if audio_id not in audios:
        raise HTTPException(status_code=404, detail="参考音频不存在")

    info = audios[audio_id]
    now = datetime.now().isoformat()

    if request.description is not None:
        info["description"] = request.description

    if request.ground_truth_text is not None:
        info["ground_truth_text"] = request.ground_truth_text
        logger.info(f"更新参考音频 {audio_id} 的ground truth文本")

    info["updated_at"] = now
    _save_metadata(metadata)

    return _to_response(audio_id, info)


@router.delete("/delete/{audio_id}")
async def delete_reference_audio(
    audio_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """删除参考音频"""
    metadata = _load_metadata()
    audios = metadata.get("audios", {})

    if audio_id not in audios:
        raise HTTPException(status_code=404, detail="参考音频不存在")

    info = audios[audio_id]
    file_path = Path(settings.paths.ref_dir) / info.get("filename", "")

    if file_path.exists():
        file_path.unlink()
        logger.info(f"删除参考音频文件: {file_path}")

    del audios[audio_id]
    _save_metadata(metadata)

    # 增量移除指纹
    _incremental_remove_from_fingerprint(audio_id)

    return {
        "message": "删除成功",
        "audio_id": audio_id,
        "filename": info.get("original_name")
    }


@router.delete("/")
async def delete_all_reference_audios(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """删除所有参考音频"""
    metadata = _load_metadata()
    audios = metadata.get("audios", {})

    deleted_count = 0
    for audio_id, info in list(audios.items()):
        file_path = Path(settings.paths.ref_dir) / info.get("filename", "")
        if file_path.exists():
            file_path.unlink()
            deleted_count += 1

    metadata["audios"] = {}
    _save_metadata(metadata)

    # 全量重建指纹库（清空）
    _rebuild_fingerprint_database()

    logger.info(f"删除所有参考音频: {deleted_count} 个文件")

    return {
        "message": f"已删除 {deleted_count} 个参考音频",
        "deleted_count": deleted_count
    }


@router.get("/check/status")
async def check_reference_audio_status(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """检查参考音频状态"""
    metadata = _load_metadata()
    audios = metadata.get("audios", {})

    valid_count = 0
    total_size = 0
    valid_files = []

    for audio_id, info in audios.items():
        file_path = Path(settings.paths.ref_dir) / info.get("filename", "")
        if file_path.exists():
            valid_count += 1
            total_size += info.get("file_size", 0)
            valid_files.append({
                "id": audio_id,
                "filename": info.get("original_name")
            })

    return {
        "has_reference": valid_count > 0,
        "total_count": len(audios),
        "valid_count": valid_count,
        "total_size": total_size,
        "total_size_formatted": _format_file_size(total_size),
        "files": valid_files
    }


# ============================================================================
# WeNet 自动转录辅助函数
# ============================================================================

# 全局 WeNet 服务实例（延迟初始化）
_wenet_service = None


def _get_wenet_service():
    """获取或初始化 WeNet 服务（单例）"""
    global _wenet_service
    if _wenet_service is None:
        try:
            from app.core.wenet_service import WeNetService
            _wenet_service = WeNetService()
            if _wenet_service.initialize():
                logger.info("[WeNet转录] WeNet 服务初始化成功")
            else:
                logger.warning("[WeNet转录] WeNet 服务初始化失败，自动转录不可用")
                _wenet_service = None
        except ImportError as e:
            logger.warning(f"[WeNet转录] WeNet 模块导入失败: {e}")
        except Exception as e:
            logger.warning(f"[WeNet转录] WeNet 服务创建失败: {e}")
            _wenet_service = None
    return _wenet_service


def _auto_transcribe_reference(audio_id: str, file_path: str, metadata: dict):
    """
    使用 WeNet 自动转录参考音频，并将结果写入 ground_truth_text。
    如果已有 ground_truth_text（预置文本），则跳过。
    转录失败不阻塞上传流程。
    """
    audios = metadata.get("audios", {})
    info = audios.get(audio_id)
    if not info:
        return

    # 如果已有 ground_truth_text（如预置的4个默认参考音频），跳过转录
    if info.get("ground_truth_text"):
        logger.info(f"[WeNet转录] {info.get('original_name')} 已有预置文本，跳过转录")
        return

    try:
        service = _get_wenet_service()
        if service is None:
            logger.warning(f"[WeNet转录] WeNet 服务不可用，跳过 {info.get('original_name')}")
            return

        logger.info(f"[WeNet转录] 开始转录: {info.get('original_name')}")
        text = service.recognize(file_path)

        if text:
            # 清理转录结果（去除多余空格）
            text = text.strip().replace("  ", " ")
            info["ground_truth_text"] = text
            _save_metadata(metadata)
            logger.info(f"[WeNet转录] ✓ {info.get('original_name')} -> \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        else:
            logger.warning(f"[WeNet转录] ✗ {info.get('original_name')} 转录结果为空")

    except Exception as e:
        logger.warning(f"[WeNet转录] 转录异常（非致命）: {e}")


# ============================================================================
# 指纹数据库增量操作辅助函数
# ============================================================================

def _get_matcher_module():
    """延迟导入reference_matcher模块"""
    import sys
    project_root = str(Path(__file__).parent.parent.parent.parent)
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'app', 'core'))
    import reference_matcher as rm
    return rm


def _incremental_add_to_fingerprint(audio_id: str, file_path: str,
                                    description: str = None):
    """增量添加单个参考音频到指纹数据库"""
    try:
        rm = _get_matcher_module()
        result = rm.add_to_matcher_database(
            ref_id=audio_id,
            audio_path=file_path,
            ref_dir=settings.paths.ref_dir,
            description=description
        )
        if result.get("success"):
            logger.info(f"[指纹库] 增量添加成功: {audio_id}, {result.get('hash_count', 0)} Hash, "
                        f"耗时 {result.get('elapsed', 0):.2f}s")
        else:
            logger.warning(f"[指纹库] 增量添加失败: {audio_id}, {result.get('error')}")
    except Exception as e:
        logger.warning(f"[指纹库] 增量添加异常（非致命）: {e}")


def _incremental_batch_add_to_fingerprint(success_list: list, metadata: dict):
    """批量增量添加参考音频到指纹数据库"""
    if not success_list:
        return
    try:
        rm = _get_matcher_module()
        audios = metadata.get("audios", {})
        for item in success_list:
            audio_id = item.get("id")
            if audio_id and audio_id in audios:
                info = audios[audio_id]
                filename = info.get("filename", "")
                file_path = str(Path(settings.paths.ref_dir) / filename)
                if os.path.exists(file_path):
                    rm.add_to_matcher_database(
                        ref_id=audio_id,
                        audio_path=file_path,
                        ref_dir=settings.paths.ref_dir
                    )
        logger.info(f"[指纹库] 批量增量添加: {len(success_list)} 个参考音频")
    except Exception as e:
        logger.warning(f"[指纹库] 批量增量添加异常（非致命）: {e}")


def _incremental_remove_from_fingerprint(audio_id: str):
    """增量移除单个参考音频"""
    try:
        rm = _get_matcher_module()
        result = rm.remove_from_matcher_database(
            ref_id=audio_id,
            ref_dir=settings.paths.ref_dir
        )
        if result.get("success"):
            logger.info(f"[指纹库] 增量移除成功: {audio_id}, 耗时 {result.get('elapsed', 0):.2f}s")
    except Exception as e:
        logger.warning(f"[指纹库] 增量移除异常（非致命）: {e}")


def _rebuild_fingerprint_database():
    """全量重建指纹数据库"""
    try:
        rm = _get_matcher_module()
        rm.rebuild_matcher_database(settings.paths.ref_dir)
        logger.info("[指纹库] 全量重建完成")
    except Exception as e:
        logger.warning(f"[指纹库] 全量重建异常（非致命）: {e}")


# ============================================================================
# 指纹数据库管理端点
# ============================================================================

@router.post("/fingerprint/build")
async def build_fingerprint_database(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """建立/重建参考音频指纹数据库"""
    import sys
    import time

    project_root = str(Path(__file__).parent.parent.parent.parent)
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'app', 'core'))

    try:
        from reference_matcher import ReferenceMatcher, rebuild_matcher_database
    except ImportError as e:
        logger.error(f"导入reference_matcher失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"指纹匹配模块不可用: {str(e)}"
        )

    ref_dir = settings.paths.ref_dir
    logger.info(f"[指纹库] 开始建立指纹数据库: {ref_dir}")

    try:
        start_time = time.time()
        matcher = ReferenceMatcher(ref_dir=ref_dir)
        stats = matcher.build_database(ref_dir)
        elapsed = time.time() - start_time

        rebuild_matcher_database(ref_dir)

        logger.info(f"[指纹库] 指纹数据库建立完成: {stats['total_references']} 个参考音频, "
                     f"{stats['total_hashes']} 个Hash, 耗时 {elapsed:.2f}s")

        return {
            "success": True,
            "message": f"指纹数据库建立完成: {stats['total_references']} 个参考音频, {stats['total_hashes']} 个Hash",
            "statistics": stats,
            "elapsed_seconds": round(elapsed, 2)
        }
    except Exception as e:
        logger.error(f"[指纹库] 建立指纹数据库失败: {e}")
        import traceback
        logger.error(f"[指纹库] 错误详情: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"建立指纹数据库失败: {str(e)}"
        )


@router.get("/fingerprint/status")
async def get_fingerprint_database_status(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """获取指纹数据库状态"""
    import sys

    project_root = str(Path(__file__).parent.parent.parent.parent)
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'app', 'core'))

    try:
        from reference_matcher import get_reference_matcher
        matcher = get_reference_matcher(ref_dir=settings.paths.ref_dir)
        stats = matcher.get_statistics()
        return {
            "has_database": stats["database"]["total_references"] > 0,
            "statistics": stats
        }
    except ImportError as e:
        return {"has_database": False, "error": f"指纹匹配模块不可用: {str(e)}"}
    except Exception as e:
        logger.error(f"获取指纹库状态失败: {e}")
        return {"has_database": False, "error": str(e)}


@router.post("/fingerprint/match-test")
async def test_content_matching(
    current_user: Annotated[User, Depends(get_current_active_user)],
    test_audio_id: str = Form(...)
) -> dict:
    """测试内容匹配功能"""
    import sys
    import time

    project_root = str(Path(__file__).parent.parent.parent.parent)
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'app', 'core'))

    try:
        from reference_pipeline import ReferencePipeline
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"匹配管道不可用: {str(e)}"
        )

    upload_dir = Path(settings.paths.upload_dir) / test_audio_id
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail=f"上传目录不存在: {test_audio_id}")

    test_files = list(upload_dir.glob("*.wav")) + list(upload_dir.glob("*.mp3"))
    if not test_files:
        raise HTTPException(status_code=404, detail="上传目录中没有音频文件")

    ref_dir = settings.paths.ref_dir

    try:
        pipeline = ReferencePipeline(ref_dir=ref_dir)
        pipeline.initialize(ref_dir, force_rebuild=True)

        all_matches = []
        for test_file in test_files:
            start_time = time.time()
            matches = pipeline.match_and_locate(
                str(test_file), min_confidence=0.2, use_dtw=True
            )
            elapsed = time.time() - start_time

            all_matches.append({
                "test_file": test_file.name,
                "match_count": len(matches),
                "elapsed_seconds": round(elapsed, 2),
                "matches": [
                    {
                        "ref_name": m["ref_name"],
                        "ref_id": m["ref_id"],
                        "offset_in_test": round(m["offset_in_test"], 3),
                        "confidence": round(m["confidence"], 3),
                        "hash_matches": m["hash_matches"],
                        "dtw_distance": round(m["dtw_distance"], 1) if m["dtw_distance"] else None,
                        "has_ground_truth": m.get("ground_truth_text") is not None
                    }
                    for m in matches
                ]
            })

        return {
            "success": True,
            "ref_dir": ref_dir,
            "test_files": all_matches
        }
    except Exception as e:
        logger.error(f"内容匹配测试失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"内容匹配失败: {str(e)}"
        )
