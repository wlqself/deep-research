import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from ..config import settings
from ..log.log_audit import audit_event
from ..log.logging_utils import log_event
from ..rag.ingestion import (
    DuplicateDocumentError,
    index_document,
    ingest_upload,
    reindex_document,
)
from ..rag.registry import DocumentStatus
from ..rag.storage import resolve_document_path
from .knowledge_commands import (
    delete_document_data,
    reindex_document_data,
    upload_and_index_document,
)
from .knowledge_models import (
    DocumentResponse,
    KnowledgeStatsResponse,
)
from .knowledge_queries import (
    get_document as get_document_record,
    resolve_existing_document_path,
)
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/knowledge/documents",
    response_model=DocumentResponse,
    status_code=201,
)
# 上传文件
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> DocumentResponse:
    
    registry = request.app.state.document_registry
    rag_service = request.app.state.rag_service

    documents_root = settings.rag_documents_path
    temp_dir = str(
        Path(documents_root).parent
        / ".rag-upload-tmp"
    )

    started_at = time.perf_counter()

    log_event(
        logger,
        logging.INFO,
        "rag.document.upload.started",
        status="started",
    )

    try:
        record = await upload_and_index_document(
            file,
            registry=registry,
            rag_service=rag_service,
            documents_root=documents_root,
            temp_dir=temp_dir,
            collection_id=settings.rag_collection_name,
            max_bytes=settings.rag_max_file_bytes,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            parent_size=settings.rag_parent_size,
            ingest_upload=ingest_upload,
            resolve_document_path=resolve_document_path,
            index_document=index_document,
        )

        log_event(
            logger,
            logging.INFO,
            "rag.document.upload.completed",
            document_id=record["document_id"],
            status="completed",
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )

        audit_event(
            "document.upload",
            "completed",
            document_id=record["document_id"],
        )
        return DocumentResponse(**record)

    except DuplicateDocumentError as error:
        log_event(
            logger,
            logging.WARNING,
            "rag.document.upload.rejected",
            document_id=error.existing_document_id,
            status="rejected",
            error_code="duplicate_document",
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "duplicate_document",
                "document_id": (
                    error.existing_document_id
                ),
            },
        ) from error

    except ValueError as error:
        message = str(error)
        error_code = (
            "upload_size_limit"
            if "size limit" in message
            else "invalid_document_upload"
        )

        log_event(
            logger,
            logging.WARNING,
            "rag.document.upload.failed",
            status="failed",
            error_code=error_code,
            exception_type=type(error).__name__,
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
            exc_info=True,
        )
        if "size limit" in message:
            raise HTTPException(
                status_code=413,
                detail="uploaded file exceeds size limit",
            ) from error

        raise HTTPException(
            status_code=400,
            detail="invalid document upload",
        ) from error

    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "rag.document.upload.failed",
            status="failed",
            error_code="document_upload_failed",
            exception_type=type(error).__name__,
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="document indexing failed",
        ) from error

    finally:
        await file.close()

@router.get(
    "/knowledge/documents",
    response_model=list[DocumentResponse],
)
async def list_documents(
    request: Request,
    status: DocumentStatus | None = None,
) -> list[DocumentResponse]:
    registry = request.app.state.document_registry
    # 异步获取记录
    records = await asyncio.to_thread(
        registry.list_documents,
        status=status,
    )

    return [
        DocumentResponse(**record)
        for record in records
    ]

@router.get(
    "/knowledge/documents/{document_id}",
    response_model=DocumentResponse,
)
async def get_document( # 查找并返回一个文件
    document_id: str,
    request: Request,
) -> DocumentResponse:
    registry = request.app.state.document_registry
    # 异步获取记录
    record = await get_document_record(
        registry,
        document_id,
    )

    if (
        record is None
        or record["status"] == "deleted"
    ):
        raise HTTPException(
            status_code=404,
            detail="document not found",
        )

    return DocumentResponse(**record)

@router.get(
    "/knowledge/documents/{document_id}/download",
)
async def download_document(
    document_id: str, # 下载一个文件
    request: Request,
) -> FileResponse:
    registry = request.app.state.document_registry

    record = await get_document_record(
        registry,
        document_id,
    )

    if (
        record is None
        or record["status"] == "deleted"
    ):
        raise HTTPException(
            status_code=404,
            detail="document not found",
        )
    # 用户看到的文件名 -> record["filename"]
    # 服务器实际路径 -> record["safe_filename"] + document_id
    try:
        document_path = resolve_existing_document_path(
            documents_root=settings.rag_documents_path,
            record=record,
            resolve_document_path=resolve_document_path,
        )
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=404,
            detail="document file not found",
        ) from error

    return FileResponse(
        path=document_path,
        media_type=record["mime_type"],
        filename=record["filename"],
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )

def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

@router.delete(
    "/knowledge/documents/{document_id}",
)
async def delete_document(
    document_id: str,
    request: Request,
) -> dict[str, str]:
    # 从state中获取信息
    registry = request.app.state.document_registry
    rag_service = request.app.state.rag_service

    # 获取要删除的文档
    record = await get_document_record(
        registry,
        document_id,
    )
    # 若已经删除，没法找到，返回404
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="document not found",
        )
    # 若状态已经改为删除，则默认删除
    if record["status"] == "deleted":
        log_event(
            logger,
            logging.INFO,
            "rag.document.delete.noop",
            document_id=document_id,
            status="no_op",
        )
        return {
            "document_id": document_id,
            "status": "deleted",
        }

    try:
        document_path = resolve_document_path( # 找出磁盘上的绝对路径
            settings.rag_documents_path,
            record["document_id"],
            record["safe_filename"],
        )
        # 出现错误报错
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail="document path is invalid",
        ) from error
    # 异步更新状态，并标明时间
    await asyncio.to_thread(
        registry.update_document_status,
        document_id,
        status="deleting",
        updated_at=_utc_now(),
    )

    started_at = time.perf_counter()

    log_event(
        logger,
        logging.INFO,
        "rag.document.delete.started",
        document_id=document_id,
        status="started",
    )

    try:
        # 通过id删除文件chunk
        await delete_document_data(
            document_id=document_id,
            record=record,
            document_path=document_path,
            rag_service=rag_service,
            registry=registry,
            utc_now=_utc_now,
        )

        audit_event(
            "document.delete",
            "completed",
            document_id=document_id,
        )
        log_event(
            logger,
            logging.INFO,
            "rag.document.deleted",
            document_id=document_id,
            status="completed",
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )

    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "rag.document.delete.failed",
            document_id=document_id,
            status="failed",
            error_code="document_delete_failed",
            exception_type=type(error).__name__,
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
            exc_info=True,
        )

        audit_event(
            "document.delete",
            "failed",
            document_id=document_id,
        )
        raise HTTPException(
            status_code=500,
            detail="document deletion failed",
        ) from error

    return {
        "document_id": document_id,
        "status": "deleted",
    }
"""
POST /knowledge/documents/{id}/reindex
  -> 查注册表
  -> 检查不是 deleted
  -> 安全解析文件路径
  -> 检查原始文件存在
  -> 旧 Chunk pending
  -> 删除旧 Chunk
  -> 重新解析和切分
  -> 重新 Embedding
  -> 写入 Qdrant
  -> indexed
"""
@router.post(
    "/knowledge/documents/{document_id}/reindex",
    response_model=DocumentResponse,
)
async def reindex_document_endpoint(
    document_id: str,
    request: Request,
) -> DocumentResponse:
    
    registry = request.app.state.document_registry
    rag_service = request.app.state.rag_service

    record = await get_document_record(
        registry,
        document_id,
    )

    if (
        record is None
        or record["status"] == "deleted"
    ):
        raise HTTPException(
            status_code=404,
            detail="document not found",
        )

    try:
        document_path = resolve_existing_document_path(
            documents_root=settings.rag_documents_path,
            record=record,
            resolve_document_path=resolve_document_path,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail="document path is invalid",
        ) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail="document file not found",
        ) from error

    started_at = time.perf_counter()

    log_event(
        logger,
        logging.INFO,
        "rag.document.reindex.requested",
        document_id=document_id,
        status="started",
    )

    try:
        # 重新索引文件
        updated_record = await reindex_document_data(
            path=document_path,
            record=record,
            registry=registry,
            rag_service=rag_service,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            parent_size=settings.rag_parent_size,
            reindex_document=reindex_document,
        )
        audit_event(
            "document.reindex",
            "completed",
            document_id=document_id,
        )
        log_event(
            logger,
            logging.INFO,
            "rag.document.reindex.completed",
            document_id=document_id,
            status="completed",
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )

    except Exception as error:
        audit_event(
            "document.reindex",
            "failed",
            document_id=document_id,
        )
        log_event(
            logger,
            logging.ERROR,
            "rag.document.reindex.endpoint_failed",
            document_id=document_id,
            status="failed",
            error_code="reindex_endpoint_failed",
            exception_type=type(error).__name__,
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="document reindex failed",
        ) from error

    return DocumentResponse(**updated_record)

@router.get(
    "/knowledge/stats",
    response_model=KnowledgeStatsResponse,
)
async def knowledge_stats(
    request: Request,
) -> KnowledgeStatsResponse:
    registry = request.app.state.document_registry

    statistics = await asyncio.to_thread(
        registry.get_statistics,
    )

    return KnowledgeStatsResponse(
        **statistics
    )
