import asyncio
import logging
from fastapi.responses import FileResponse
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel

from ..config import settings
from ..rag.ingestion import (
    DuplicateDocumentError,
    index_document,
    ingest_upload,
)
from ..rag.storage import resolve_document_path
from ..rag.registry import DocumentStatus
from datetime import datetime, timezone
from ..rag.ingestion import (
    DuplicateDocumentError,
    index_document,
    ingest_upload,
    reindex_document,
)

router = APIRouter()
logger = logging.getLogger(__name__)

class KnowledgeStatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    indexed_count: int
    processing_count: int
    failed_count: int
    pending_count: int

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    mime_type: str
    size_bytes: int
    status: str
    page_count: int
    chunk_count: int
    error: str
    created_at: str
    updated_at: str

@router.post(
    "/knowledge/documents",
    response_model=DocumentResponse,
    status_code=201,
)

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

    try:
        record = await ingest_upload( # ← 存临时文件 → 移到正式目录 → SQLite 写 pending 记录
            file,
            registry=registry,
            documents_root=documents_root,
            temp_dir=temp_dir,
            collection_id=settings.rag_collection_name,
            max_bytes=settings.rag_max_file_bytes,
        )

        document_path = resolve_document_path( # ← 算出文件在磁盘上的绝对路径
            documents_root,
            record["document_id"],
            record["safe_filename"],
        )

        record = await index_document( # ← SQLite processing → 解析 → chunk → Qdrant 写入 → SQLite indexed
            path=document_path, 
            record=record,
            registry=registry,
            rag_service=rag_service,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

        return DocumentResponse(**record)

    except DuplicateDocumentError as error:
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
        logger.exception(
            "document indexing failed: %s",
            error,
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
        logger.exception(
            "document upload failed: filename=%s",
            file.filename or "",
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
    record = await asyncio.to_thread(
        registry.get_document,
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

    record = await asyncio.to_thread(
        registry.get_document,
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
        document_path = resolve_document_path( # 找出磁盘上的绝对路径
            settings.rag_documents_path,
            record["document_id"],
            record["safe_filename"],
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail="document not found",
        ) from error

    if (
        document_path.is_symlink()
        or not document_path.is_file()
    ):
        raise HTTPException(
            status_code=404,
            detail="document file not found",
        )

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
    record = await asyncio.to_thread(
        registry.get_document,
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

    try:
        # 通过id删除文件chunk
        await asyncio.to_thread(
            rag_service.delete_document_chunks,
            document_id,
        )
        # 不能删除符号链接
        if document_path.is_symlink():
            raise RuntimeError(
                "document path must not be a symlink"
            )
        # 必须是文件
        if document_path.exists():
            if not document_path.is_file():
                raise RuntimeError(
                    "document path is not a file"
                )

            await asyncio.to_thread(
                document_path.unlink,
            )

        await asyncio.to_thread(
            registry.archive_and_delete_document,
            record,
            event_type="deleted",
            created_at=_utc_now(),
        )

    except Exception as error:
        logger.exception(
            "document deletion failed: document_id=%s",
            document_id,
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

    record = await asyncio.to_thread(
        registry.get_document,
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
        document_path = resolve_document_path(
            settings.rag_documents_path,
            record["document_id"],
            record["safe_filename"],
        )
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail="document path is invalid",
        ) from error

    if (
        document_path.is_symlink()
        or not document_path.is_file()
    ):
        raise HTTPException(
            status_code=404,
            detail="document file not found",
        )

    try:
        # 重新索引文件
        updated_record = await reindex_document(
            path=document_path,
            record=record,
            registry=registry,
            rag_service=rag_service,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    except Exception as error:
        logger.exception(
            "document reindex endpoint failed: document_id=%s",
            document_id,
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