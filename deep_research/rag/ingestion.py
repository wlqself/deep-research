"""
validate_upload_type
  ↓
生成 document_id
  ↓
write_upload_to_temp
  ↓
get_document_by_sha256
  ↓
创建 pending 注册记录
  ↓
build_document_path
  ↓
finalize_document_file
  ↓
返回 DocumentRecord

"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from langchain_core.documents import Document
from fastapi import UploadFile

from .service import RagService
from .registry import (
    DocumentRecord,
    DocumentRegistry,
)
from .storage import (
    build_document_path,
    finalize_document_file,
    validate_upload_type,
    write_upload_to_temp,
)
from .chunking import chunk_documents
from .parsers import parse_document
from ..log.logging_utils import log_event


logger = logging.getLogger(__name__)

def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

class DuplicateDocumentError(Exception):
    def __init__(
        self,
        existing_document_id: str,
    ) -> None:
        self.existing_document_id = (
            existing_document_id
        )
        super().__init__(
            "document already exists"
        )

async def ingest_upload(
    upload: UploadFile,
    *,
    registry: DocumentRegistry,
    documents_root: str,
    temp_dir: str,
    collection_id: str,
    max_bytes: int,
) -> DocumentRecord:
    # 函数开头先处理文件名和 MIME：
    filename = upload.filename or ""
    content_type = upload.content_type or ""

    extension = validate_upload_type(
        filename,
        content_type,
    )

    normalized_mime_type = (
        content_type
        .split(";", 1)[0]
        .strip()
        .lower()
    )

    # 然后生成服务端文档 ID 和最终安全路径：
    document_id = str(uuid4())

    safe_filename, document_path = (
        build_document_path(
            documents_root,
            document_id,
            extension,
        )
    )

    temporary_path: Path | None = None
    record_created = False
    # 流式写入临时文件
    # 计算大小
    # 计算 SHA-256
    try:
        temporary_path, size_bytes, sha256 = (
            await write_upload_to_temp(
                upload,
                temp_dir,
                max_bytes,
            )
        )
        existing = await asyncio.to_thread(
            registry.get_document_by_sha256,
            sha256,
        )

        if existing is not None:
            raise DuplicateDocumentError(
                existing["document_id"]
            )
        
        # 创建 pending 注册记录
        now = _utc_now()

        record: DocumentRecord = {
            "document_id": document_id,
            "collection_id": collection_id,
            "filename": filename,
            "safe_filename": safe_filename,
            "mime_type": normalized_mime_type,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "status": "pending",
            "page_count": 0,
            "chunk_count": 0,
            "error": "",
            "created_at": now,
            "updated_at": now,
        }           
        # 写入 SQLite
        await asyncio.to_thread(
            registry.create_document,
            record,
        )
        record_created = True

        # 最终放入安全路径
        await asyncio.to_thread(
            finalize_document_file,
            temporary_path,
            document_path,
        )

        return record
    # 失败处理
    except DuplicateDocumentError:
        raise

    except Exception:
        if record_created:
            try:
                await asyncio.to_thread(
                    registry.update_document_status,
                    document_id,
                    status="failed",
                    error="storage_failed",
                    updated_at=_utc_now(),
                )
            except Exception:
                pass

        raise

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink(
                missing_ok=True
            )


def prepare_document_chunks(
    path: Path,
    *,
    record: DocumentRecord,
    chunk_size: int,
    chunk_overlap: int,
    parent_size: int = 4,
) -> tuple[list[Document], int]:
    documents, page_count = parse_document(
        path,
        document_id=record["document_id"],
        collection_id=record["collection_id"],
        filename=record["filename"],
        mime_type=record["mime_type"],
    )

    chunks = chunk_documents(
        documents,
        document_hash=record["sha256"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_size=parent_size,
    )

    return chunks, page_count

async def index_document(
    *,
    path: Path,
    record: DocumentRecord,
    registry: DocumentRegistry,
    rag_service: RagService,
    chunk_size: int,
    chunk_overlap: int,
    parent_size: int = 4,
) -> DocumentRecord:
    started_at = time.perf_counter()

    # 1. 先更新为 processing
    await asyncio.to_thread(
        registry.update_document_status,
        record["document_id"],
        status="processing",
        updated_at=_utc_now(),
    )

    log_event(
        logger,
        logging.INFO,
        "rag.document.index.started",
        document_id=record["document_id"],
        status="started",
    )
    """
    成功顺序是：
    processing
    -> parse
    -> chunk
    -> Qdrant 写入
    -> 数量校验
    -> Qdrant status=indexed
    -> SQLite status=indexed
    """
    # 2. 解析并切 Chunk
    try:
        chunks, page_count = await asyncio.to_thread(
            prepare_document_chunks,
            path,
            record=record,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_size=parent_size,
        )
        # 3. 写入并验证数量
        written_count = await asyncio.to_thread(
            rag_service.index_chunks,
            chunks,
        )

        if written_count != len(chunks):
            raise RuntimeError(
                "indexed chunk count mismatch"
            )
        # 4. 更新 Qdrant 状态
        if chunks:
            await asyncio.to_thread(
                rag_service.mark_document_chunks_indexed,
                record["document_id"],
            )
        # 5. 更新 SQLite 状态
        updated_at = _utc_now()

        await asyncio.to_thread(
            registry.update_document_status,
            record["document_id"],
            status="indexed",
            page_count=page_count,
            chunk_count=len(chunks),
            updated_at=updated_at,
        )

        log_event(
            logger,
            logging.INFO,
            "rag.document.indexed",
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

        return {
            **record,
            "status": "indexed",
            "page_count": page_count,
            "chunk_count": len(chunks),
            "updated_at": updated_at,
        }

    # 6. 失败清理
    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "rag.document.index.failed",
            document_id=record["document_id"],
            status="failed",
            error_code="indexing_failed",
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
        try:
            # 删除向量
            await asyncio.to_thread(
                rag_service.delete_document_chunks,
                record["document_id"],
            )
            # 若失败更新状态
        finally:
            await asyncio.to_thread(
                registry.update_document_status,
                record["document_id"],
                status="failed",
                error="indexing_failed",
                updated_at=_utc_now(),
            )

        raise
"""
重新索引流程
SQLite processing
  ↓
metadata.status = pending
  ↓
删除旧 Qdrant Points
  ↓
重新解析
  ↓
重新切 Chunk
  ↓
重新 Embedding
  ↓
写入新 Points
  ↓
metadata.status = indexed
  ↓
SQLite indexed
"""

async def reindex_document(
    *,
    path: Path,
    record: DocumentRecord,
    registry: DocumentRegistry,
    rag_service: RagService,
    chunk_size: int,
    chunk_overlap: int,
    parent_size: int = 4,
) -> DocumentRecord:
    document_id = record["document_id"]
    started_at = time.perf_counter()

    # 更新状态并加上时间
    await asyncio.to_thread(
        registry.update_document_status,
        document_id,
        status="processing",
        updated_at=_utc_now(),
    )

    log_event(
        logger,
        logging.INFO,
        "rag.document.reindex.started",
        document_id=document_id,
        status="started",
    )
    try:
        # 旧 Chunk 先变成不可检索状态
        await asyncio.to_thread(
            rag_service.mark_document_chunks_pending,
            document_id,
        )

        # 删除旧向量
        await asyncio.to_thread(
            rag_service.delete_document_chunks,
            document_id,
        )

        # 重新解析、切 Chunk、Embedding、写入
        result = await index_document(
            path=path,
            record=record,
            registry=registry,
            rag_service=rag_service,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_size=parent_size,
        )

        log_event(
            logger,
            logging.INFO,
            "rag.document.reindexed",
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

        return result

    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "rag.document.reindex.failed",
            document_id=document_id,
            status="failed",
            error_code="reindex_failed",
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
        # 失败的时候更新状态
        await asyncio.to_thread(
            registry.update_document_status,
            document_id,
            status="failed",
            error="reindex_failed",
            updated_at=_utc_now(),
        )
        raise
        
