import asyncio

from fastapi import UploadFile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
"""
文档上传到索引的三步流水线编排函数：
它依次调用"接收落盘 → 解析路径 → 切分索引"三个可注入依赖，
把上传的文件最终变成可被 RAG 检索的向量数据，
并返回索引后的文档状态记录。
"""
async def upload_and_index_document(
    file: UploadFile,
    *,
    registry: Any,
    rag_service: Any,
    documents_root: str,
    temp_dir: str,
    collection_id: str,
    max_bytes: int,
    chunk_size: int,
    chunk_overlap: int,
    parent_size: int = 4,
    ingest_upload: Callable[..., Awaitable[Any]],
    resolve_document_path: Callable[..., Path],
    index_document: Callable[..., Awaitable[Any]],
) -> Any:
    # 接收上传并落盘
    record = await ingest_upload(
        file,
        registry=registry,
        documents_root=documents_root,
        temp_dir=temp_dir,
        collection_id=collection_id,
        max_bytes=max_bytes,
    )
    # 解析文档的最终存储路径
    document_path = resolve_document_path(
        documents_root,
        record["document_id"],
        record["safe_filename"],
    )
    # 切分并索引到 RAG 服务
    return await index_document(
        path=document_path,
        record=record,
        registry=registry,
        rag_service=rag_service,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_size=parent_size,
    )
# 负责把一个文档从向量库、文件系统、元数据中完整删除。
async def delete_document_data(
    *,
    document_id: str,
    record: Any, # 文档的元数据记录
    document_path: Path,
    rag_service: Any,
    registry: Any, # 文档注册表（用来归档并删除元数据）
    utc_now: Callable[[], str],
) -> None:
    # 删除向量库中的文档块
    await asyncio.to_thread(
        rag_service.delete_document_chunks,
        document_id,
    )
    # 安全校验：禁止符号链接
    if document_path.is_symlink():
        raise RuntimeError(
            "document path must not be a symlink"
        )
    # 删除磁盘上的文件
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
        created_at=utc_now(),
    )
"""
reindex_document_data 是一个重新索引的薄包装函数：
它把文档路径、元数据和切分参数透传给注入的 reindex_document 依赖，
由后者完成"清理旧向量 → 重切分 → 重索引 → 更新状态"的完整流程，本身只负责统一入口和参数转发。
"""
async def reindex_document_data(
    *,
    path: Path,
    record: Any,
    registry: Any,
    rag_service: Any,
    chunk_size: int,
    chunk_overlap: int,
    parent_size: int = 4,
    reindex_document: Callable[..., Awaitable[Any]],
) -> Any:
    return await reindex_document(
        path=path,
        record=record,
        registry=registry,
        rag_service=rag_service,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_size=parent_size,
    )
