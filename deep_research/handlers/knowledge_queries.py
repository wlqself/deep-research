import asyncio
from pathlib import Path
from typing import Any, Callable

# 获取文档
async def get_document(
    registry: Any,
    document_id: str,
) -> Any:
    return await asyncio.to_thread(
        registry.get_document,
        document_id,
    )

# 处理文档链接
def resolve_existing_document_path(
    *,
    documents_root: str,
    record: Any,
    resolve_document_path: Callable[..., Path],
) -> Path:
    document_path = resolve_document_path(
        documents_root,
        record["document_id"],
        record["safe_filename"],
    )

    if document_path.is_symlink():
        raise FileNotFoundError("document file not found")

    if not document_path.is_file():
        raise FileNotFoundError("document file not found")

    return document_path