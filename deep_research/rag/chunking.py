import hashlib
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

def chunk_documents(
    documents: list[Document],
    *,
    document_hash: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            " ",
            "。",
            "！",
            "？",
            "；",
            "，",
            "、",
            ".",
            ",",
            "",
        ],
    )

    chunks: list[Document] = []

    for document in documents:
        pieces = splitter.split_text(
            document.page_content
        )

        for piece in pieces:
            content = piece.strip()

            if not content:
                continue

            metadata = document.metadata
            document_id = str(
                metadata["document_id"]
            )

            chunk_index = len(chunks)
            
            # 重新索引时可以覆盖相同 Point，而不是随机产生新 ID
            chunk_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{document_id}:{chunk_index}",
                )
            )

            content_hash = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()

            chunk_metadata = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "collection_id": str(
                    metadata["collection_id"]
                ),
                "filename": str(
                    metadata["filename"]
                ),
                "title": str(
                    metadata.get(
                        "title",
                        metadata["filename"],
                    )
                ),
                "mime_type": str(
                    metadata["mime_type"]
                ),
                "page_number": metadata.get(
                    "page_number",
                    1,
                ),
                "section_title": str(
                    metadata.get(
                        "section_title",
                        "",
                    )
                ),
                "chunk_index": chunk_index,
                "content_hash": content_hash,
                "document_hash": document_hash,
                "created_at": _utc_now(),
            }

            chunks.append(
                Document(
                    page_content=content,
                    metadata=chunk_metadata,
                )
            )

    return chunks