import re
from pathlib import Path
from langchain_core.documents import Document
from pypdf import PdfReader

# 读取 UTF-8 文件
#   ↓
# 转换成 LangChain Document
#   ↓
# 保留基础 metadata

def parse_txt_document(
        path: Path,
        *,
        document_id: str,
        collection_id: str,
        filename: str,
        mime_type: str,
) -> list[Document]:
    text = path.read_text(
        encoding= "utf-8",
        errors="strict",
    )

    if not text.strip():
        return []

    return[
        Document(
            page_content=text,
            metadata={
                "document_id": document_id,
                "collection_id": collection_id,
                "filename": filename,
                "mime_type": mime_type,
                "page_number": 1,
                "section_title": "",      
            },
        )
    ]

def parse_pdf_document(
    path: Path,
    *,
    document_id: str,
    collection_id: str,
    filename: str,
    mime_type: str,
) -> tuple[list[Document], int]:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    documents: list[Document] = []

    # 逐页提取
    for page_index, page in enumerate(
        reader.pages,
        start=1,
    ):  
        text = page.extract_text() or ""    # extract_text() 可能返回 None（某些加密/扫描页），or "" 兜底。
        text = text.strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "collection_id": collection_id,
                    "filename": filename,
                    "mime_type": mime_type,
                    "page_number": page_index,
                    "section_title": "",
                },
            )
        )

    return documents, page_count

def parse_markdown_document(
    path: Path,
    *,
    document_id: str,
    collection_id: str,
    filename: str,
    mime_type: str,
) -> list[Document]:
    text = path.read_text(
        encoding="utf-8",
        errors="strict",
    )

    if not text.strip():
        return []

    documents: list[Document] = []
    current_title = ""
    current_lines: list[str] = []

    def flush_section() -> None:
        content = "\n".join(
            current_lines
        ).strip()

        if not content:
            return

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "document_id": document_id,
                    "collection_id": collection_id,
                    "filename": filename,
                    "mime_type": mime_type,
                    "page_number": 1,
                    "section_title": current_title,
                },
            )
        )

    for line in text.splitlines():
        heading = re.match(
            r"^#{1,6}\s+(.+?)\s*$",
            line,
        )

        if heading:
            flush_section()
            current_lines.clear()
            current_title = heading.group(1).strip()
            continue

        current_lines.append(line)

    flush_section()

    return documents

def parse_document(
    path: Path,
    *,
    document_id: str,
    collection_id: str,
    filename: str,
    mime_type: str,
) -> tuple[list[Document], int]:
    common = {
        "document_id": document_id,
        "collection_id": collection_id,
        "filename": filename,
        "mime_type": mime_type,
    }

    if mime_type == "application/pdf":
        return parse_pdf_document(
            path,
            **common,
        )

    if mime_type == "text/markdown":
        documents = parse_markdown_document(
            path,
            **common,
        )
        return documents, 1 if documents else 0

    if mime_type == "text/plain":
        documents = parse_txt_document(
            path,
            **common,
        )
        return documents, 1 if documents else 0

    raise ValueError(
        "unsupported document MIME type"
    )