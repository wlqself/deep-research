from typing import Literal
from typing_extensions import NotRequired, TypedDict


class PersistentSource(TypedDict):
    source_id: str # 来源的唯一编号（对应之前讲过的 register_source 生成的 S 编号）
    title: str
    url: str
    snippet: str # 来源的摘要/片段文本

    source_type: NotRequired[ # 来源类型
        Literal["web", "rag"]
    ]
    document_id: NotRequired[str] # 当来源来自 RAG 检索时，标记对应的文档 ID
    chunk_id: NotRequired[str] # 文本块 ID
    page_number: NotRequired[int] # 页码
    section_title: NotRequired[str] # 章节标题