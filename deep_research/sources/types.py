from typing import Literal
from typing_extensions import NotRequired, TypedDict


class PersistentSource(TypedDict):
    source_id: str
    title: str
    url: str
    snippet: str

    source_type: NotRequired[
        Literal["web", "rag"]
    ]
    document_id: NotRequired[str]
    chunk_id: NotRequired[str]
    page_number: NotRequired[int]
    section_title: NotRequired[str]