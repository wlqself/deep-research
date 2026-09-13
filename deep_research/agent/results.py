from typing import Literal

from pydantic import BaseModel

# 定义researcher的结果组成
class ResearcherResult(BaseModel):
    summary: str
    finding_ids: list[str]
    source_ids: list[str]
    knowledge_gaps: list[str]
    conflicts: list[str]
    recommended_action: Literal[
        "answer",
        "research_more",
    ]