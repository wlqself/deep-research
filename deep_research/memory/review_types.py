from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .type import MemoryCandidate, MemoryEntry


# 一条待审查的对话消息，把「上次审查之后新产生的对话」打包传给 LLM 去提取记忆。
class ReviewMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal[
        "user",
        "assistant",
    ]
    content: str = Field(min_length=1)

class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    included_messages: list[ReviewMessage] # 本批真正交给 extractor 的消息
    last_included_message_id: str | None = None # 本批最后一条真正纳入的原始消息 ID
    has_more_unreviewed: bool = False # 游标之后是否还有未纳入本批的内容
    total_chars: int = Field(ge=0) # 本批实际纳入消息的字符总数
    truncated_fields: list[str] = Field(
        default_factory=list # 哪些字段因为限制被截断或排除
    )

# 告诉 LLM「你之前已经生成过这些文件了」，避免重复创建，也能让记忆里正确引用文件路径。
class ReviewArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    workspace_path: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=1)

# 喂给 LLM 做记忆提取的完整上下文
class MemoryReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_summary: str | None = None
    unreviewed_messages: list[ReviewMessage]
    review_batch: ReviewBatch
    latest_user_message: str | None = None
    latest_main_message: str | None = None
    artifact_metadata: list[ReviewArtifactMetadata]
    similar_old_memories: list[MemoryEntry]


# LLM 的输出结果
class MemoryExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[MemoryCandidate] = Field(
        default_factory=list,
        max_length=5,
    )


