from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)


MemoryEntryKind = Literal[
    "user",
    "reference",
    "project",
    "feedback",
]

MemoryCandidateKind = Literal[
    "user",
    "reference",
    "project",
    "feedback",
    "ignore",
]

MemoryStatus = Literal[
    "active",
    "superseded",
]

MemoryScope = Literal[
    "global",
    "project",
]

MemoryWriteIntent = Literal[
    "automatic",
    "explicit_remember",
    "explicit_update",
]

MemoryDecisionAction = Literal[
    "create",
    "update",
    "no-op",
    "ignore",
]

class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1) # 某条具体记录的唯一 ID
    memory_key: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_.-]*$",
    ) # 判断是否属于同一个事实槽位
    scope: MemoryScope = "global" # 说明是全局偏好还是项目范围记忆
    kind: MemoryEntryKind
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    content: str = Field(min_length=1)
    keywords: list[str]

    source_type: str = Field(min_length=1)
    source_thread_id: str | None = None
    url: HttpUrl | None = None
    verified_at: datetime | None = None

    incorrect: str | None = None
    correct: str | None = None
    applies_when: str | None = None

    created_at: datetime
    updated_at: datetime
    status: MemoryStatus

    @model_validator(mode="after")
    def validate_kind_specific_fields(self):
        # 判断类型为reference
        if self.kind == "reference":
            if self.url is None or self.verified_at is None:
                raise ValueError(
                    "reference memory requires url and verified_at"
                )

        if self.kind == "feedback":
            feedback_fields = (
                self.incorrect,
                self.correct,
                self.applies_when,
            )

            if any(
                value is None or not value.strip()
                for value in feedback_fields
            ):
                raise ValueError(
                    "feedback memory requires incorrect, correct, "
                    "and applies_when"
                )

        return self
class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MemoryCandidateKind
    memory_key: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_.-]*$",
    )
    scope: MemoryScope = "global"

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    content: str = Field(min_length=1)
    keywords: list[str]

    source_type: str = Field(min_length=1)
    source_thread_id: str | None = None
    url: HttpUrl | None = None
    verified_at: datetime | None = None

    incorrect: str | None = None
    correct: str | None = None
    applies_when: str | None = None

    @model_validator(mode="after")
    def validate_kind_specific_fields(self):
        if self.kind == "reference":
            if self.url is None:
                raise ValueError("reference memory requires url")
            if not str(self.url).startswith(
                ("http://", "https://")
            ):
                raise ValueError("reference url must be http or https")
            if self.verified_at is None:
                raise ValueError("reference memory requires verified_at")

        elif self.kind == "feedback":
            feedback_fields = (
                self.incorrect,
                self.correct,
                self.applies_when,
            )
            if any(
                value is None or not value.strip()
                for value in feedback_fields
            ):
                raise ValueError(
                    "feedback memory requires incorrect, correct, "
                    "and applies_when"
                )

        elif self.kind == "ignore":
            # ignore 不要求任何字段，直接通过
            # 它只是标记"这条候选不写入 Store"
            pass
        elif self.kind in ("user", "project"):
            pass

        elif self.kind == "ignore":
            pass
        else:
            raise ValueError(f"unsupported memory kind: {self.kind}")

        return self
class SummaryArchive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    summary_hash: str = Field(min_length=1)

    original_summary: str = Field(min_length=1)
    retrieval_summary: str = Field(min_length=1)
    topics: list[str]

    memory_keys: list[str] = Field(default_factory=list) # 这个归档涉及哪些记忆 identity
    recallable: bool = False # 是否允许进入自动召回
    excluded_reason: str | None = None # 不允许召回的内部原因

    version: int = Field(ge=1)
    archived_at: datetime

class MemoryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: MemoryDecisionAction
    existing_id: str | None = None
    reason: str = Field(min_length=1)


class MemorySuppression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suppression_id: str = Field(min_length=1)
    kind: MemoryEntryKind
    memory_key: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_.-]*$",
    )
    scope: MemoryScope
    forgotten_memory_id: str = Field(min_length=1)
    source_thread_id: str | None = None
    created_at: datetime
    reason: str = Field(min_length=1)