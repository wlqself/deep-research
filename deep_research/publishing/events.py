"""Stable, user-safe events emitted by the publishing workflow.

These events are intentionally smaller than the application's structured logs.
They describe progress that may be shown in Todo/Activity UI and do not carry
raw exceptions, file paths, database paths, credentials, or article content.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from .models import PublicationChannel

# 枚举——业务事件类型
class PublishingEventType(StrEnum):
    """Business progress events exposed to application-facing consumers."""

    ARTICLE_CREATED = "article_created" # 文章从制品导入创建完成
    ARTICLE_UPDATED = "article_updated" # 文章被编辑更新
    APPROVAL_REQUESTED = "approval_requested" # 文章提交审核（对应 approve 操作）
    ARTICLE_APPROVED = "article_approved" # 用户批准发布
    PUBLISHING_STARTED = "publishing_started" # 发布流程启动（对应 start_publishing）
    PUBLISHED = "published" # 发布成功
    PUBLISHING_FAILED = "publishing_failed" # 发布失败

#  枚举——事件状态词汇
class PublishingEventStatus(StrEnum):
    """Small, stable status vocabulary for progress displays."""

    IN_PROGRESS = "in_progress"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

# 数据类——不可变事件对象
@dataclass(frozen=True, slots=True)
class PublishingEvent:
    """A safe progress event for a single publishing workflow transition.

    ``summary`` is a controlled display label supplied by the application
    layer. It must not contain raw exception text or sensitive implementation
    details. Consumers can also ignore it and map ``event_type`` to localized
    UI text.
    """

    event_id: str # 事件唯一标识（UUID hex)
    event_type: PublishingEventType # 事件类型（上面的枚举）
    status: PublishingEventStatus # 事件状态（上面的枚举）
    occurred_at: str # 事件发生时间（ISO 格式字符串）
    article_id: str | None = None # 关联的文章标识
    article_version: int | None = None 
    publication_id: str | None = None  # 关联的发布记录标识
    channel: PublicationChannel | None = None  # 发布渠道
    error_code: str | None = None  # 失败时的错误码
    summary: str = "" # 受控的展示标签（不含原始异常或敏感信息，消费者也可以忽略它自行做国际化映射）
    # 对事件字段做完整性校验
    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if not self.occurred_at.strip():
            raise ValueError("occurred_at must not be empty")
        if self.article_version is not None and self.article_version < 1:
            raise ValueError("article_version must be positive")
        if self.status is PublishingEventStatus.FAILED and not self.error_code:
            raise ValueError("failed events require an error_code")
        if self.status is not PublishingEventStatus.FAILED and self.error_code:
            raise ValueError("error_code is only valid for failed events")

    @classmethod
    # 工厂方法
    def create(
        cls,
        *,
        event_type: PublishingEventType,
        status: PublishingEventStatus,
        article_id: str | None = None,
        article_version: int | None = None,
        publication_id: str | None = None,
        channel: PublicationChannel | None = None,
        error_code: str | None = None,
        summary: str = "",
    ) -> "PublishingEvent":
        """Create an event with a fresh identifier and UTC timestamp."""

        return cls(
            event_id=uuid4().hex,
            event_type=event_type,
            status=status,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            article_id=article_id,
            article_version=article_version,
            publication_id=publication_id,
            channel=channel,
            error_code=error_code,
            summary=summary,
        )
    # 将事件转为 JSON 兼容的字典
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for stream consumers."""

        values = asdict(self)
        values["event_type"] = self.event_type.value
        values["status"] = self.status.value
        if self.channel is not None:
            values["channel"] = self.channel.value
        return values
