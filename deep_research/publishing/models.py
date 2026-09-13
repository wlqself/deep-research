"""Domain models for the publishing workflow.

This module deliberately contains no persistence, HTTP, or filesystem code.
The publishing service will be responsible for coordinating these models with
the existing research Artifact store and the publishing repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

# 发布领域（Publishing Domain）中所有确定性业务异常的父类
class PublishingDomainError(Exception):
    """Base exception for deterministic publishing-domain failures."""

# 状态转换异常
class InvalidStateTransitionError(PublishingDomainError):
    """Raised when an entity is asked to make an illegal state transition."""

# 非法领域值异常
class InvalidDomainValueError(PublishingDomainError):
    """Raised when a domain object is created with an invalid value."""

# 文章状态枚举
class ArticleStatus(StrEnum):
    DRAFT = "draft" # 草稿状态
    APPROVED = "approved" # 已审核通过，等待发布
    PUBLISHING = "publishing" # 正在发布中
    PUBLISHED = "published" # 已发布成功
    FAILED = "failed" # 发布失败

# 发布渠道枚举
class PublicationChannel(StrEnum):
    LOCAL_STATIC_SITE = "local_static_site"
    WECHAT_OFFICIAL_ACCOUNT = "wechat_official_account"
    XIAOHONGSHU = "xiaohongshu"
    DOUYIN = "douyin"

# 发布状态枚举
class PublicationStatus(StrEnum):
    PUBLISHING = "publishing"
    DRAFTED = "drafted"
    PUBLISHED = "published"
    FAILED = "failed"
    DELIVERY_UNKNOWN = "delivery_unknown"


# 时间返回函数
def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for domain records."""

    return datetime.now(timezone.utc)


@dataclass
class WeChatCoverAsset:
    """A versioned, reusable WeChat permanent cover material."""

    asset_id: str
    content_sha256: str
    remote_media_id: str
    is_active: bool = False
    created_at: datetime = field(default_factory=utc_now)
    last_verified_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_text(self.asset_id, "asset_id")
        _validate_sha256(self.content_sha256, "content_sha256")
        _require_text(self.remote_media_id, "remote_media_id")
        if not isinstance(self.is_active, bool):
            raise InvalidDomainValueError("is_active must be a boolean")


@dataclass
class ImageAttachment:
    """A user-uploaded image kept outside the publishing database body."""

    attachment_id: str
    thread_id: str
    filename: str
    storage_path: str
    content_type: str
    size_bytes: int
    content_sha256: str
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for value, name in (
            (self.attachment_id, "attachment_id"),
            (self.thread_id, "thread_id"),
            (self.filename, "filename"),
            (self.storage_path, "storage_path"),
            (self.content_type, "content_type"),
        ):
            _require_text(value, name)
        _validate_sha256(self.content_sha256, "content_sha256")
        if not isinstance(self.size_bytes, int) or self.size_bytes < 1:
            raise InvalidDomainValueError("size_bytes must be positive")


@dataclass
class ImageAttachmentAnalysis:
    """Derived, replaceable understanding of a user-uploaded image."""

    analysis_id: str
    attachment_id: str
    status: str = "pending"
    image_type: str | None = None
    confidence: float | None = None
    summary: str | None = None
    ocr_text: str | None = None
    structured_result: str | None = None
    analysis_model: str | None = None
    analysis_version: str = "v1"
    error_code: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for value, name in (
            (self.analysis_id, "analysis_id"),
            (self.attachment_id, "attachment_id"),
            (self.status, "status"),
            (self.analysis_version, "analysis_version"),
        ):
            _require_text(value, name)
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise InvalidDomainValueError("confidence must be between 0 and 1")

# 通用校验工具，确保某些字段是"非空字符串"
def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidDomainValueError(f"{field_name} must be a string")

    normalized = value.strip()
    if not normalized:
        raise InvalidDomainValueError(f"{field_name} must not be empty")

    return normalized

#  SHA-256 哈希来唯一标识文章内容、资源文件等
def _validate_sha256(value: str, field_name: str) -> str:
    normalized = _require_text(value, field_name).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise InvalidDomainValueError(
            f"{field_name} must be a lowercase hexadecimal SHA-256 digest"
        )

    return normalized


@dataclass(frozen=True)
class ArtifactSnapshot:
    """Validated, self-contained research content copied into an Article."""

    source_thread_id: str # 来源线程 ID
    source_artifact_id: str # 来源产物 ID
    workspace_path: str # 工作区中的文件路径
    filename: str # 文件名
    size_bytes: int # 文件大小
    sha256: str # 文件内容的 SHA-256 哈希值
    markdown_content: str # Markdown 格式的完整文章内容。

    def __post_init__(self) -> None:
        # 保证非空字符
        _require_text(self.source_thread_id, "source_thread_id")
        _require_text(self.source_artifact_id, "source_artifact_id")
        _require_text(self.workspace_path, "workspace_path")
        _require_text(self.filename, "filename")
        # 保证不重复
        _validate_sha256(self.sha256, "sha256")

        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise InvalidDomainValueError(
                "size_bytes must be a non-negative integer"
            )

        if not isinstance(self.markdown_content, str):
            raise InvalidDomainValueError(
                "markdown_content must be a string"
            )


@dataclass
class Article:
    """An editable publishing draft or an immutable published version."""

    article_id: str # 文章的唯一标识符
    source_thread_id: str
    source_artifact_id: str
    source_artifact_sha256: str # 来源产物的 SHA-256 哈希
    title: str
    slug: str # URL 友好的短标识（如 "python-async-best-practices"），用于生成文章链接；
    markdown_content: str
    excerpt: str # 文章摘要/导语。
    tags: list[str] = field(default_factory=list) # 文章标签列表，默认空列表
    status: ArticleStatus = ArticleStatus.DRAFT  # 之前 ArticleStatus 枚举
    version: int = 1 # 版本号，默认 1，每次内容更新时递增，用于乐观锁或变更追踪。
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    approved_at: datetime | None = None # 审核通过时间
    published_at: datetime | None = None # 发布时间

    def __post_init__(self) -> None:
        _require_text(self.article_id, "article_id")
        _require_text(self.source_thread_id, "source_thread_id")
        _require_text(self.source_artifact_id, "source_artifact_id")
        _validate_sha256(
            self.source_artifact_sha256,
            "source_artifact_sha256",
        )
        self.title = _require_text(self.title, "title")
        self.slug = _require_text(self.slug, "slug")

        if not isinstance(self.markdown_content, str):
            raise InvalidDomainValueError(
                "markdown_content must be a string"
            )

        if not isinstance(self.excerpt, str):
            raise InvalidDomainValueError("excerpt must be a string")

        if not isinstance(self.tags, list) or not all(
            isinstance(tag, str) and tag.strip()
            for tag in self.tags
        ):
            raise InvalidDomainValueError(
                "tags must be a list of non-empty strings"
            )

        self.tags = [tag.strip() for tag in self.tags]

        if not isinstance(self.status, ArticleStatus):
            self.status = ArticleStatus(self.status)

        if not isinstance(self.version, int) or self.version < 1:
            raise InvalidDomainValueError(
                "version must be a positive integer"
            )

    def edit(
        self,
        *,
        title: str | None = None,
        slug: str | None = None,
        markdown_content: str | None = None,
        excerpt: str | None = None,
        tags: list[str] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Edit a draft in place; published versions must be copied first."""
        # 只能就地编辑草稿，已发布的版本必须先复制（创建新版本）再编辑
        if self.status is not ArticleStatus.DRAFT:
            raise InvalidStateTransitionError(
                "only draft articles can be edited"
            )

        if title is not None:
            self.title = _require_text(title, "title")
        if slug is not None:
            self.slug = _require_text(slug, "slug")

        if markdown_content is not None:
            if not isinstance(markdown_content, str):
                raise InvalidDomainValueError(
                    "markdown_content must be a string"
                )
            self.markdown_content = markdown_content

        if excerpt is not None:
            if not isinstance(excerpt, str):
                raise InvalidDomainValueError("excerpt must be a string")
            self.excerpt = excerpt

        if tags is not None:
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) and tag.strip()
                for tag in tags
            ):
                raise InvalidDomainValueError(
                    "tags must be a list of non-empty strings"
                )
            self.tags = [tag.strip() for tag in tags]

        self.updated_at = now or utc_now()

    # 完成从"草稿"到"已审核"的状态跃迁
    def approve(self, *, now: datetime | None = None) -> None:
        self._transition_from(
            ArticleStatus.DRAFT,
            ArticleStatus.APPROVED,
        )
        timestamp = now or utc_now()
        self.approved_at = timestamp
        self.updated_at = timestamp

    def can_request_publication(self) -> bool:
        """Return whether this version may target another channel.

        Article status is an editorial/version lifecycle.  Per-channel
        delivery state belongs to ``Publication``; therefore an in-flight or
        failed delivery to one channel must not block a different channel.
        """

        return self.status in {
            ArticleStatus.APPROVED,
            ArticleStatus.PUBLISHING,
            ArticleStatus.FAILED,
            ArticleStatus.PUBLISHED,
        }

    def can_request_publication_approval(self) -> bool:
        """Return whether this version can be reviewed for one channel.

        The exact channel's durable ``Publication`` record is checked by the
        publishing workflow.  Do not use the article's global status as a
        cross-channel lock.
        """

        return self.status in {
            ArticleStatus.DRAFT,
            ArticleStatus.APPROVED,
            ArticleStatus.PUBLISHING,
            ArticleStatus.FAILED,
            ArticleStatus.PUBLISHED,
        }

    # 发行时间记录
    def start_publishing(self, *, now: datetime | None = None) -> None:
        # 只有这两种状态允许开始发布
        if self.status not in {
            ArticleStatus.APPROVED,
            ArticleStatus.FAILED,
        }:
            raise InvalidStateTransitionError(
                "only approved or failed articles can start publishing"
            )

        self.status = ArticleStatus.PUBLISHING
        self.updated_at = now or utc_now()

    # 完成发布流程的最终状态跃迁
    def mark_published(self, *, now: datetime | None = None) -> None:
        self._transition_from(
            ArticleStatus.PUBLISHING,
            ArticleStatus.PUBLISHED,
        )
        timestamp = now or utc_now()
        self.published_at = timestamp
        self.updated_at = timestamp

    # 标记失败
    def mark_failed(self, *, now: datetime | None = None) -> None:
        # 调用 _transition_from(PUBLISHING, FAILED)：确保当前状态是"发布中"才能标记为"失败
        self._transition_from(
            ArticleStatus.PUBLISHING,
            ArticleStatus.FAILED,
        )
        self.updated_at = now or utc_now()

    def return_to_approved(self, *, now: datetime | None = None) -> None:
        """Keep a version editable after a channel saves only an external draft."""

        self._transition_from(
            ArticleStatus.PUBLISHING,
            ArticleStatus.APPROVED,
        )
        self.updated_at = now or utc_now()

    # 创建新草稿版本，同时不改变当前已发布的版本
    def create_revision(
        self,
        *,
        title: str | None = None,
        slug: str | None = None,
        markdown_content: str | None = None,
        excerpt: str | None = None,
        tags: list[str] | None = None,
        now: datetime | None = None,
    ) -> Article:
        """Create a new draft version without changing the current version."""
        # 已审核或已发布版本都不能原地修改，必须生成新的草稿版本。
        if self.status not in {
            ArticleStatus.APPROVED,
            # PUBLISHING is allowed here only after the application service
            # has verified that no publication for this version is still
            # in-flight.  A delivery-unknown/failed attempt can therefore be
            # revised into a new draft without mutating the old snapshot.
            ArticleStatus.PUBLISHING,
            ArticleStatus.PUBLISHED,
            ArticleStatus.FAILED,
        }:
            raise InvalidStateTransitionError(
                "only approved, publishing, published, or failed articles can create a revision"
            )

        timestamp = now or utc_now()
        return Article(
            article_id=self.article_id,
            source_thread_id=self.source_thread_id,
            source_artifact_id=self.source_artifact_id,
            source_artifact_sha256=self.source_artifact_sha256,
            # 对每个内容字段：如果调用方传了新值就用新值，否则继承当前文章的值
            title=self.title if title is None else title,
            slug=self.slug if slug is None else slug,
            markdown_content=(
                self.markdown_content
                if markdown_content is None
                else markdown_content
            ),
            excerpt=self.excerpt if excerpt is None else excerpt,
            tags=list(self.tags) if tags is None else list(tags),
            status=ArticleStatus.DRAFT,
            version=self.version + 1,
            created_at=timestamp,
            updated_at=timestamp,
        )
    # 接收期望的当前状态 expected 和目标状态 target
    def _transition_from(
        self,
        expected: ArticleStatus,
        target: ArticleStatus,
    ) -> None:
        if self.status is not expected:
            raise InvalidStateTransitionError(
                f"cannot transition article from {self.status.value} "
                f"to {target.value}"
            )
        self.status = target


@dataclass
# 一次幂等的发布尝试
class Publication:
    """One idempotent attempt to publish one Article version to one channel."""

    publication_id: str # 本次发布尝试的唯一标识
    article_id: str # 关联的 Article 的 ID
    article_version: int # 关联的 Article 版本号
    channel: PublicationChannel # 发布渠道（之前的 PublicationChannel 枚举，如 LOCAL_STATIC_SITE）
    status: PublicationStatus # 发布状态
    idempotency_key: str # 幂等键——调用方可以用相同的 key 重试，系统据此识别是同一操作
    content_sha256: str # 发布内容的 SHA-256 哈希——用于校验内容完整性，也用于去重判断
    external_id: str | None = None # 外部系统返回的 ID
    public_url: str | None = None # 发布后的公开访问 URL
    attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    attempt_count: int = 0 # 尝试次数，初始 0，每次重试递增
    error_code: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    published_at: datetime | None = None
    # 初始化校验——文本与哈希
    def __post_init__(self) -> None:
        _require_text(self.publication_id, "publication_id")
        _require_text(self.article_id, "article_id")
        _require_text(self.idempotency_key, "idempotency_key")
        _validate_sha256(self.content_sha256, "content_sha256")

        if not isinstance(self.article_version, int) or self.article_version < 1:
            raise InvalidDomainValueError(
                "article_version must be a positive integer"
            )

        if not isinstance(self.channel, PublicationChannel):
            self.channel = PublicationChannel(self.channel)
        if not isinstance(self.status, PublicationStatus):
            self.status = PublicationStatus(self.status)

        if not isinstance(self.attempt_count, int) or self.attempt_count < 0:
            raise InvalidDomainValueError(
                "attempt_count must be a non-negative integer"
            )

        if not isinstance(self.attachment_ids, (tuple, list)):
            raise InvalidDomainValueError(
                "attachment_ids must be a tuple or list of strings"
            )
        normalized_attachment_ids = []
        for attachment_id in self.attachment_ids:
            if not isinstance(attachment_id, str) or not attachment_id.strip():
                raise InvalidDomainValueError(
                    "attachment_ids must contain non-empty strings"
                )
            normalized_attachment_ids.append(attachment_id.strip())
        self.attachment_ids = tuple(dict.fromkeys(normalized_attachment_ids))
    # 标记成功发布
    def mark_published(
        self,
        *,
        public_url: str | None = None,
        external_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        # 只有 PUBLISHING 状态才能变为 PUBLISHED
        if self.status is not PublicationStatus.PUBLISHING:
            raise InvalidStateTransitionError(
                "only publishing publications can become published"
            )

        timestamp = now or utc_now()
        self.status = PublicationStatus.PUBLISHED
        self.public_url = public_url
        self.external_id = external_id
        self.published_at = timestamp
        self.updated_at = timestamp

    def mark_drafted(
        self,
        *,
        external_id: str,
        now: datetime | None = None,
    ) -> None:
        """Record a durable external draft without claiming it is public."""

        if self.status is not PublicationStatus.PUBLISHING:
            raise InvalidStateTransitionError(
                "only publishing publications can become drafted"
            )
        self.status = PublicationStatus.DRAFTED
        self.external_id = _require_text(external_id, "external_id")
        self.updated_at = now or utc_now()

    def mark_delivery_unknown(
        self,
        *,
        error_code: str,
        now: datetime | None = None,
    ) -> None:
        """Quarantine an externally ambiguous request from automatic retry."""

        if self.status is not PublicationStatus.PUBLISHING:
            raise InvalidStateTransitionError(
                "only publishing publications can become delivery unknown"
            )
        self.status = PublicationStatus.DELIVERY_UNKNOWN
        self.error_code = _require_text(error_code, "error_code")
        self.updated_at = now or utc_now()
    # 标记发布失败
    def mark_failed(
        self,
        *,
        error_code: str,
        now: datetime | None = None,
    ) -> None:
        if self.status is not PublicationStatus.PUBLISHING:
            raise InvalidStateTransitionError(
                "only publishing publications can become failed"
            )

        self.status = PublicationStatus.FAILED
        self.error_code = _require_text(error_code, "error_code")
        self.updated_at = now or utc_now()
    # 标记重试，重试有最大次数，熔断机制
    def retry(self, *, now: datetime | None = None) -> None:
        if self.status is not PublicationStatus.FAILED:
            raise InvalidStateTransitionError(
                "only failed publications can be retried"
            )

        self.status = PublicationStatus.PUBLISHING
        self.attempt_count += 1
        self.error_code = None
        self.updated_at = now or utc_now()

    def retry_delivery_unknown(self, *, now: datetime | None = None) -> None:
        """Explicitly retry an externally ambiguous publication.

        This transition is intentionally separate from ``retry`` so callers
        cannot accidentally retry a request whose provider may already have
        applied the side effect.  The application layer must obtain an
        explicit human confirmation before calling it.
        """

        if self.status is not PublicationStatus.DELIVERY_UNKNOWN:
            raise InvalidStateTransitionError(
                "only delivery-unknown publications can be retried"
            )

        self.status = PublicationStatus.PUBLISHING
        self.attempt_count += 1
        self.error_code = None
        self.updated_at = now or utc_now()
