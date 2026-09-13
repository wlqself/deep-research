"""Shared contracts and deterministic helpers for publishing services."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath
from typing import Protocol

from .models import (
    Article,
    PublicationChannel,
    PublicationStatus,
    PublishingDomainError,
)

# 发布服务层所有异常的基类，继承领域错误 PublishingDomainError
class PublishingServiceError(PublishingDomainError):
    """Base exception for application-level publishing failures."""

class InvalidSlugError(PublishingServiceError):
    """Raised when a requested article slug violates the public contract."""

    error_code = "invalid_slug"
    retryable = False

# 当外部制品（Artifact）无法被导入为唯一文章草稿时抛出
class ArticleImportConflictError(PublishingServiceError):
    """Raised when an Artifact cannot become a unique Article draft."""

# 当请求的发布渠道没有对应的发布器配置时抛出
class PublisherConfigurationError(PublishingServiceError):
    """Raised when a publisher does not match the requested channel."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "unsupported_publication_channel",
    ) -> None:
        self.error_code = error_code.strip() or "unsupported_publication_channel"
        super().__init__(message)

# 渠道发布器执行失败时抛出的结构化错误
class PublicationExecutionError(PublishingServiceError):
    """A safe, stable error from a channel publisher."""

    def __init__(self, error_code: str) -> None:
        normalized = error_code.strip()
        self.error_code = normalized or "publisher_failed" # 空值兜底为 "publisher_failed"
        super().__init__(self.error_code)


class PublicationDeliveryUnknownError(PublicationExecutionError):
    """The provider may have accepted a side effect but gave no receipt."""

    def __init__(self, error_code: str, *, external_id: str | None = None) -> None:
        super().__init__(error_code)
        self.external_id = external_id


class PublicationPollingError(PublicationExecutionError):
    """A status read failed transiently; keep the publication in flight."""


class PublicationValidationError(PublishingServiceError):
    """Raised when an Article violates a channel input contract."""

    def __init__(self, error_code: str, message: str) -> None:
        normalized = error_code.strip()
        self.error_code = normalized or "publication_validation_failed"
        super().__init__(message)


@dataclass(frozen=True)
class PublicationResult:
    """Channel output that is safe to persist and return to the caller."""

    public_url: str | None = None # 发布后的公开访问地址
    external_id: str | None = None # 渠道侧的唯一 ID
    status: PublicationStatus = PublicationStatus.PUBLISHED # 发布状态，默认 PUBLISHED


class PublicationPublisher(Protocol):
    """Interface implemented by a concrete channel publisher."""

    channel: PublicationChannel # 必须有 channel 属性标识渠道类型，以及 publish(article) 方法

    def publish(
        self,
        article: Article,
        *,
        attachment_ids: tuple[str, ...] = (),
    ) -> PublicationResult:
        """Publish one approved Article version."""


_SLUG_MAX_LENGTH = 120
_SLUG_SEPARATOR_PATTERN = re.compile(r"-+")

# 规范化
def validate_slug(slug: str) -> str:
    """Normalize and validate a URL path segment used by the static site."""

    if not isinstance(slug, str):
        raise PublishingServiceError("slug must be a string")
    # Unicode 规范化：unicodedata.normalize("NFKC", slug) 将字符转为兼容等价形式（如全角字母转半角）
    normalized = unicodedata.normalize("NFKC", slug).strip().casefold() # .casefold() 做不区分大小写的比较和存储
    if (
        not normalized
        or len(normalized) > _SLUG_MAX_LENGTH
        or normalized.startswith("-")
        or normalized.endswith("-")
        or any(
            not (character.isalnum() or character == "-")
            for character in normalized
        )
    ):
        raise InvalidSlugError("invalid slug")

    return normalized

# 从标题生成 slug
def slugify_title(title: str) -> str:
    """Create a safe slug from an Article title."""

    normalized = unicodedata.normalize("NFKC", title).casefold()
    slug = "".join(
        character if character.isalnum() else "-"
        for character in normalized
    )
    slug = _SLUG_SEPARATOR_PATTERN.sub("-", slug).strip("-")
    return (
        validate_slug(slug[:_SLUG_MAX_LENGTH].strip("-"))
        if slug
        else "article"
    )

# 从文件名提取标题
def title_from_filename(filename: str) -> str:
    """Derive a readable fallback title from an Artifact filename."""

    stem = PurePath(filename).stem
    title = re.sub(r"[-_]+", " ", stem).strip()
    return title or "Research article"

# 提取摘要
def excerpt_from_markdown(markdown_content: str) -> str:
    """Extract a bounded plain-text excerpt from Markdown paragraphs."""

    paragraphs = [
        " ".join(line.strip() for line in block.splitlines()).strip()
        for block in re.split(r"\n\s*\n", markdown_content)
    ]
    excerpt = next((paragraph for paragraph in paragraphs if paragraph), "")
    return " ".join(excerpt.split())[:240]

# 标签规范化
def normalized_tags(tags: list[str] | None) -> list[str]:
    """Normalize tags while preserving their first-seen display spelling."""

    if tags is None:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = tag.strip()
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            normalized.append(value)
            seen.add(key)
    return normalized


__all__ = [
    "ArticleImportConflictError",
    "InvalidSlugError",
    "PublicationExecutionError",
    "PublicationDeliveryUnknownError",
    "PublicationPollingError",
    "PublicationValidationError",
    "PublicationPublisher",
    "PublicationResult",
    "PublisherConfigurationError",
    "PublishingServiceError",
    "excerpt_from_markdown",
    "normalized_tags",
    "slugify_title",
    "title_from_filename",
    "validate_slug",
]
