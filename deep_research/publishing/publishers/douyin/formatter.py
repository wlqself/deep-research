"""Deterministic conversion of an Article into a Douyin image-text post."""

from __future__ import annotations

from dataclasses import dataclass
import re

from ...models import Article
from ...inline_images import ATTACHMENT_IMAGE_PATTERN
from ...service_support import PublicationValidationError, normalized_tags


_IMAGE_MARKDOWN = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_MARKDOWN = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_CODE_FENCE = re.compile(r"^\s*```.*$", re.MULTILINE)
_INLINE_MARKDOWN = re.compile(r"[*_~`]")


@dataclass(frozen=True)
class DouyinContent:
    title: str
    content: str
    tags: tuple[str, ...]


def _plain_text(markdown: str) -> str:
    # Douyin receives a gallery array, so remove typed body image markers from
    # the text instead of exposing their alt text as a local filename.
    text = ATTACHMENT_IMAGE_PATTERN.sub("", markdown)
    text = _IMAGE_MARKDOWN.sub(r"\1", text)
    text = _LINK_MARKDOWN.sub(r"\1", text)
    text = _CODE_FENCE.sub("", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*+]\s+", "• ", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = _INLINE_MARKDOWN.sub("", line)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def format_douyin_content(
    article: Article,
    *,
    max_title_length: int = 30,
) -> DouyinContent:
    """Build a safe image-text payload without mutating the Article."""

    if not isinstance(max_title_length, int) or max_title_length < 1:
        raise ValueError("max_title_length must be positive")
    title = " ".join(article.title.split())
    if not title:
        raise PublicationValidationError(
            "douyin_title_empty",
            "Douyin title must not be empty",
        )
    if len(title) > max_title_length:
        raise PublicationValidationError(
            "douyin_title_too_long",
            "Douyin title exceeds the configured limit",
        )

    content = _plain_text(article.markdown_content)
    if not content:
        raise PublicationValidationError(
            "douyin_content_empty",
            "Douyin content must not be empty",
        )

    tags = tuple(normalized_tags(article.tags))
    return DouyinContent(title=title, content=content, tags=tags)


__all__ = ["DouyinContent", "format_douyin_content"]
