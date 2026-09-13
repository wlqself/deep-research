"""Deterministic conversion of an Article into a Xiaohongshu note."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from ...models import Article
from ...inline_images import ATTACHMENT_IMAGE_PATTERN
from ...service_support import PublicationValidationError, normalized_tags


_IMAGE_MARKDOWN = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_MARKDOWN = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_CODE_FENCE = re.compile(r"^\s*```.*$", re.MULTILINE)
_INLINE_MARKDOWN = re.compile(r"[*_~`]")
_HASHTAG = re.compile(
    r"#([^\s#，。！？、；：,!?;:()（）\[\]{}<>《》【】]+)"
)
_XIAOHONGSHU_TAG_ALLOWED = re.compile(r"^[0-9A-Za-z\u3400-\u9fff]+$")


@dataclass(frozen=True)
class XiaohongshuContent:
    title: str
    content: str
    tags: tuple[str, ...]


def _normalize_xiaohongshu_tags(tags: list[str]) -> tuple[str, ...]:
    """Return platform-safe topic tags without mutating the Article.

    Xiaohongshu's browser topic input rejects punctuation such as the dot in
    ``V4.1``.  Tags are a platform-specific projection, so preserve normal
    punctuation in title/body but remove unsupported tag characters here.
    This also makes ``#V4.1`` and ``V4.1`` converge to the same safe tag.
    """

    safe_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        if not isinstance(raw_tag, str):
            continue
        value = unicodedata.normalize("NFKC", raw_tag).strip().lstrip("#")
        value = re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", value)
        if not value or not _XIAOHONGSHU_TAG_ALLOWED.fullmatch(value):
            continue
        key = value.casefold()
        if key in seen:
            continue
        safe_tags.append(value)
        seen.add(key)
    return tuple(safe_tags)


def _plain_text(markdown: str) -> tuple[str, list[str]]:
    # Body attachment references become gallery images on Xiaohongshu; their
    # Markdown alt text must not leak into the note body as a filename.
    text = ATTACHMENT_IMAGE_PATTERN.sub("", markdown)
    text = _IMAGE_MARKDOWN.sub(r"\1", text)
    text = _LINK_MARKDOWN.sub(r"\1", text)
    text = _CODE_FENCE.sub("", text)
    embedded_tags: list[str] = []

    def remove_hashtag(match: re.Match[str]) -> str:
        tag = match.group(1).strip()
        if tag:
            embedded_tags.append(tag)
        return " "

    # Xiaohongshu receives tags through a dedicated field.  Remove hashtags
    # from the body first, otherwise the browser service appends the same
    # labels again and the published note contains duplicates.
    text = _HASHTAG.sub(remove_hashtag, text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*+]\s+", "• ", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = _INLINE_MARKDOWN.sub("", line)
        if line:
            lines.append(line)
    return "\n".join(lines).strip(), embedded_tags


def format_xiaohongshu_content(
    article: Article,
    *,
    max_title_length: int = 20,
    max_content_length: int = 1000,
) -> XiaohongshuContent:
    """Build a safe note payload without mutating the Article."""

    if not isinstance(max_title_length, int) or max_title_length < 1:
        raise ValueError("max_title_length must be positive")
    if not isinstance(max_content_length, int) or max_content_length < 1:
        raise ValueError("max_content_length must be positive")
    title = " ".join(article.title.split())
    if not title:
        raise PublicationValidationError(
            "xiaohongshu_title_empty",
            "Xiaohongshu title must not be empty",
        )
    if len(title) > max_title_length:
        raise PublicationValidationError(
            "xiaohongshu_title_too_long",
            (
                "Xiaohongshu title is too long: "
                f"{len(title)} characters; maximum is {max_title_length}."
            ),
        )

    content, embedded_tags = _plain_text(article.markdown_content)
    if not content:
        raise PublicationValidationError(
            "xiaohongshu_content_empty",
            "Xiaohongshu content must not be empty",
        )
    if len(content) > max_content_length:
        raise PublicationValidationError(
            "xiaohongshu_content_too_long",
            (
                "Xiaohongshu content is too long: "
                f"{len(content)} characters; maximum is {max_content_length}."
            ),
        )

    tags = _normalize_xiaohongshu_tags(
        normalized_tags([*(article.tags or []), *embedded_tags])
    )
    return XiaohongshuContent(title=title, content=content, tags=tags)


__all__ = ["XiaohongshuContent", "format_xiaohongshu_content"]
