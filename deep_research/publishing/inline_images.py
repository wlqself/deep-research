"""Shared parsing helpers for images embedded in article Markdown.

Articles keep image references as ``attachment://<attachment_id>`` instead of
copying local paths or remote URLs into the canonical content.  Platform
adapters resolve those references at publish time, which keeps the article
portable and prevents an editor from smuggling arbitrary files into a
publisher.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Iterable
from collections.abc import Mapping


_ATTACHMENT_ID = r"[A-Za-z0-9_-]{1,128}"
ATTACHMENT_IMAGE_PATTERN = re.compile(
    rf"!\[(?P<alt>[^\]]*)\]\(\s*attachment://(?P<attachment_id>{_ATTACHMENT_ID})"
    r"(?:\s+\"[^\"]*\")?\s*\)"
)


@dataclass(frozen=True)
class InlineImageReference:
    """One ordered image reference in an article body."""

    attachment_id: str
    alt: str = ""


def extract_inline_image_references(markdown: str) -> tuple[InlineImageReference, ...]:
    """Return valid attachment references in source order.

    Duplicate references are retained because the same image may intentionally
    appear twice in a WeChat article.  Publishers can de-duplicate uploads
    separately while still rendering both positions.
    """

    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")
    return tuple(
        InlineImageReference(
            attachment_id=match.group("attachment_id"),
            alt=match.group("alt").strip(),
        )
        for match in ATTACHMENT_IMAGE_PATTERN.finditer(markdown)
    )


def ordered_unique_attachment_ids(
    inline_references: Iterable[InlineImageReference],
    selected_attachment_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Combine body references and explicitly selected gallery images."""

    result: list[str] = []
    seen: set[str] = set()
    for candidate in (
        [reference.attachment_id for reference in inline_references]
        + list(selected_attachment_ids)
    ):
        if not isinstance(candidate, str):
            continue
        value = candidate.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def replace_inline_attachment_urls(
    markdown: str,
    urls_by_attachment_id: Mapping[str, str],
) -> str:
    """Replace typed attachment targets while preserving Markdown and alt text."""

    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")

    def replace(match: re.Match[str]) -> str:
        attachment_id = match.group("attachment_id")
        target = urls_by_attachment_id.get(attachment_id)
        if not isinstance(target, str) or not target.strip():
            return match.group(0)
        return f"![{match.group('alt')}]({target.strip()})"

    return ATTACHMENT_IMAGE_PATTERN.sub(replace, markdown)


__all__ = [
    "ATTACHMENT_IMAGE_PATTERN",
    "InlineImageReference",
    "extract_inline_image_references",
    "ordered_unique_attachment_ids",
    "replace_inline_attachment_urls",
]
