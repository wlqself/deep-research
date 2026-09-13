"""Build embedding text without changing the content returned to users."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ORIGINAL_CONTENT_SEPARATOR = "\n__DEEP_RESEARCH_ORIGINAL_CONTENT__\n"


def build_metadata_weighted_embedding_text(
    content: str,
    metadata: Mapping[str, Any],
    *,
    weight: int = 2,
) -> str:
    """Prefix title-like metadata for embedding while retaining raw content."""
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if not content.strip():
        raise ValueError("content must not be empty")

    values = [
        str(metadata.get("title", "")).strip(),
        str(metadata.get("filename", "")).strip(),
        str(metadata.get("section_title", "")).strip(),
    ]
    metadata_text = " ".join(value for value in values if value)
    prefix = ((metadata_text + " ") * weight).strip()
    if not prefix:
        return content
    return f"{prefix}{ORIGINAL_CONTENT_SEPARATOR}{content}"


def restore_original_content(content: str) -> str:
    """Remove the internal embedding prefix from a retrieved document."""
    if ORIGINAL_CONTENT_SEPARATOR not in content:
        return content
    return content.split(ORIGINAL_CONTENT_SEPARATOR, 1)[1]
