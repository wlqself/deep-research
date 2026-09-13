"""Small parent-child and metadata-weighting helpers for retrieval experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def build_parent_child_view(
    rows: Iterable[Mapping[str, Any]],
    *,
    parent_size: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """Group ordered child chunks into stable parent windows.

    The children remain the searchable units. ``parents`` stores the larger
    context and ``child_to_parent`` lets a caller collapse child hits to
    parent hits without changing the original child IDs.
    """
    if parent_size <= 0:
        raise ValueError("parent_size must be positive")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_row in rows:
        row = dict(raw_row)
        metadata = row.get("metadata")
        document_id = metadata.get("document_id") if isinstance(metadata, Mapping) else None
        chunk_id = row.get("_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Each row must have metadata.document_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Each row must have a non-empty _id")
        grouped[document_id].append(row)

    children: list[dict[str, Any]] = []
    parents: dict[str, dict[str, Any]] = {}
    child_to_parent: dict[str, str] = {}

    for document_id, document_rows in grouped.items():
        document_rows.sort(
            key=lambda row: (
                int(row.get("metadata", {}).get("chunk_index", 0)),
                str(row["_id"]),
            )
        )
        for parent_index in range(0, len(document_rows), parent_size):
            window = document_rows[parent_index:parent_index + parent_size]
            parent_number = parent_index // parent_size
            parent_id = f"{document_id}:parent:{parent_number}"
            first_metadata = dict(window[0].get("metadata", {}))
            parents[parent_id] = {
                "_id": parent_id,
                "title": str(window[0].get("title") or first_metadata.get("title") or document_id),
                "text": "\n\n".join(str(row["text"]) for row in window),
                "metadata": {
                    "document_id": document_id,
                    "filename": first_metadata.get("filename", ""),
                    "parent_index": parent_number,
                    "child_ids": [str(row["_id"]) for row in window],
                },
            }

            for row in window:
                child = dict(row)
                metadata = dict(child.get("metadata", {}))
                metadata["parent_id"] = parent_id
                metadata["parent_index"] = parent_number
                child["metadata"] = metadata
                children.append(child)
                child_to_parent[str(child["_id"])] = parent_id

    return children, parents, child_to_parent


def collapse_child_rankings(
    child_ids: Iterable[str],
    child_to_parent: Mapping[str, str],
    *,
    top_k: int,
) -> list[str]:
    """Deduplicate child rankings into parent rankings while preserving order."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    parent_ids: list[str] = []
    for child_id in child_ids:
        parent_id = child_to_parent.get(child_id)
        if parent_id is None or parent_id in parent_ids:
            continue
        parent_ids.append(parent_id)
        if len(parent_ids) >= top_k:
            break
    return parent_ids


def build_reranker_text(parent: Mapping[str, Any]) -> str:
    """Format one parent context for a cross-encoder reranker.

    Rerankers should see the same identity clues that help a human identify
    the source, but in a compact labelled form instead of repeated metadata.
    """
    metadata = parent.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("parent must contain metadata")

    fields = [
        ("文档标题", parent.get("title") or metadata.get("title")),
        ("文件名", metadata.get("filename")),
        ("章节", metadata.get("section_title")),
    ]
    header = [
        f"[{label}] {value.strip()}"
        for label, value in fields
        if isinstance(value, str) and value.strip()
    ]
    text = str(parent.get("text", "")).strip()
    if not text:
        raise ValueError("parent must contain non-empty text")
    return "\n".join([*header, "[正文]", text])


def build_metadata_weighted_text(
    row: Mapping[str, Any],
    *,
    metadata_weight: int = 2,
    parent_text: str | None = None,
) -> str:
    """Prefix searchable metadata and optional parent context to child text."""
    if metadata_weight < 0:
        raise ValueError("metadata_weight must be non-negative")

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("row must contain metadata")

    metadata_values = [
        str(row.get("title") or metadata.get("title") or "").strip(),
        str(metadata.get("filename", "")).strip(),
        str(metadata.get("section_title", "")).strip(),
    ]
    metadata_values = [value for value in metadata_values if value]
    prefix = " ".join(metadata_values * metadata_weight)
    child_text = str(row.get("text", "")).strip()
    if not child_text:
        raise ValueError("row must contain non-empty text")

    parts = [part for part in (prefix, parent_text or "", child_text) if part.strip()]
    return "\n".join(parts)
