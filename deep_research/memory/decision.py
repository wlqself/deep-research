from collections.abc import Sequence

from .type import (
    MemoryCandidate,
    MemoryDecision,
    MemoryEntry,
    MemoryWriteIntent,
)


def _normalized_signature(
    value: MemoryCandidate | MemoryEntry,
) -> tuple[object, ...]:
    return (
        value.kind,
        value.memory_key,
        value.scope,
        # 去除首尾空格
        value.title.strip(),
        value.summary.strip(),
        value.content.strip(),
        # 去除空格并转化为小写
        tuple(sorted(keyword.strip().lower() for keyword in value.keywords)),
        str(value.url) if value.url is not None else None,
        value.incorrect.strip() if value.incorrect else None,
        value.correct.strip() if value.correct else None,
        value.applies_when.strip() if value.applies_when else None,
    )


def decide_candidate(
    candidate: MemoryCandidate,
    same_key_entries: Sequence[MemoryEntry],
    *,
    intent: MemoryWriteIntent,
) -> MemoryDecision:
    # 判断候选记忆类型
    if candidate.kind == "ignore":
        return MemoryDecision(
            action="ignore",
            reason="candidate_marked_ignore",
        )

    matching_entries = [
        entry
        for entry in same_key_entries
        if (
            entry.kind == candidate.kind
            and entry.memory_key == candidate.memory_key
            and entry.scope == candidate.scope
            and entry.status == "active"
        )
    ]

    if len(matching_entries) > 1:
        raise ValueError(
            "multiple active entries share the same memory_key"
        )
    # 没有同 kind、同 key、同 scope 的 active 记录
    if not matching_entries:
        return MemoryDecision(
            action="create",
            reason="no_active_entry_with_same_key",
        )

    existing = matching_entries[0]
    # 同 key 且内容等价
    if _normalized_signature(candidate) == _normalized_signature(existing):
        return MemoryDecision(
            action="no-op",
            existing_id=existing.id,
            reason="same_content",
        )
    # 同 key 内容不同，明确修改
    if intent == "explicit_update":
        return MemoryDecision(
            action="update",
            existing_id=existing.id,
            reason="explicit_update_same_key",
        )
    # 同 key 内容不同，但只是自动提取或普通记住
    return MemoryDecision(
        action="no-op",
        existing_id=existing.id,
        reason="conflict_without_explicit_update",
    )