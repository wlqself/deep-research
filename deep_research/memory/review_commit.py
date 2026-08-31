from dataclasses import dataclass
import re

from datetime import datetime, timezone

from ..state.runtime import agent_config
from .decision import decide_candidate
from .reviewer import MemoryReviewOutcome
from .service import MemoryService
from .type import (
    MemoryCandidate,
    SummaryArchive,
)
from collections.abc import Sequence

"""
未触发
  -> memory_review_turn_count + 1
  -> 不推进游标

触发且提取失败
  -> 不写 state
  -> 不重置计数器
  -> 不推进游标

触发且没有候选
  -> 提取成功
  -> 计数器清零
  -> 推进游标

触发且有候选
  -> 逐条查 key
  -> automatic 决策
  -> create/no-op/ignore
  -> 全部完成后清零并推进游标
"""
@dataclass(frozen=True)
class MemoryReviewCommitResult:
    created: int
    updated: int
    no_op: int
    ignored: int


def _archive_topics(
    candidates: Sequence[MemoryCandidate],
) -> list[str]:
    topics: list[str] = []

    for candidate in candidates:
        for keyword in candidate.keywords:
            normalized = keyword.strip()

            if normalized and normalized not in topics:
                topics.append(normalized)

    return topics[:8]

"""
有成功处理的候选
  -> recallable=True
  -> retrieval_summary 只包含候选事实

没有候选
  -> recallable=False
  -> retrieval_summary 使用安全占位文本
  -> original_summary 仍保留内部审计
"""
def _build_summary_archive(
    thread_id: str,
    outcome: MemoryReviewOutcome,
    *,
    candidates: Sequence[MemoryCandidate],
) -> SummaryArchive | None:
    
    if (
        outcome.current_summary is None
        or outcome.summary_hash is None
    ):
        return None

    original_summary = outcome.current_summary.strip()

    if not original_summary:
        return None
    
    retrieval_summary = _build_retrieval_summary(
        candidates
    )

    memory_keys = [
        _candidate_memory_identity(candidate)
        for candidate in candidates
    ]

    return SummaryArchive(
        id=f"{thread_id}:{outcome.summary_hash}",
        thread_id=thread_id,
        summary_hash=outcome.summary_hash,
        original_summary=original_summary,
        retrieval_summary=retrieval_summary,
        topics=_archive_topics(candidates),
        memory_keys=memory_keys,
        recallable=bool(candidates),
        excluded_reason=(
            None
            if candidates
            else "no_recallable_memory_facts"
        ),
        version=1,
        archived_at=datetime.now(timezone.utc),
    )

async def commit_review_outcome(
    agent,
    thread_id: str,
    memory_service: MemoryService,
    outcome: MemoryReviewOutcome,
) -> MemoryReviewCommitResult:
    # 本轮不审查，读取之前已经做好的判断
    if not outcome.trigger.should_review:
        state_update: dict[str, object] = {
            "memory_review_turn_count": (
                outcome.next_turn_count
            ),
            "memory_review_backlog_pending": (
                outcome.has_more_unreviewed
            ),
        }

        await agent.aupdate_state(
            agent_config(thread_id),
            state_update,
        )

        return MemoryReviewCommitResult(
            created=0,
            updated=0,
            no_op=0,
            ignored=0,
        )
    # 触发了，但没内容可提
    # 系统认为本轮应该复查
    # 但没有可提交的提取结果
    if outcome.extraction is None:
        return MemoryReviewCommitResult(
            created=0,
            updated=0,
            no_op=0,
            ignored=0,
        )

    created = 0
    updated = 0
    no_op = 0
    ignored = 0
    archive_candidates: list[MemoryCandidate] = []
    # 正常处理
    for candidate in outcome.extraction.candidates:
        if candidate.kind == "ignore":
            ignored += 1
            continue

        # 用户以前是否明确禁止这个 (kind, memory_key, scope) 被自动恢复？
        suppression = await memory_service.find_suppression(
            candidate.kind,
            candidate.memory_key,
            candidate.scope,
        )

        if suppression is not None:
            ignored += 1
            continue

        existing_entries = (
            await memory_service.find_active_by_key(
                candidate.kind,
                candidate.memory_key,
                candidate.scope,
            )
        )

        decision = decide_candidate(
            candidate,
            existing_entries,
            intent="automatic",
        )

        existing = (
            existing_entries[0]
            if len(existing_entries) == 1
            else None
        )
        # existing是这上面判断的，如果只有一条，说明是更新，唯一事实，0条要创建，超过一条就是矛盾
        await memory_service.apply_decision(
            candidate,
            decision,
            existing=existing,
        )
        archive_candidates.append(candidate)

        if decision.action == "create":
            created += 1
        elif decision.action == "update":
            updated += 1
        elif decision.action == "no-op":
            no_op += 1

    archive = _build_summary_archive(
        thread_id,
        outcome,
        candidates=archive_candidates,
    )

    if archive is not None:
        await memory_service.put_summary_archive(archive)

    state_update: dict[str, object] = {
        "memory_review_turn_count": (
            outcome.next_turn_count
            if outcome.has_more_unreviewed
            else 0
        ),
        "memory_review_backlog_pending": (
            outcome.has_more_unreviewed
        ),
    }

    if outcome.latest_message_id is not None:
        state_update["last_reviewed_message_id"] = (
            outcome.latest_message_id
        )

    if archive is not None:
        state_update["last_archived_summary_hash"] = (
            archive.summary_hash
        )

    await agent.aupdate_state(
        agent_config(thread_id),
        state_update,
    )

    return MemoryReviewCommitResult(
        created=created,
        updated=updated,
        no_op=no_op,
        ignored=ignored,
    )

def _candidate_memory_identity(
    candidate: MemoryCandidate,
) -> str:
    return (
        f"{candidate.kind}:"
        f"{candidate.scope}:"
        f"{candidate.memory_key.strip()}"
    )

def _build_retrieval_summary(
    candidates: Sequence[MemoryCandidate],
) -> str:
    facts: list[str] = []

    for candidate in candidates:
        title = _clean_retrieval_text(candidate.title)
        summary = _clean_retrieval_text(candidate.summary)

        if not title and not summary:
            continue

        if title and summary:
            facts.append(f"{title}: {summary}")
        elif summary:
            facts.append(summary)
        else:
            facts.append(title)

    if not facts:
        return "No recallable memory facts."

    return (
        "Recallable memory facts:\n"
        + "\n".join(f"- {fact}" for fact in facts)
    )


_OLD_SOURCE_RE = re.compile(r"\[?S\d+\]?", re.IGNORECASE)
_DROP_RETRIEVAL_LINE_PREFIXES = (
    "todo:",
    "temporary todo:",
    "tool:",
    "tool log:",
    "tool call:",
    "toolmessage:",
)


def _clean_retrieval_text(value: str) -> str:
    lines: list[str] = []

    for raw_line in value.splitlines():
        line = _OLD_SOURCE_RE.sub("", raw_line).strip()
        normalized = line.casefold()

        if not line:
            continue

        if normalized.startswith(
            _DROP_RETRIEVAL_LINE_PREFIXES
        ):
            continue

        lines.append(line)

    return " ".join(lines).strip()
