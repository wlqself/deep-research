import hashlib
import hashlib

from collections.abc import Sequence
from dataclasses import dataclass

from ..config import settings
from ..state.access import thread_values
from .extractor import MemoryExtractor
from .review_input import build_review_input
from .review_types import (
    MemoryExtractionResult,
)
from .triggers import (
    MemoryTriggerDecision,
    MemoryTriggerInput,
    decide_memory_trigger,
)
from .type import MemoryEntry
from .service import MemoryService

@dataclass(frozen=True)
class MemoryReviewOutcome:
    trigger: MemoryTriggerDecision
    extraction: MemoryExtractionResult | None
    latest_message_id: str | None
    has_more_unreviewed: bool # 当前这一次 build_review_input() 生成的 batch 后面，是否还存在没有纳入本批的消息
    next_turn_count: int
    current_summary: str | None
    summary_hash: str | None

# 计算摘要 hash
def _summary_hash(summary: str | None) -> str | None:
    if not summary:
        return None

    return hashlib.sha256(
        summary.encode("utf-8")
    ).hexdigest()


async def review_successful_turn(
    agent,
    extractor: MemoryExtractor,
    thread_id: str,
    *,
    answer: str,
    rule_triggered: bool = False,
    report_saved: bool = False,
    project_decision_confirmed: bool = False,
    explicit_correction: bool = False,
    similar_old_memories: Sequence[MemoryEntry] = (),
    memory_service: MemoryService | None = None,
    force_review: bool = False,
) -> MemoryReviewOutcome:
    # 读取线程数据
    values = await thread_values(
        agent,
        thread_id,
    )
    """
    如果字段缺失 / 非 int / 负数 → 退化成 0
    无论如何都 +1，保证「成功轮次」被计数
    这个 next_turn_count 会传给触发器做「间隔触发」判断
    """
    raw_count = values.get(
        "memory_review_turn_count",
        0,
    )

    current_count = (
        raw_count
        if isinstance(raw_count, int) and raw_count >= 0 
        else 0
    )

    next_turn_count = current_count + 1

    raw_cursor = values.get(
        "last_reviewed_message_id"
    )

    last_reviewed_message_id = (
        raw_cursor
        if isinstance(raw_cursor, str) # 游标必须是str类型
        else None
    )

    raw_backlog = values.get(
        "memory_review_backlog_pending",
        False,
    )
    backlog_pending = (
        raw_backlog
        if isinstance(raw_backlog, bool)
        else False
    )

    review_input, latest_message_id = build_review_input(
        values,
        last_reviewed_message_id=last_reviewed_message_id,
        similar_old_memories=similar_old_memories,
    )
    # 计算摘要哈希，看内容是否发生变化
    current_hash = _summary_hash(
        review_input.current_summary
    )

    raw_archived_hash = values.get(
        "last_archived_summary_hash"
    )

    last_archived_hash = (
        raw_archived_hash
        if isinstance(raw_archived_hash, str)
        else None
    )

    # 要不要触发记忆提取
    trigger = decide_memory_trigger(
        MemoryTriggerInput(
            success=True,
            answer=answer,
            rule_triggered=rule_triggered,
            summary_changed=(
                current_hash is not None
                and current_hash != last_archived_hash
            ),
            report_saved=report_saved,
            project_decision_confirmed=(
                project_decision_confirmed
            ),
            explicit_correction=explicit_correction,
            successful_turn_count=next_turn_count,
            review_interval=(
                settings.memory_review_interval_turns
            ),
        )
    )

    if (force_review or backlog_pending) and not trigger.should_review:
        trigger = trigger.model_copy(
            update={
                "should_review": True,
                "reasons": [*trigger.reasons, "backlog"],
            }
        )
    elif force_review or backlog_pending:
        if "backlog" not in trigger.reasons:
            trigger = trigger.model_copy(
                update={
                    "reasons": [*trigger.reasons, "backlog"],
                }
            )

    if not trigger.should_review: # 是否要调 LLM 提取
        return MemoryReviewOutcome(
            trigger=trigger,
            extraction=None,
            latest_message_id=latest_message_id,
            next_turn_count=next_turn_count,
            current_summary=review_input.current_summary,
            summary_hash=current_hash,
            has_more_unreviewed=(
                review_input.review_batch.has_more_unreviewed
            ),
        )
    # 如果既没有历史摘要，也没有任何未审查的新消息​ → 没有任何内容可提取 → 直接返回，extraction=None，避免一次无意义的 LLM 调用。
    if (
        review_input.current_summary is None
        and not review_input.unreviewed_messages
    ):
        return MemoryReviewOutcome(
            trigger=trigger,
            extraction=None,
            latest_message_id=latest_message_id,
            next_turn_count=next_turn_count,
            current_summary=review_input.current_summary,
            summary_hash=current_hash,
            has_more_unreviewed=(
                review_input.review_batch.has_more_unreviewed
            ),
        )
    
    if (
        memory_service is not None
        and not similar_old_memories
    ):
        query = _review_query(review_input)

        similar_old_memories = (
            await memory_service.find_review_memories(
                query,
                limit=settings.memory_recall_limit,
            )
        )

        review_input = review_input.model_copy(
            update={
                "similar_old_memories": list(
                    similar_old_memories
                ),
            }
        )
    extraction = await extractor.extract(
        review_input
    )

    return MemoryReviewOutcome(
        trigger=trigger,
        extraction=extraction,
        latest_message_id=latest_message_id,
        next_turn_count=next_turn_count,
        current_summary=review_input.current_summary,
        summary_hash=current_hash,
        has_more_unreviewed=(
            review_input.review_batch.has_more_unreviewed
        ),
    )

def _review_query(review_input) -> str:
    # 按「摘要 → 用户 → 助手」顺序收集上下文。摘要提供长期话题，后两条提供"刚才具体在问/答什么"
    parts = [
        review_input.current_summary,
        review_input.latest_user_message,
        review_input.latest_main_message,
    ]

    return " ".join(
        part.strip()
        for part in parts
        if isinstance(part, str) and part.strip()
    )
