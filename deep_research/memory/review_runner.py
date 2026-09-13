import asyncio
import asyncio
import logging
import time

from ..config import settings
from .extractor import MemoryExtractor
from .review_commit import (
    MemoryReviewCommitResult,
    commit_review_outcome,
)
from .reviewer import review_successful_turn
from .service import MemoryService

from ..log.logging_utils import log_event

logger = logging.getLogger(__name__)

"""
用户答案已经生成
  -> review_successful_turn()
  -> 判断触发条件
  -> 必要时检索旧记忆并调用 Extractor
  -> commit_review_outcome()
  -> 候选写入 Store
  -> 成功后推进游标
"""
async def review_after_success(
    agent,
    thread_id: str,
    *,
    answer: str,
    memory_service: MemoryService,
    extractor: MemoryExtractor,
    rule_triggered: bool = False,
    report_saved: bool = False,
    project_decision_confirmed: bool = False,
    explicit_correction: bool = False,
) -> MemoryReviewCommitResult | None:
    started_at = time.perf_counter()

    log_event(
        logger,
        logging.INFO,
        "memory.review.started",
        thread_id=thread_id,
        status="started",
    )
    try:
        totals = MemoryReviewCommitResult(
            created=0,
            updated=0,
            no_op=0,
            ignored=0,
        )

        for batch_index in range(
            settings.memory_review_max_batches_per_turn
        ):
            outcome = await review_successful_turn(
                agent,
                extractor,
                thread_id,
                answer=answer,
                rule_triggered=rule_triggered,
                report_saved=report_saved,
                project_decision_confirmed=(
                    project_decision_confirmed
                ),
                explicit_correction=explicit_correction,
                memory_service=memory_service,
                force_review=batch_index > 0,
            )

            result = await commit_review_outcome(
                agent,
                thread_id,
                memory_service,
                outcome,
            )

            totals = MemoryReviewCommitResult(
                created=totals.created + result.created,
                updated=totals.updated + result.updated,
                no_op=totals.no_op + result.no_op,
                ignored=totals.ignored + result.ignored,
            )

            if not outcome.trigger.should_review:
                break

            if not outcome.has_more_unreviewed:
                break

        log_event(
            logger,
            logging.INFO,
            "memory.review.completed",
            thread_id=thread_id,
            status="completed",
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )

        return totals

    except asyncio.CancelledError:
        log_event(
            logger,
            logging.WARNING,
            "memory.review.cancelled",
            thread_id=thread_id,
            status="cancelled",
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )
        raise

    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "memory.review.failed",
            thread_id=thread_id,
            status="failed",
            error_code="memory_review_failed",
            exception_type=type(error).__name__,
            elapsed_ms=round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
            exc_info=True,
        )
        return None
