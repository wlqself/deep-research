import unittest

from deep_research.memory.review_commit import (
    _build_retrieval_summary,
    commit_review_outcome,
)
from deep_research.memory.review_types import MemoryExtractionResult
from deep_research.memory.reviewer import MemoryReviewOutcome
from deep_research.memory.triggers import MemoryTriggerDecision
from deep_research.memory.type import MemoryCandidate


class FakeAgent:
    def __init__(self) -> None:
        self.update_calls: list[tuple[object, dict[str, object]]] = []

    async def aupdate_state(
        self,
        config: object,
        update: dict[str, object],
    ) -> None:
        self.update_calls.append((config, update))


class FakeMemoryService:
    def __init__(
        self,
        *,
        fail: bool = False,
        archive_fail: bool = False,
        archive_created: bool = True,
        suppressed_keys: set[str] | None = None,
    ) -> None:
        self.fail = fail
        self.archive_fail = archive_fail
        self.archive_created = archive_created
        self.suppressed_keys = suppressed_keys or set()
        self.lookup_calls: list[tuple[str, str, str]] = []
        self.suppression_calls: list[tuple[str, str, str]] = []
        self.apply_calls: list[tuple[object, object, object]] = []
        self.archive_calls: list[object] = []

    async def find_active_by_key(
        self,
        kind: str,
        memory_key: str,
        scope: str,
    ) -> list[object]:
        self.lookup_calls.append((kind, memory_key, scope))
        return []

    async def find_suppression(
        self,
        kind: str,
        memory_key: str,
        scope: str,
    ) -> object | None:
        self.suppression_calls.append((kind, memory_key, scope))
        suppression_id = f"{kind}:{scope}:{memory_key}"
        return object() if suppression_id in self.suppressed_keys else None

    async def apply_decision(
        self,
        candidate: object,
        decision: object,
        *,
        existing: object = None,
    ) -> None:
        if self.fail:
            raise RuntimeError("candidate store failed")
        self.apply_calls.append((candidate, decision, existing))

    async def put_summary_archive(self, archive: object) -> bool:
        if self.archive_fail:
            raise RuntimeError("summary archive failed")

        self.archive_calls.append(archive)
        return self.archive_created


def make_candidate(
    *,
    kind: str = "user",
    memory_key: str = "user.answer_style.conciseness",
) -> MemoryCandidate:
    return MemoryCandidate.model_validate(
        {
            "kind": kind,
            "memory_key": memory_key,
            "scope": "global",
            "title": "Answer style",
            "summary": "The user prefers concise answers.",
            "content": "Prefer concise answers.",
            "keywords": ["concise"],
            "source_type": "automatic_extraction",
            "source_thread_id": "thread-1",
        }
    )


def make_outcome(
    *,
    should_review: bool,
    extraction: MemoryExtractionResult | None,
    latest_message_id: str | None = "message-2",
    has_more_unreviewed: bool = False,
    next_turn_count: int = 10,
) -> MemoryReviewOutcome:
    return MemoryReviewOutcome(
        trigger=MemoryTriggerDecision(
            should_review=should_review,
            reasons=["interval"] if should_review else [],
        ),
        extraction=extraction,
        latest_message_id=latest_message_id,
        has_more_unreviewed=has_more_unreviewed,
        next_turn_count=next_turn_count,
        current_summary="Current summary.",
        summary_hash="summary-hash",
    )


class MemoryReviewCommitTests(unittest.IsolatedAsyncioTestCase):
    def test_retrieval_summary_removes_legacy_sources_and_tool_lines(self):
        candidates = [
            make_candidate(
                memory_key="user.answer_style.conciseness",
            ).model_copy(
                update={
                    "title": "Answer style [S1]",
                    "summary": (
                        "Keep answers concise.\n"
                        "Temporary Todo: clean up later.\n"
                        "Tool log: web search output."
                    ),
                }
            )
        ]

        rendered = _build_retrieval_summary(candidates)

        self.assertIn("Answer style", rendered)
        self.assertIn("Keep answers concise.", rendered)
        self.assertNotIn("S1", rendered)
        self.assertNotIn("Temporary Todo", rendered)
        self.assertNotIn("Tool log", rendered)

    async def test_untriggered_turn_only_persists_incremented_counter(self):
        agent = FakeAgent()
        service = FakeMemoryService()

        result = await commit_review_outcome(
            agent,
            "thread-1",
            service,
            make_outcome(
                should_review=False,
                extraction=None,
                next_turn_count=4,
            ),
        )

        self.assertEqual(result.created, 0)
        self.assertEqual(service.lookup_calls, [])
        self.assertEqual(
            agent.update_calls[0][1],
            {
                "memory_review_turn_count": 4,
                "memory_review_backlog_pending": False,
            },
        )

    async def test_successful_extraction_commits_candidates_and_cursor(self):
        agent = FakeAgent()
        service = FakeMemoryService()
        extraction = MemoryExtractionResult(
            candidates=[
                make_candidate(),
                make_candidate(
                    kind="ignore",
                    memory_key="user.temporary.request",
                ),
            ]
        )

        result = await commit_review_outcome(
            agent,
            "thread-1",
            service,
            make_outcome(
                should_review=True,
                extraction=extraction,
            ),
        )

        self.assertEqual(result.created, 1)
        self.assertEqual(result.ignored, 1)
        self.assertEqual(len(service.lookup_calls), 1)
        self.assertEqual(len(service.apply_calls), 1)
        self.assertEqual(len(service.archive_calls), 1)
        self.assertEqual(
            service.archive_calls[0].summary_hash,
            "summary-hash",
        )
        self.assertEqual(
            agent.update_calls[0][1],
            {
                "memory_review_turn_count": 0,
                "last_reviewed_message_id": "message-2",
                "last_archived_summary_hash": "summary-hash",
                "memory_review_backlog_pending": False,
            },
        )

    async def test_partial_batch_keeps_backlog_and_cursor_progress(self):
        agent = FakeAgent()
        service = FakeMemoryService()

        await commit_review_outcome(
            agent,
            "thread-1",
            service,
            make_outcome(
                should_review=True,
                extraction=MemoryExtractionResult(candidates=[]),
                has_more_unreviewed=True,
                next_turn_count=11,
            ),
        )

        state_update = agent.update_calls[0][1]
        self.assertEqual(
            state_update["memory_review_turn_count"],
            11,
        )
        self.assertTrue(
            state_update["memory_review_backlog_pending"]
        )
        self.assertEqual(
            state_update["last_reviewed_message_id"],
            "message-2",
        )

    async def test_empty_extraction_is_successful_and_commits_cursor(self):
        agent = FakeAgent()
        service = FakeMemoryService()

        result = await commit_review_outcome(
            agent,
            "thread-1",
            service,
            make_outcome(
                should_review=True,
                extraction=MemoryExtractionResult(candidates=[]),
            ),
        )

        self.assertEqual(result.created, 0)
        self.assertEqual(
            agent.update_calls[0][1]["memory_review_turn_count"],
            0,
        )
        self.assertEqual(
            agent.update_calls[0][1]["last_reviewed_message_id"],
            "message-2",
        )
        self.assertEqual(
            agent.update_calls[0][1]["last_archived_summary_hash"],
            "summary-hash",
        )

    async def test_existing_archive_still_commits_review_state(self):
        agent = FakeAgent()
        service = FakeMemoryService(archive_created=False)

        await commit_review_outcome(
            agent,
            "thread-1",
            service,
            make_outcome(
                should_review=True,
                extraction=MemoryExtractionResult(candidates=[]),
            ),
        )

        self.assertEqual(
            agent.update_calls[0][1]["last_archived_summary_hash"],
            "summary-hash",
        )

    async def test_archive_failure_does_not_commit_review_state(self):
        agent = FakeAgent()
        service = FakeMemoryService(archive_fail=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "summary archive failed",
        ):
            await commit_review_outcome(
                agent,
                "thread-1",
                service,
                make_outcome(
                    should_review=True,
                    extraction=MemoryExtractionResult(candidates=[]),
                ),
            )

        self.assertEqual(agent.update_calls, [])

    async def test_candidate_failure_does_not_commit_cursor(self):
        agent = FakeAgent()
        service = FakeMemoryService(fail=True)

        with self.assertRaisesRegex(RuntimeError, "candidate store failed"):
            await commit_review_outcome(
                agent,
                "thread-1",
                service,
                make_outcome(
                    should_review=True,
                    extraction=MemoryExtractionResult(
                        candidates=[make_candidate()]
                    ),
                ),
            )

        self.assertEqual(agent.update_calls, [])

    async def test_suppressed_candidate_is_ignored_without_store_write(self):
        agent = FakeAgent()
        candidate = make_candidate()
        service = FakeMemoryService(
            suppressed_keys={
                "user:global:user.answer_style.conciseness"
            }
        )

        result = await commit_review_outcome(
            agent,
            "thread-1",
            service,
            make_outcome(
                should_review=True,
                extraction=MemoryExtractionResult(
                    candidates=[candidate]
                ),
            ),
        )

        self.assertEqual(result.ignored, 1)
        self.assertEqual(service.lookup_calls, [])
        self.assertEqual(service.apply_calls, [])


if __name__ == "__main__":
    unittest.main()
