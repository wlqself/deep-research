import asyncio
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from deep_research.memory.review_commit import MemoryReviewCommitResult
from deep_research.memory.review_runner import review_after_success


class MemoryReviewRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_review_runs_and_returns_commit_result(self):
        outcome = SimpleNamespace(
            trigger=SimpleNamespace(should_review=True),
            has_more_unreviewed=False,
        )
        commit_result = MemoryReviewCommitResult(
            created=1,
            updated=0,
            no_op=0,
            ignored=0,
        )

        with patch(
            "deep_research.memory.review_runner.review_successful_turn",
            new=AsyncMock(return_value=outcome),
        ) as review, patch(
            "deep_research.memory.review_runner.commit_review_outcome",
            new=AsyncMock(return_value=commit_result),
        ) as commit:
            result = await review_after_success(
                "agent",
                "thread-1",
                answer="A successful answer.",
                memory_service="memory-service",
                extractor="extractor",
                explicit_correction=True,
            )

        self.assertEqual(result, commit_result)
        review.assert_awaited_once()
        commit.assert_awaited_once_with(
            "agent",
            "thread-1",
            "memory-service",
            outcome,
        )
        self.assertTrue(
            review.await_args.kwargs["explicit_correction"]
        )

    async def test_processes_at_most_configured_batches_and_accumulates_results(self):
        outcomes = [
            SimpleNamespace(
                trigger=SimpleNamespace(should_review=True),
                has_more_unreviewed=True,
            ),
            SimpleNamespace(
                trigger=SimpleNamespace(should_review=True),
                has_more_unreviewed=False,
            ),
            SimpleNamespace(
                trigger=SimpleNamespace(should_review=True),
                has_more_unreviewed=True,
            ),
        ]
        commit_results = [
            MemoryReviewCommitResult(
                created=1,
                updated=0,
                no_op=0,
                ignored=1,
            ),
            MemoryReviewCommitResult(
                created=0,
                updated=1,
                no_op=2,
                ignored=0,
            ),
        ]

        with patch(
            "deep_research.memory.review_runner.review_successful_turn",
            new=AsyncMock(side_effect=outcomes),
        ) as review, patch(
            "deep_research.memory.review_runner.commit_review_outcome",
            new=AsyncMock(side_effect=commit_results),
        ) as commit:
            result = await review_after_success(
                "agent",
                "thread-1",
                answer="A successful answer.",
                memory_service="memory-service",
                extractor="extractor",
            )

        self.assertEqual(
            result,
            MemoryReviewCommitResult(
                created=1,
                updated=1,
                no_op=2,
                ignored=1,
            ),
        )
        self.assertEqual(review.await_count, 2)
        self.assertEqual(commit.await_count, 2)
        self.assertFalse(review.await_args_list[0].kwargs["force_review"])
        self.assertTrue(review.await_args_list[1].kwargs["force_review"])

    async def test_untriggered_turn_with_backlog_stops_until_next_success(self):
        outcome = SimpleNamespace(
            trigger=SimpleNamespace(should_review=False),
            has_more_unreviewed=True,
        )
        commit_result = MemoryReviewCommitResult(
            created=0,
            updated=0,
            no_op=0,
            ignored=0,
        )

        with patch(
            "deep_research.memory.review_runner.review_successful_turn",
            new=AsyncMock(return_value=outcome),
        ) as review, patch(
            "deep_research.memory.review_runner.commit_review_outcome",
            new=AsyncMock(return_value=commit_result),
        ) as commit:
            result = await review_after_success(
                "agent",
                "thread-1",
                answer="A successful answer.",
                memory_service="memory-service",
                extractor="extractor",
            )

        self.assertEqual(result, commit_result)
        review.assert_awaited_once()
        commit.assert_awaited_once()

    async def test_review_failure_is_logged_and_does_not_raise(self):
        with patch(
            "deep_research.memory.review_runner.review_successful_turn",
            new=AsyncMock(
                side_effect=RuntimeError("review failed")
            ),
        ):
            result = await review_after_success(
                "agent",
                "thread-1",
                answer="A successful answer.",
                memory_service="memory-service",
                extractor="extractor",
            )

        self.assertIsNone(result)

    async def test_cancellation_is_propagated(self):
        with patch(
            "deep_research.memory.review_runner.review_successful_turn",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await review_after_success(
                    "agent",
                    "thread-1",
                    answer="A successful answer.",
                    memory_service="memory-service",
                    extractor="extractor",
                )


if __name__ == "__main__":
    unittest.main()
