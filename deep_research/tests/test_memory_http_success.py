import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from deep_research.handlers import research as research_module


def fake_http_request() -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                memory_service=object(),
                memory_extractor=object(),
            )
        )
    )


class MemoryHttpSuccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonempty_success_answer_triggers_review(self):
        thread_id = uuid4()

        with patch.object(
            research_module,
            "run_research",
            new=AsyncMock(return_value="A successful answer."),
        ), patch.object(
            research_module,
            "_artifact_ids_for_thread",
            new=AsyncMock(side_effect=[set(), set()]),
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            response = await research_module.research(
                research_module.ResearchRequest(
                    question="A question",
                    thread_id=thread_id,
                ),
                fake_http_request(),
            )

        self.assertEqual(response.answer, "A successful answer.")
        review.assert_awaited_once()
        self.assertEqual(review.await_args.args[1], str(thread_id))
        self.assertEqual(
            review.await_args.kwargs["answer"],
            "A successful answer.",
        )
        self.assertFalse(review.await_args.kwargs["report_saved"])

    async def test_new_artifact_marks_report_as_saved(self):
        thread_id = uuid4()

        with patch.object(
            research_module,
            "run_research",
            new=AsyncMock(return_value="A report answer."),
        ), patch.object(
            research_module,
            "_artifact_ids_for_thread",
            new=AsyncMock(
                side_effect=[
                    set(),
                    {"artifact-1"},
                ]
            ),
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            await research_module.research(
                research_module.ResearchRequest(
                    question="Save a report",
                    thread_id=thread_id,
                ),
                fake_http_request(),
            )

        self.assertTrue(review.await_args.kwargs["report_saved"])

    async def test_existing_artifact_without_new_artifact_is_not_report_saved(self):
        thread_id = uuid4()

        with patch.object(
            research_module,
            "run_research",
            new=AsyncMock(return_value="An answer."),
        ), patch.object(
            research_module,
            "_artifact_ids_for_thread",
            new=AsyncMock(
                side_effect=[
                    {"artifact-1"},
                    {"artifact-1"},
                ]
            ),
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            await research_module.research(
                research_module.ResearchRequest(
                    question="Continue",
                    thread_id=thread_id,
                ),
                fake_http_request(),
            )

        self.assertFalse(review.await_args.kwargs["report_saved"])

    async def test_empty_success_answer_does_not_trigger_review(self):
        with patch.object(
            research_module,
            "run_research",
            new=AsyncMock(return_value="   "),
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            response = await research_module.research(
                research_module.ResearchRequest(question="A question"),
                fake_http_request(),
            )

        self.assertEqual(response.answer, "   ")
        review.assert_not_awaited()

    async def test_rule_signal_is_forwarded_to_review(self):
        with patch.object(
            research_module,
            "run_research",
            new=AsyncMock(return_value="A concise answer."),
        ), patch.object(
            research_module,
            "_artifact_ids_for_thread",
            new=AsyncMock(side_effect=[set(), set()]),
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            await research_module.research(
                research_module.ResearchRequest(
                    question="Please remember my preference.",
                ),
                fake_http_request(),
            )

        self.assertTrue(review.await_args.kwargs["rule_triggered"])

    async def test_research_failure_does_not_trigger_review(self):
        with patch.object(
            research_module,
            "run_research",
            new=AsyncMock(side_effect=RuntimeError("research failed")),
        ), patch.object(
            research_module,
            "review_after_success",
            new=AsyncMock(),
        ) as review:
            with self.assertRaisesRegex(RuntimeError, "research failed"):
                await research_module.research(
                    research_module.ResearchRequest(question="A question"),
                    fake_http_request(),
                )

        review.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
