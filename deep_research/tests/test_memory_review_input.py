import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from unittest.mock import patch

from deep_research.config import settings
from deep_research.memory.review_input import build_review_input


class MemoryReviewInputTests(unittest.TestCase):
    def test_summary_is_separate_and_only_unreviewed_messages_are_kept(self):
        messages = [
            HumanMessage(
                id="summary-1",
                content=(
                    "Here is a summary of the conversation to date:\n\n"
                    "The user prefers concise answers."
                ),
                additional_kwargs={"lc_source": "summarization"},
            ),
            HumanMessage(
                id="old-user",
                content="Old message already reviewed.",
            ),
            AIMessage(
                id="tool-call",
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "secret"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                id="tool-result",
                content="Tool result must not be reviewed directly.",
                tool_call_id="call-1",
            ),
            HumanMessage(
                id="new-user",
                content="Please keep future answers short.",
            ),
            AIMessage(
                id="new-main",
                content="Understood.",
            ),
        ]

        review_input, latest_id = build_review_input(
            {"messages": messages},
            last_reviewed_message_id="old-user",
            similar_old_memories=[],
        )

        self.assertEqual(
            review_input.current_summary,
            "The user prefers concise answers.",
        )
        self.assertEqual(
            [message.content for message in review_input.unreviewed_messages],
            [
                "Please keep future answers short.",
                "Understood.",
            ],
        )
        self.assertEqual(
            review_input.latest_user_message,
            "Please keep future answers short.",
        )
        self.assertEqual(review_input.latest_main_message, "Understood.")
        self.assertEqual(latest_id, "new-main")

    def test_artifact_input_contains_metadata_but_not_content(self):
        review_input, _ = build_review_input(
            {
                "messages": [],
                "artifacts": {
                    "artifact-1": {
                        "artifact_id": "artifact-1",
                        "filename": "report.md",
                        "workspace_path": "/final/report.md",
                        "created_at": "2026-08-29T00:00:00Z",
                        "size_bytes": 10,
                        "sha256": "a" * 64,
                        "content": "full report body",
                    }
                },
            },
            last_reviewed_message_id=None,
            similar_old_memories=[],
        )

        self.assertEqual(len(review_input.artifact_metadata), 1)
        self.assertEqual(
            review_input.artifact_metadata[0].artifact_id,
            "artifact-1",
        )
        self.assertNotIn(
            "content",
            review_input.artifact_metadata[0].model_dump(),
        )

    def test_missing_old_cursor_falls_back_to_messages_after_summary(self):
        messages = [
            HumanMessage(
                id="summary-1",
                content="Here is a summary of the conversation to date:\n\nSummary.",
                additional_kwargs={"lc_source": "summarization"},
            ),
            HumanMessage(id="retained-user", content="Retained user message."),
            AIMessage(id="retained-main", content="Retained answer."),
        ]

        review_input, latest_id = build_review_input(
            {"messages": messages},
            last_reviewed_message_id="removed-by-summary",
            similar_old_memories=[],
        )

        self.assertEqual(
            [message.content for message in review_input.unreviewed_messages],
            ["Retained user message.", "Retained answer."],
        )
        self.assertEqual(latest_id, "retained-main")

    def test_message_count_limit_keeps_oldest_messages_and_reports_backlog(self):
        messages = [
            HumanMessage(id="m1", content="first"),
            AIMessage(id="m2", content="second"),
            HumanMessage(id="m3", content="third"),
        ]

        with patch.object(settings, "memory_extraction_max_messages", 2), patch.object(
            settings, "memory_extraction_max_chars", 100
        ):
            review_input, latest_id = build_review_input(
                {"messages": messages},
                last_reviewed_message_id=None,
                similar_old_memories=[],
            )

        self.assertEqual(
            [message.content for message in review_input.unreviewed_messages],
            ["first", "second"],
        )
        self.assertEqual(latest_id, "m2")
        self.assertTrue(review_input.review_batch.has_more_unreviewed)
        self.assertIn(
            "message_count",
            review_input.review_batch.truncated_fields,
        )

    def test_character_limit_keeps_oldest_complete_messages(self):
        messages = [
            HumanMessage(id="m1", content="1234"),
            AIMessage(id="m2", content="5678"),
            HumanMessage(id="m3", content="90"),
        ]

        with patch.object(settings, "memory_extraction_max_messages", 30), patch.object(
            settings, "memory_extraction_max_chars", 8
        ):
            review_input, latest_id = build_review_input(
                {"messages": messages},
                last_reviewed_message_id=None,
                similar_old_memories=[],
            )

        self.assertEqual(
            [message.content for message in review_input.unreviewed_messages],
            ["1234", "5678"],
        )
        self.assertEqual(latest_id, "m2")
        self.assertEqual(review_input.review_batch.total_chars, 8)
        self.assertTrue(review_input.review_batch.has_more_unreviewed)
        self.assertIn(
            "message_chars",
            review_input.review_batch.truncated_fields,
        )

    def test_single_oversized_message_is_truncated_and_cursor_advances_explicitly(self):
        messages = [
            HumanMessage(id="m1", content="1234567890"),
            AIMessage(id="m2", content="later"),
        ]

        with patch.object(settings, "memory_extraction_max_chars", 5):
            review_input, latest_id = build_review_input(
                {"messages": messages},
                last_reviewed_message_id=None,
                similar_old_memories=[],
            )

        self.assertEqual(
            [message.content for message in review_input.unreviewed_messages],
            ["12345"],
        )
        self.assertEqual(latest_id, "m1")
        self.assertTrue(review_input.review_batch.has_more_unreviewed)
        self.assertIn(
            "message_content",
            review_input.review_batch.truncated_fields,
        )

    def test_cursor_only_advances_to_messages_in_current_batch(self):
        messages = [
            HumanMessage(id="m1", content="reviewed"),
            AIMessage(id="m2", content="first new"),
            HumanMessage(id="m3", content="second new"),
        ]

        with patch.object(settings, "memory_extraction_max_messages", 1):
            review_input, latest_id = build_review_input(
                {"messages": messages},
                last_reviewed_message_id="m1",
                similar_old_memories=[],
            )

        self.assertEqual(
            [message.content for message in review_input.unreviewed_messages],
            ["first new"],
        )
        self.assertEqual(latest_id, "m2")
        self.assertTrue(review_input.review_batch.has_more_unreviewed)

    def test_artifact_limit_is_separate_from_message_character_budget(self):
        artifacts = {
            f"artifact-{index}": {
                "filename": f"file-{index}.md",
                "workspace_path": f"/file-{index}.md",
                "created_at": "2026-08-30T00:00:00Z",
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
            for index in range(3)
        }

        with patch.object(settings, "memory_extraction_max_artifacts", 2):
            review_input, _ = build_review_input(
                {"messages": [], "artifacts": artifacts},
                last_reviewed_message_id=None,
                similar_old_memories=[],
            )

        self.assertEqual(len(review_input.artifact_metadata), 2)
        self.assertIn(
            "artifact_metadata",
            review_input.review_batch.truncated_fields,
        )
        self.assertEqual(review_input.review_batch.total_chars, 0)

    def test_summary_limit_is_reported_without_changing_message_cursor(self):
        messages = [
            HumanMessage(
                id="summary-1",
                content=(
                    "Here is a summary of the conversation to date:\n\n"
                    "1234567890"
                ),
                additional_kwargs={"lc_source": "summarization"},
            ),
            HumanMessage(id="m1", content="new message"),
        ]

        with patch.object(
            settings,
            "memory_extraction_max_summary_chars",
            5,
        ):
            review_input, latest_id = build_review_input(
                {"messages": messages},
                last_reviewed_message_id=None,
                similar_old_memories=[],
            )

        self.assertEqual(review_input.current_summary, "12345")
        self.assertEqual(latest_id, "m1")
        self.assertIn(
            "current_summary",
            review_input.review_batch.truncated_fields,
        )


if __name__ == "__main__":
    unittest.main()
