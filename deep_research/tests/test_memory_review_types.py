import unittest

from pydantic import ValidationError

from deep_research.memory.review_types import (
    MemoryExtractionResult,
    MemoryReviewInput,
    ReviewArtifactMetadata,
    ReviewBatch,
    ReviewMessage,
)
from deep_research.memory.type import MemoryCandidate


def make_candidate(index: int) -> MemoryCandidate:
    return MemoryCandidate.model_validate(
        {
            "kind": "user",
            "memory_key": f"user.preference.{index}",
            "scope": "global",
            "title": f"Preference {index}",
            "summary": f"Summary {index}",
            "content": f"Content {index}",
            "keywords": [f"keyword-{index}"],
            "source_type": "automatic_extraction",
            "source_thread_id": "thread-1",
        }
    )


class MemoryReviewTypeTests(unittest.TestCase):
    def test_review_input_accepts_only_safe_message_and_artifact_shapes(self):
        review_input = MemoryReviewInput.model_validate(
            {
                "current_summary": "Current summary.",
                "unreviewed_messages": [
                    {
                        "role": "user",
                        "content": "Please remember my preference.",
                    },
                    {
                        "role": "assistant",
                        "content": "I will consider it.",
                    },
                ],
                "review_batch": {
                    "included_messages": [
                        {
                            "role": "user",
                            "content": "Please remember my preference.",
                        },
                        {
                            "role": "assistant",
                            "content": "I will consider it.",
                        },
                    ],
                    "last_included_message_id": "message-2",
                    "total_chars": 53,
                },
                "latest_user_message": "Please remember my preference.",
                "latest_main_message": "I will consider it.",
                "artifact_metadata": [
                    {
                        "artifact_id": "artifact-1",
                        "filename": "report.md",
                        "workspace_path": "/final/report.md",
                        "created_at": "2026-08-29T00:00:00Z",
                        "size_bytes": 100,
                        "sha256": "a" * 64,
                    }
                ],
                "similar_old_memories": [],
            }
        )

        self.assertEqual(
            review_input.unreviewed_messages[0].role,
            "user",
        )
        self.assertIsInstance(
            review_input.artifact_metadata[0],
            ReviewArtifactMetadata,
        )

    def test_tool_messages_and_artifact_content_are_rejected(self):
        with self.assertRaises(ValidationError):
            ReviewMessage.model_validate(
                {
                    "role": "tool",
                    "content": "Tool output",
                }
            )

        with self.assertRaises(ValidationError):
            ReviewArtifactMetadata.model_validate(
                {
                    "artifact_id": "artifact-1",
                    "filename": "report.md",
                    "workspace_path": "/final/report.md",
                    "created_at": "2026-08-29T00:00:00Z",
                    "size_bytes": 100,
                    "sha256": "a" * 64,
                    "content": "full artifact body",
                }
            )

    def test_extraction_result_allows_empty_but_caps_candidates_at_five(self):
        empty = MemoryExtractionResult.model_validate({})
        self.assertEqual(empty.candidates, [])

        valid = MemoryExtractionResult.model_validate(
            {"candidates": [make_candidate(index) for index in range(5)]}
        )
        self.assertEqual(len(valid.candidates), 5)

        with self.assertRaises(ValidationError):
            MemoryExtractionResult.model_validate(
                {"candidates": [make_candidate(index) for index in range(6)]}
            )

    def test_review_input_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            MemoryReviewInput.model_validate(
                {
                    "current_summary": "summary",
                    "unreviewed_messages": [],
                    "review_batch": ReviewBatch().model_dump(),
                    "artifact_metadata": [],
                    "similar_old_memories": [],
                    "full_conversation": "must reject",
                }
            )


if __name__ == "__main__":
    unittest.main()
