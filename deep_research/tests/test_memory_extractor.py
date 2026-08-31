import json
import unittest

from deep_research.memory.extractor import MemoryExtractor
from deep_research.memory.review_types import (
    MemoryReviewInput,
    ReviewArtifactMetadata,
    ReviewBatch,
    ReviewMessage,
)


def make_review_input() -> MemoryReviewInput:
    return MemoryReviewInput(
        current_summary="The user prefers focused explanations.",
        unreviewed_messages=[
            ReviewMessage(
                role="user",
                content="Please keep future answers concise.",
            ),
            ReviewMessage(
                role="assistant",
                content="I will apply that preference.",
            ),
        ],
        review_batch=ReviewBatch(
            included_messages=[
                ReviewMessage(
                    role="user",
                    content="Please keep future answers concise.",
                ),
                ReviewMessage(
                    role="assistant",
                    content="I will apply that preference.",
                ),
            ],
            last_included_message_id="message-2",
            total_chars=76,
        ),
        latest_user_message="Please keep future answers concise.",
        latest_main_message="I will apply that preference.",
        artifact_metadata=[
            ReviewArtifactMetadata(
                artifact_id="artifact-1",
                filename="report.md",
                workspace_path="/final/report.md",
                created_at="2026-08-29T00:00:00Z",
                size_bytes=10,
                sha256="a" * 64,
            )
        ],
        similar_old_memories=[],
    )


class FakeStructuredModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.messages: list[object] | None = None

    async def ainvoke(self, messages: list[object]) -> object:
        self.messages = messages
        return self.response


class FakeModel:
    def __init__(self, structured_model: FakeStructuredModel) -> None:
        self.structured_model = structured_model
        self.schema = None

    def with_structured_output(self, schema: object) -> FakeStructuredModel:
        self.schema = schema
        return self.structured_model


class MemoryExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_sends_summary_window_and_metadata_to_model(self):
        structured_model = FakeStructuredModel(
            {
                "candidates": [
                    {
                        "kind": "user",
                        "memory_key": "user.answer_style.conciseness",
                        "scope": "global",
                        "title": "Answer style",
                        "summary": "The user prefers concise answers.",
                        "content": "Prefer concise answers.",
                        "keywords": ["concise"],
                        "source_type": "automatic_extraction",
                        "source_thread_id": "thread-1",
                    }
                ]
            }
        )
        model = FakeModel(structured_model)
        extractor = MemoryExtractor(model)

        result = await extractor.extract(make_review_input())

        self.assertIs(model.schema, __import__(
            "deep_research.memory.review_types",
            fromlist=["MemoryExtractionResult"],
        ).MemoryExtractionResult)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.candidates[0].memory_key,
            "user.answer_style.conciseness",
        )
        self.assertIsNotNone(structured_model.messages)
        self.assertEqual(len(structured_model.messages), 2)
        self.assertIn("长期记忆筛选器", structured_model.messages[0].content)

        payload = json.loads(structured_model.messages[1].content)
        self.assertEqual(
            payload["current_summary"],
            "The user prefers focused explanations.",
        )
        self.assertEqual(
            payload["unreviewed_messages"][0]["role"],
            "user",
        )
        self.assertEqual(
            payload["artifact_metadata"][0]["artifact_id"],
            "artifact-1",
        )

    async def test_model_failure_is_propagated(self):
        class FailingStructuredModel:
            async def ainvoke(self, messages: list[object]) -> object:
                raise RuntimeError("extraction failed")

        extractor = MemoryExtractor(FakeModel(FailingStructuredModel()))

        with self.assertRaisesRegex(RuntimeError, "extraction failed"):
            await extractor.extract(make_review_input())


if __name__ == "__main__":
    unittest.main()
