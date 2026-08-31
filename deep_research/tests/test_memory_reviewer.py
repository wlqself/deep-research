import unittest

from langchain_core.messages import AIMessage, HumanMessage

from deep_research.memory.review_types import MemoryExtractionResult
from deep_research.memory.reviewer import review_successful_turn
from deep_research.memory.type import MemoryEntry


class Snapshot:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values


class FakeAgent:
    def __init__(self, values: dict[str, object]) -> None:
        self.snapshot = Snapshot(values)
        self.update_calls: list[object] = []

    async def aget_state(self, config: object) -> Snapshot:
        return self.snapshot

    async def aupdate_state(self, *args: object) -> None:
        self.update_calls.append(args)


class FakeExtractor:
    def __init__(self) -> None:
        self.inputs: list[object] = []

    async def extract(self, review_input: object) -> MemoryExtractionResult:
        self.inputs.append(review_input)
        return MemoryExtractionResult(candidates=[])


class FakeMemoryService:
    def __init__(self, entries: list[MemoryEntry]) -> None:
        self.entries = entries
        self.queries: list[tuple[str, int]] = []

    async def find_review_memories(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[MemoryEntry]:
        self.queries.append((query, limit))
        return self.entries


class FailingExtractor:
    async def extract(self, review_input: object) -> MemoryExtractionResult:
        raise RuntimeError("extraction failed")


def base_values() -> dict[str, object]:
    return {
        "messages": [
            HumanMessage(id="user-1", content="Please answer concisely."),
            AIMessage(id="main-1", content="Understood."),
        ],
        "memory_review_turn_count": 1,
    }


def make_old_memory() -> MemoryEntry:
    return MemoryEntry.model_validate(
        {
            "id": "old-memory-1",
            "memory_key": "user.answer_style.conciseness",
            "scope": "global",
            "kind": "user",
            "title": "Answer style",
            "summary": "The user prefers concise answers.",
            "content": "Prefer concise answers.",
            "keywords": ["concise"],
            "source_type": "explicit_user",
            "source_thread_id": "thread-old",
            "created_at": "2026-08-28T00:00:00Z",
            "updated_at": "2026-08-28T00:00:00Z",
            "status": "active",
        }
    )


class MemoryReviewerTests(unittest.IsolatedAsyncioTestCase):
    async def test_untriggered_success_returns_cursor_without_extracting(self):
        agent = FakeAgent(base_values())
        extractor = FakeExtractor()

        outcome = await review_successful_turn(
            agent,
            extractor,
            "thread-1",
            answer="Understood.",
        )

        self.assertFalse(outcome.trigger.should_review)
        self.assertIsNone(outcome.extraction)
        self.assertEqual(outcome.latest_message_id, "main-1")
        self.assertEqual(outcome.next_turn_count, 2)
        self.assertEqual(extractor.inputs, [])
        self.assertEqual(agent.update_calls, [])

    async def test_changed_summary_triggers_extraction_without_state_update(self):
        values = base_values()
        values.update(
            {
                "messages": [
                    HumanMessage(
                        id="summary-1",
                        content=(
                            "Here is a summary of the conversation to date:\n\n"
                            "The user prefers concise answers."
                        ),
                        additional_kwargs={"lc_source": "summarization"},
                    ),
                    HumanMessage(
                        id="user-2",
                        content="Keep future answers short.",
                    ),
                    AIMessage(id="main-2", content="Understood."),
                ],
                "last_archived_summary_hash": "old-hash",
            }
        )
        agent = FakeAgent(values)
        extractor = FakeExtractor()

        outcome = await review_successful_turn(
            agent,
            extractor,
            "thread-1",
            answer="Understood.",
        )

        self.assertTrue(outcome.trigger.should_review)
        self.assertIn("summary_changed", outcome.trigger.reasons)
        self.assertIsNotNone(outcome.extraction)
        self.assertEqual(outcome.latest_message_id, "main-2")
        self.assertEqual(len(extractor.inputs), 1)
        self.assertEqual(agent.update_calls, [])

    async def test_tenth_successful_turn_triggers_extraction(self):
        agent = FakeAgent(
            {
                **base_values(),
                "memory_review_turn_count": 9,
            }
        )
        extractor = FakeExtractor()

        outcome = await review_successful_turn(
            agent,
            extractor,
            "thread-1",
            answer="Understood.",
        )

        self.assertEqual(outcome.next_turn_count, 10)
        self.assertTrue(outcome.trigger.should_review)
        self.assertIn("interval", outcome.trigger.reasons)
        self.assertEqual(len(extractor.inputs), 1)

    async def test_similar_old_memories_are_added_before_extraction(self):
        agent = FakeAgent(
            {
                **base_values(),
                "memory_review_turn_count": 9,
            }
        )
        extractor = FakeExtractor()
        memory_service = FakeMemoryService([make_old_memory()])

        outcome = await review_successful_turn(
            agent,
            extractor,
            "thread-1",
            answer="Understood.",
            memory_service=memory_service,
        )

        self.assertIsNotNone(outcome.extraction)
        self.assertEqual(len(memory_service.queries), 1)
        self.assertIn("Please answer concisely.", memory_service.queries[0][0])
        self.assertEqual(memory_service.queries[0][1], 3)
        self.assertEqual(
            extractor.inputs[0].similar_old_memories[0].id,
            "old-memory-1",
        )

    async def test_untriggered_turn_does_not_search_similar_memories(self):
        agent = FakeAgent(base_values())
        extractor = FakeExtractor()
        memory_service = FakeMemoryService([make_old_memory()])

        outcome = await review_successful_turn(
            agent,
            extractor,
            "thread-1",
            answer="Understood.",
            memory_service=memory_service,
        )

        self.assertIsNone(outcome.extraction)
        self.assertEqual(memory_service.queries, [])

    async def test_extraction_failure_propagates_without_state_update(self):
        agent = FakeAgent(
            {
                **base_values(),
                "memory_review_turn_count": 9,
            }
        )

        with self.assertRaisesRegex(RuntimeError, "extraction failed"):
            await review_successful_turn(
                agent,
                FailingExtractor(),
                "thread-1",
                answer="Understood.",
            )

        self.assertEqual(agent.update_calls, [])


if __name__ == "__main__":
    unittest.main()
