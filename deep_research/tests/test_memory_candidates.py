import unittest

from deep_research.memory.service import MemoryService
from deep_research.memory.type import MemoryCandidate


def make_candidate(**overrides: object) -> MemoryCandidate:
    data: dict[str, object] = {
        "kind": "project",
        "memory_key": "project.research.boundary",
        "scope": "project",
        "title": "Project constraint",
        "summary": "The project requires explicit boundaries.",
        "content": "Keep memory access in Main.",
        "keywords": ["memory", "boundary"],
        "source_type": "explicit_user",
        "source_thread_id": "thread-1",
    }
    data.update(overrides)
    return MemoryCandidate.model_validate(data)


class CapturingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def asearch(self, *args: object, **kwargs: object) -> list[object]:
        self.calls.append((args, kwargs))
        return []


class FailingStore:
    async def asearch(self, *args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("embedding search failed")


class MemoryCandidateTests(unittest.IsolatedAsyncioTestCase):
    async def test_ignore_candidate_does_not_search(self):
        store = CapturingStore()
        service = MemoryService(store, user_id="local-user")

        results = await service.find_similar_candidates(
            make_candidate(kind="ignore"),
            limit=5,
        )

        self.assertEqual(results, [])
        self.assertEqual(store.calls, [])

    async def test_candidate_builds_query_and_reuses_fixed_namespace(self):
        store = CapturingStore()
        service = MemoryService(store, user_id="local-user")
        candidate = make_candidate()

        results = await service.find_similar_candidates(
            candidate,
            limit=4,
        )

        self.assertEqual(results, [])
        self.assertEqual(len(store.calls), 1)

        args, kwargs = store.calls[0]
        self.assertEqual(
            args,
            (("memories", "local-user", "project"),),
        )
        self.assertEqual(
            kwargs,
            {
                "query": (
                    "Project constraint "
                    "The project requires explicit boundaries. "
                    "memory boundary"
                ),
                "filter": {"status": "active"},
                "limit": 4,
            },
        )

    async def test_store_failure_is_not_converted_to_empty_results(self):
        service = MemoryService(FailingStore(), user_id="local-user")

        with self.assertRaisesRegex(RuntimeError, "embedding search failed"):
            await service.find_similar_candidates(
                make_candidate(),
                limit=3,
            )


if __name__ == "__main__":
    unittest.main()
