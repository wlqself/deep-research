import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import HTTPException

from deep_research.handlers import memory as memory_module
from deep_research.memory.type import MemoryEntry


def make_request(service) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(memory_service=service),
        ),
    )


def make_entry(
    entry_id: str,
    *,
    kind: str = "user",
    status: str = "active",
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry.model_validate(
        {
            "id": entry_id,
            "memory_key": f"{kind}.preference.{entry_id}",
            "scope": "global",
            "kind": kind,
            "title": f"Title {entry_id}",
            "summary": f"Summary {entry_id}",
            "content": f"Content {entry_id}",
            "keywords": ["memory", entry_id],
            "source_type": "explicit_user",
            "source_thread_id": "thread-1",
            "created_at": now,
            "updated_at": now,
            "status": status,
        }
    )


class FakeMemoryService:
    def __init__(self, entries: list[MemoryEntry]) -> None:
        self.entries = {
            entry.id: entry
            for entry in entries
        }
        self.apply_calls = []
        self.deleted_ids = []
        self.forgotten_ids = []
        self.clear_all_calls = 0

    async def list_active_memories(
        self,
        *,
        kind=None,
        keyword=None,
        page=1,
        page_size=20,
    ):
        entries = [
            entry
            for entry in self.entries.values()
            if entry.status == "active"
            and (kind is None or entry.kind == kind)
        ]

        if keyword:
            normalized = keyword.lower()
            entries = [
                entry
                for entry in entries
                if normalized in entry.title.lower()
                or normalized in entry.summary.lower()
                or normalized in entry.content.lower()
                or any(
                    normalized in item.lower()
                    for item in entry.keywords
                )
            ]

        start = (page - 1) * page_size
        return entries[start:start + page_size]

    async def get_entry(self, kind, entry_id):
        entry = self.entries.get(entry_id)
        return entry if entry is not None and entry.kind == kind else None

    async def apply_decision(self, candidate, decision, *, existing=None):
        self.apply_calls.append((candidate, decision, existing))

        if existing is None:
            return None

        updated = MemoryEntry(
            id="updated-id",
            created_at=existing.created_at,
            updated_at=datetime.now(timezone.utc),
            status="active",
            **candidate.model_dump(mode="python"),
        )
        self.entries[existing.id] = existing.model_copy(
            update={"status": "superseded"},
        )
        self.entries[updated.id] = updated
        return updated

    async def delete_entry(self, kind, entry_id):
        entry = self.entries.get(entry_id)
        if entry is not None and entry.kind == kind:
            self.deleted_ids.append(entry_id)
            del self.entries[entry_id]

    async def forget_entry(self, kind, entry_id, **kwargs):
        entry = self.entries.get(entry_id)
        if entry is not None and entry.kind == kind:
            self.forgotten_ids.append(entry_id)
            del self.entries[entry_id]

    async def clear_all_memories(self):
        self.clear_all_calls += 1
        deleted_count = sum(
            1
            for entry in self.entries.values()
            if entry.status in {"active", "superseded"}
        )
        self.entries.clear()
        return deleted_count


class MemoryHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_returns_only_public_fields(self):
        service = FakeMemoryService([make_entry("user-1")])

        response = await memory_module.list_memories(
            make_request(service),
            page=1,
        )

        self.assertEqual(response.memories[0].id, "user-1")
        self.assertNotIn(
            "source_thread_id",
            response.memories[0].model_dump(),
        )
        self.assertEqual(response.page_size, 20)

    async def test_update_uses_existing_identity_and_explicit_update(self):
        service = FakeMemoryService([make_entry("user-1")])

        response = await memory_module.update_memory(
            "user",
            "user-1",
            memory_module.MemoryUpdateRequest(
                title="Edited title",
                summary="Edited summary",
                content="Edited content",
                keywords=["edited"],
            ),
            make_request(service),
        )

        candidate, decision, existing = service.apply_calls[0]
        self.assertEqual(candidate.memory_key, existing.memory_key)
        self.assertEqual(candidate.scope, existing.scope)
        self.assertEqual(candidate.title, "Edited title")
        self.assertEqual(decision.action, "update")
        self.assertEqual(response.memory.id, "updated-id")

    async def test_delete_removes_one_active_memory(self):
        service = FakeMemoryService([make_entry("user-1")])

        response = await memory_module.delete_memory(
            "user",
            "user-1",
            make_request(service),
        )

        self.assertEqual(response.deleted_memory_id, "user-1")
        self.assertEqual(service.forgotten_ids, ["user-1"])

    async def test_clear_all_uses_service_clear_method(self):
        service = FakeMemoryService(
            [
                make_entry("user-1"),
                make_entry("project-1", kind="project"),
                make_entry("old-1", status="superseded"),
            ]
        )

        response = await memory_module.clear_memories(
            make_request(service),
        )

        self.assertEqual(response.deleted_count, 3)
        self.assertEqual(service.clear_all_calls, 1)
        self.assertEqual(service.entries, {})

    async def test_kind_scoped_clear_is_rejected(self):
        service = FakeMemoryService([make_entry("user-1")])

        with self.assertRaises(HTTPException) as context:
            await memory_module.clear_memories(
                make_request(service),
                kind="user",
            )

        self.assertEqual(context.exception.status_code, 422)
        self.assertEqual(service.clear_all_calls, 0)

    async def test_missing_memory_service_returns_503(self):
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(memory_service=None),
            ),
        )

        with self.assertRaises(HTTPException) as context:
            await memory_module.list_memories(request)

        self.assertEqual(context.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
