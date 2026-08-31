import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import ANY, patch

from deep_research.config import settings
from deep_research import agent as agent_module
from deep_research import main as main_module


class FakeRegistry:
    def initialize(self) -> None:
        pass


class FakeRagService:
    def close(self) -> None:
        pass


class FakeMemoryStore:
    pass


class MemoryLifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_store_is_available_during_lifespan_and_cleared_after(self):
        fake_rag_service = FakeRagService()
        fake_memory_store = FakeMemoryStore()
        fake_agent = object()

        @asynccontextmanager
        async def fake_memory_context():
            yield fake_memory_store

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoints.sqlite"

            with patch.object(settings, "checkpoint_db_path", str(checkpoint_path)), patch.object(
                main_module, "DocumentRegistry", return_value=FakeRegistry()
            ), patch.object(
                main_module, "build_rag_service", return_value=fake_rag_service
            ), patch.object(
                main_module, "sqlite_memory_store", fake_memory_context
            ), patch.object(
                agent_module, "build_agent", return_value=fake_agent
            ) as build_agent:
                async with main_module.lifespan(main_module.app):
                    self.assertIs(
                        main_module.app.state.memory_store,
                        fake_memory_store,
                    )
                    self.assertIs(
                        agent_module.agent,
                        fake_agent,
                    )

                self.assertIsNone(main_module.app.state.memory_store)
                build_agent.assert_called_once_with(
                    ANY,
                    fake_rag_service,
                    ANY,
                )


if __name__ == "__main__":
    unittest.main()
