import importlib
import tempfile
import unittest
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pathlib import Path

from deep_research.config import settings
from unittest.mock import patch

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.embeddings import Embeddings

from deep_research.context import ResearchContext
from deep_research.state import ResearchState

from fastapi.testclient import TestClient


main_module = importlib.import_module(
    "deep_research.main"
)
agent_module = importlib.import_module(
    "deep_research.agent"
)


class FakeRagService:
    def close(self) -> None:
        pass


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * settings.embedding_dimensions for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * settings.embedding_dimensions


class FakeMemoryExtractor:
    def __init__(self, model) -> None:
        self.model = model


class _FailingNetworkClient:
    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("network access is forbidden in lifespan tests")


class _LifespanIsolationMixin:
    def _isolate_external_state(self) -> None:
        self._settings_originals = {
            "memory_db_path": settings.memory_db_path,
            "rag_registry_db_path": settings.rag_registry_db_path,
            "publishing_db_path": settings.publishing_db_path,
            "hitl_db_path": settings.hitl_db_path,
        }
        self._external_tempdir = tempfile.TemporaryDirectory()
        root = Path(self._external_tempdir.name)
        settings.memory_db_path = str(root / "memory.sqlite")
        settings.rag_registry_db_path = str(root / "registry.sqlite")
        settings.publishing_db_path = str(root / "publishing.sqlite")
        settings.hitl_db_path = str(root / "hitl.sqlite")
        self.addCleanup(self._restore_external_state)

    def _restore_external_state(self) -> None:
        settings.memory_db_path = self._settings_originals["memory_db_path"]
        settings.rag_registry_db_path = self._settings_originals[
            "rag_registry_db_path"
        ]
        settings.publishing_db_path = self._settings_originals[
            "publishing_db_path"
        ]
        settings.hitl_db_path = self._settings_originals["hitl_db_path"]
        self._external_tempdir.cleanup()

    def _patch_external_services(self) -> None:
        self.rag_service_patcher = patch.object(
            main_module,
            "build_rag_service",
            return_value=FakeRagService(),
        )
        self.embedding_patcher = patch(
            "deep_research.persistence.memory.create_embeddings",
            return_value=FakeEmbeddings(),
        )
        self.extractor_patcher = patch.object(
            main_module,
            "MemoryExtractor",
            FakeMemoryExtractor,
        )
        self.network_patcher = patch(
            "deep_research.tools.web_search.TavilySearch",
            _FailingNetworkClient,
        )
        self.http_patcher = patch(
            "deep_research.tools.read_page.httpx.AsyncClient",
            _FailingNetworkClient,
        )
        self.review_patcher = patch(
            "deep_research.handlers.research.review_after_success",
            autospec=True,
        )
        for patcher in (
            self.rag_service_patcher,
            self.embedding_patcher,
            self.extractor_patcher,
            self.network_patcher,
            self.http_patcher,
            self.review_patcher,
        ):
            patcher.start()
        self.addCleanup(self._stop_external_services)

    def _stop_external_services(self) -> None:
        for patcher in (
            getattr(self, "network_patcher", None),
            getattr(self, "http_patcher", None),
            getattr(self, "review_patcher", None),
            getattr(self, "extractor_patcher", None),
            getattr(self, "embedding_patcher", None),
            getattr(self, "rag_service_patcher", None),
        ):
            if patcher is not None:
                patcher.stop()


class AppLifespanTests(_LifespanIsolationMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._isolate_external_state()
        self._patch_external_services()

    async def test_lifespan_replaces_agent_and_closes_sqlite(
        self,
    ):
        original_path = settings.checkpoint_db_path
        original_agent = agent_module.agent

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = (
                Path(temp_dir) / "app-checkpoints.sqlite"
            )

            settings.checkpoint_db_path = str(
                database_path
            )

            try:
                with patch.object(agent_module, "build_agent", return_value=object()):
                    async with main_module.lifespan(main_module.app):
                        self.assertIsNot(agent_module.agent, original_agent)
                        self.assertTrue(database_path.exists())
                        self.assertIsNotNone(
                            main_module.app.state.hitl_repository
                        )
                        self.assertTrue(Path(settings.hitl_db_path).exists())
            finally:
                settings.checkpoint_db_path = (
                    original_path
                )

        self.assertIs(
            agent_module.agent,
            original_agent,
        )
        self.assertIsNone(main_module.app.state.hitl_repository)

    async def test_lifespan_restart_restores_messages_and_sources(
        self,
    ):
        original_path = settings.checkpoint_db_path
        original_agent = agent_module.agent

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = (
                Path(temp_dir) / "restart-checkpoints.sqlite"
            )

            settings.checkpoint_db_path = str(
                database_path
            )

            def fake_build_agent(
                checkpointer,
                rag_service,
                memory_service=None,
            ):
                return create_agent(
                    model=GenericFakeChatModel(
                        messages=iter(["第一轮回答"])
                    ),
                    tools=[],
                    context_schema=ResearchContext,
                    state_schema=ResearchState,
                    checkpointer=checkpointer,
                )

            config = {
                "configurable": {
                    "thread_id": "lifecycle-thread",
                }
            }

            try:
                with patch.object(
                    agent_module,
                    "build_agent",
                    side_effect=fake_build_agent,
                ):
                    async with main_module.lifespan(
                        main_module.app
                    ):
                        await agent_module.agent.ainvoke(
                            {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "第一轮问题",
                                    }
                                ]
                            },
                            config=config,
                            context=(
                                ResearchContext.from_settings()
                            ),
                        )

                        await agent_module.agent.aupdate_state(
                            config,
                            {
                                "sources": {
                                    "S1": {
                                        "source_id": "S1",
                                        "title": "Lifecycle source",
                                        "url": (
                                            "https://example.com/lifecycle"
                                        ),
                                        "snippet": (
                                            "lifecycle source"
                                        ),
                                    }
                                },
                                "next_source_number": 2,
                            },
                        )

                    async with main_module.lifespan(
                        main_module.app
                    ):
                        restored = (
                            await agent_module.agent.aget_state(
                                config
                            )
                        )

                contents = [
                    message.content
                    for message in restored.values["messages"]
                ]

                self.assertEqual(
                    contents,
                    [
                        "第一轮问题",
                        "第一轮回答",
                    ],
                )
                self.assertEqual(
                    restored.values["sources"]["S1"]["title"],
                    "Lifecycle source",
                )
                self.assertEqual(
                    restored.values["next_source_number"],
                    2,
                )

            finally:
                settings.checkpoint_db_path = (
                    original_path
                )
                agent_module.agent = original_agent

class HttpLifecycleTests(_LifespanIsolationMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._isolate_external_state()
        self._patch_external_services()

    def test_research_endpoint_uses_lifespan_agent(self):
        original_path = settings.checkpoint_db_path
        original_agent = agent_module.agent

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = (
                Path(temp_dir) / "http-checkpoints.sqlite"
            )

            settings.checkpoint_db_path = str(
                database_path
            )

            def fake_build_agent(
                checkpointer,
                rag_service,
                memory_service=None,
            ):
                return create_agent(
                    model=GenericFakeChatModel(
                        messages=iter(["HTTP 测试回答"])
                    ),
                    tools=[],
                    context_schema=ResearchContext,
                    state_schema=ResearchState,
                    checkpointer=checkpointer,
                )

            thread_id = (
                "11111111-1111-1111-1111-111111111111"
            )

            try:
                with patch.object(
                    agent_module,
                    "build_agent",
                    side_effect=fake_build_agent,
                ):
                    with TestClient(
                        main_module.app
                    ) as client:
                        self.assertIsNot(
                            agent_module.agent,
                            original_agent,
                        )

                        response = client.post(
                            "/research",
                            json={
                                "question": "测试生命周期 Agent",
                                "thread_id": thread_id,
                            },
                        )

                        self.assertEqual(
                            response.status_code,
                            200,
                        )

                        payload = response.json()

                        self.assertEqual(
                            payload["thread_id"],
                            thread_id,
                        )
                        self.assertEqual(
                            payload["answer"],
                            "HTTP 测试回答",
                        )

                self.assertIs(
                    agent_module.agent,
                    original_agent,
                )
                self.assertTrue(
                    database_path.exists()
                )

            finally:
                settings.checkpoint_db_path = (
                    original_path
                )
                agent_module.agent = original_agent

class HttpThreadContinuityTests(_LifespanIsolationMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._isolate_external_state()
        self._patch_external_services()

    def test_same_http_thread_continues_after_restart(self):
        original_path = settings.checkpoint_db_path
        original_agent = agent_module.agent

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = (
                Path(temp_dir) / "http-thread.sqlite"
            )

            settings.checkpoint_db_path = str(
                database_path
            )

            thread_id = (
                "22222222-2222-2222-2222-222222222222"
            )

            def fake_build_agent(
                checkpointer,
                rag_service,
                memory_service=None,
            ):
                return create_agent(
                    model=GenericFakeChatModel(
                        messages=iter(
                            [
                                "第一轮回答",
                                "第二轮回答",
                            ]
                        )
                    ),
                    tools=[],
                    context_schema=ResearchContext,
                    state_schema=ResearchState,
                    checkpointer=checkpointer,
                )

            async def read_restored_state():
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                    }
                }

                async with (
                    AsyncSqliteSaver.from_conn_string(
                        str(database_path)
                    ) as checkpointer
                ):
                    graph = create_agent(
                        model=GenericFakeChatModel(
                            messages=iter([])
                        ),
                        tools=[],
                        context_schema=ResearchContext,
                        state_schema=ResearchState,
                        checkpointer=checkpointer,
                    )

                    return await graph.aget_state(config)

            try:
                with patch.object(
                    agent_module,
                    "build_agent",
                    side_effect=fake_build_agent,
                ):
                    with TestClient(
                        main_module.app
                    ) as client:
                        first_response = client.post(
                            "/research",
                            json={
                                "question": "第一轮问题",
                                "thread_id": thread_id,
                            },
                        )

                        second_response = client.post(
                            "/research",
                            json={
                                "question": "第二轮问题",
                                "thread_id": thread_id,
                            },
                        )

                self.assertEqual(
                    first_response.status_code,
                    200,
                )
                self.assertEqual(
                    second_response.status_code,
                    200,
                )

                self.assertEqual(
                    first_response.json()["answer"],
                    "第一轮回答",
                )
                self.assertEqual(
                    second_response.json()["answer"],
                    "第二轮回答",
                )

                restored = asyncio.run(
                    read_restored_state()
                )

                contents = [
                    message.content
                    for message in restored.values["messages"]
                ]

                self.assertEqual(
                    contents,
                    [
                        "第一轮问题",
                        "第一轮回答",
                        "第二轮问题",
                        "第二轮回答",
                    ],
                )

            finally:
                settings.checkpoint_db_path = (
                    original_path
                )
                agent_module.agent = original_agent
if __name__ == "__main__":
    unittest.main()
