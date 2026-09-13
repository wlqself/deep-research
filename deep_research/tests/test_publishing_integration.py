import hashlib
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_research.app_lifecycle import (
    _build_wechat_publisher,
    build_lifespan,
)
from deep_research.hitl.repository import (
    RepositoryClosedError as HITLRepositoryClosedError,
    HITLRepository,
)
from deep_research.hitl.models import HITLAction, HITLStatus
from deep_research.hitl.service import HITLService
from deep_research.hitl.decision_service import HITLDecisionService
from deep_research.handlers.publishing import router
from deep_research.publishing.artifacts import ArtifactIntegrityError
from deep_research.publishing.models import (
    Article,
    ArticleStatus,
    ArtifactSnapshot,
    PublicationChannel,
    PublicationStatus,
    PublishingDomainError,
)
from deep_research.publishing.publishers.local_static import (
    LocalStaticPublisher,
)
from deep_research.publishing.repository import (
    PublishingRepository,
    RepositoryClosedError,
)
from deep_research.publishing.service import (
    PublicationResult,
    PublishingService,
)
from deep_research.tools import (
    build_researcher_tools,
    build_supervisor_tools,
)
from deep_research.tools.publishing import build_prepare_article_tool


THREAD_ID = "11111111-1111-1111-1111-111111111111"
ARTIFACT_ID = "abcdef1234567890abcdef1234567890"
CONTENT = "# Research report\n\nEvidence and conclusion."


class FakeArtifactService:
    def __init__(self, snapshot: ArtifactSnapshot) -> None:
        self.snapshot = snapshot

    async def read(self, *, thread_id: str, artifact_id: str) -> ArtifactSnapshot:
        if thread_id != THREAD_ID or artifact_id != ARTIFACT_ID:
            raise ArtifactIntegrityError("wrong artifact scope")
        return self.snapshot


class FakeWeChatPublisher:
    channel = PublicationChannel.WECHAT_OFFICIAL_ACCOUNT

    def __init__(self) -> None:
        self.calls: list[str] = []

    def publish(self, article: Article) -> PublicationResult:
        self.calls.append(article.article_id)
        return PublicationResult(
            status=PublicationStatus.PUBLISHING,
            external_id="wechat-publish-1",
        )


def build_snapshot() -> ArtifactSnapshot:
    return ArtifactSnapshot(
        source_thread_id=THREAD_ID,
        source_artifact_id=ARTIFACT_ID,
        workspace_path="/final/research-report.md",
        filename="research-report.md",
        size_bytes=len(CONTENT.encode("utf-8")),
        sha256=hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
        markdown_content=CONTENT,
    )


class PublishingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.repository = PublishingRepository(root / "publishing.sqlite")
        self.repository.initialize()
        self.service = PublishingService(
            self.repository,
            FakeArtifactService(build_snapshot()),
        )

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_dir.cleanup()

    async def test_agent_tool_prepares_draft_and_never_publishes(self):
        tool = build_prepare_article_tool(self.service)

        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        result = await tool.coroutine(
            artifact_id=ARTIFACT_ID,
            runtime=runtime,
            title="Prepared article",
            markdown_content="# Prepared article\n\nBody",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "draft")
        self.assertTrue(result["approval_required"])
        self.assertEqual(len(self.repository.list_articles()), 1)
        self.assertEqual(
            self.repository.list_articles()[0].status,
            ArticleStatus.DRAFT,
        )

    async def test_agent_tool_returns_precise_non_retryable_slug_error(self):
        tool = build_prepare_article_tool(self.service)
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": THREAD_ID}},
        )

        result = await tool.coroutine(
            artifact_id=ARTIFACT_ID,
            runtime=runtime,
            title="LangChain 1.0 updates",
            slug="langchain-1.0-updates",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_slug")
        self.assertFalse(result["retryable"])
        self.assertIn("1-0", result["message"])
        self.assertEqual(self.repository.list_articles(), [])

    async def test_prepare_tool_description_documents_slug_contract(self):
        tool = build_prepare_article_tool(self.service)

        self.assertIn("120", tool.description)
        self.assertIn("langchain-1-0-updates", tool.description)
        self.assertIn("retryable=false", tool.description)

    async def test_identical_non_retryable_prepare_call_is_rejected(self):
        tool = build_prepare_article_tool(self.service)
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        arguments = {
            "artifact_id": ARTIFACT_ID,
            "runtime": runtime,
            "title": "LangChain 1.0 updates",
            "slug": "langchain-1.0-updates",
        }

        first = await tool.coroutine(**arguments)
        repeated = await tool.coroutine(**arguments)

        self.assertEqual(first["error_code"], "invalid_slug")
        self.assertNotIn("duplicate", first)
        self.assertEqual(repeated["error_code"], "invalid_slug")
        self.assertTrue(repeated["duplicate"])
        self.assertFalse(repeated["retryable"])
        self.assertEqual(self.repository.list_articles(), [])

    async def test_publishing_tool_is_only_in_supervisor_tools(self):
        supervisor_names = {
            tool.name
            for tool in build_supervisor_tools(
                publishing_service=self.service,
            )
        }
        researcher_names = {
            tool.name for tool in build_researcher_tools()
        }

        self.assertIn("prepare_article_for_publication", supervisor_names)
        self.assertIn("request_publication_approval", supervisor_names)
        self.assertNotIn("prepare_article_for_publication", researcher_names)
        self.assertNotIn("request_publication_approval", researcher_names)

    async def test_local_publisher_writes_versioned_markdown(self):
        article = await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id=ARTIFACT_ID,
            slug="versioned-article",
        )
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            actor="agent",
        )
        self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
            decision_reason="reviewed",
        )
        publisher = LocalStaticPublisher(
            Path(self.temp_dir.name) / "published-site"
        )

        publication = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="integration-1",
            publisher=publisher,
        )

        output = (
            Path(self.temp_dir.name)
            / "published-site"
            / "articles"
            / "versioned-article"
            / "v1"
            / "index.md"
        )
        self.assertEqual(publication.public_url, "/articles/versioned-article/v1/index.md")
        self.assertEqual(output.read_text(encoding="utf-8"), CONTENT + "\n")

    async def test_local_publisher_failure_preserves_previous_file(self):
        article = Article(
            article_id="article-1",
            source_thread_id=THREAD_ID,
            source_artifact_id=ARTIFACT_ID,
            source_artifact_sha256="a" * 64,
            title="Article",
            slug="safe-article",
            markdown_content="new content",
            excerpt="excerpt",
            tags=[],
            status=ArticleStatus.PUBLISHING,
            version=1,
        )
        site_dir = Path(self.temp_dir.name) / "published-site"
        output = site_dir / "articles" / "safe-article" / "v1" / "index.md"
        output.parent.mkdir(parents=True)
        output.write_text("old content", encoding="utf-8")

        publisher = LocalStaticPublisher(site_dir)
        with patch(
            "deep_research.publishing.publishers.local_static.os.replace",
            side_effect=OSError("simulated replace failure"),
        ):
            with self.assertRaises(OSError):
                publisher.publish(article)

        self.assertEqual(output.read_text(encoding="utf-8"), "old content")

    async def test_local_publisher_copies_inline_attachment_images(self):
        source = Path(self.temp_dir.name) / "source.jpg"
        source.write_bytes(b"image")
        article = Article(
            article_id="article-inline-image",
            source_thread_id=THREAD_ID,
            source_artifact_id=ARTIFACT_ID,
            source_artifact_sha256="a" * 64,
            title="Article with image",
            slug="article-with-image",
            markdown_content="Before\n\n![Photo](attachment://image-id)\n\nAfter",
            excerpt="excerpt",
            tags=[],
            status=ArticleStatus.PUBLISHING,
            version=1,
        )
        site_dir = Path(self.temp_dir.name) / "published-site"
        publisher = LocalStaticPublisher(
            site_dir,
            attachment_resolver=lambda _article, ids: [
                (str(source), "image/jpeg") for _ in ids
            ],
        )

        publisher.publish(article)

        output = site_dir / "articles" / "article-with-image" / "v1" / "index.md"
        copied = output.parent / "images" / "image-id.jpg"
        self.assertIn("![Photo](images/image-id.jpg)", output.read_text(encoding="utf-8"))
        self.assertEqual(copied.read_bytes(), b"image")

    async def test_http_api_enforces_approval_and_publish_idempotency(self):
        app = FastAPI()
        app.include_router(router)
        app.state.publishing_service = self.service
        app.state.local_static_publisher = LocalStaticPublisher(
            Path(self.temp_dir.name) / "published-site"
        )

        with TestClient(app) as client:
            created = client.post(
                "/publishing/articles/from-artifact",
                json={
                    "thread_id": THREAD_ID,
                    "artifact_id": ARTIFACT_ID,
                    "slug": "api-article",
                },
            )
            self.assertEqual(created.status_code, 201)
            article_id = created.json()["article_id"]
            self.assertEqual(created.json()["markdown_content"].strip(), CONTENT)

            unapproved = client.post(
                f"/publishing/articles/{article_id}/publish",
                json={
                    "idempotency_key": "api-1",
                },
            )
            self.assertIn(unapproved.status_code, (400, 409))
            self.assertEqual(
                set(unapproved.json()["detail"]),
                {"error_code"},
            )

            approved = client.post(
                f"/publishing/articles/{article_id}/approve"
            )
            self.assertEqual(approved.status_code, 200)

            approval = self.service.request_publication_approval(
                article_id,
                channel=PublicationChannel.LOCAL_STATIC_SITE,
                actor="agent",
            )
            self.service.approve_publication_request(
                approval.approval_id,
                decision_actor="user",
                decision_reason="reviewed",
            )

            first = client.post(
                f"/publishing/articles/{article_id}/publish",
                json={"idempotency_key": "api-1"},
            )
            second = client.post(
                f"/publishing/articles/{article_id}/publish",
                json={"idempotency_key": "api-1"},
            )
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(
                first.json()["publication_id"],
                second.json()["publication_id"],
            )
            self.assertNotIn("publishing.sqlite", first.text)
            self.assertNotIn(str(Path(self.temp_dir.name)), first.text)

    async def test_center_approval_synchronizes_conversation_hitl(self):
        app = FastAPI()
        app.include_router(router)
        app.state.publishing_service = self.service

        hitl_repository = HITLRepository(
            Path(self.temp_dir.name) / "hitl.sqlite",
        )
        hitl_repository.initialize()
        hitl_service = HITLService(hitl_repository)
        app.state.hitl_service = hitl_service

        article = await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id=ARTIFACT_ID,
            slug="center-sync-article",
        )
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.XIAOHONGSHU,
            actor="agent",
        )
        interaction = hitl_service.create_or_get_interaction(
            thread_id=THREAD_ID,
            run_id="center-sync-run",
            action=HITLAction.APPROVE,
            target_type="publication_approval",
            target_id=approval.approval_id,
            target_version=approval.article_version,
        )

        try:
            with TestClient(app) as client:
                response = client.post(
                    f"/publishing/approval-requests/{approval.approval_id}/approve",
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                hitl_service.get_interaction(interaction.interaction_id).status,
                HITLStatus.APPROVED,
            )
        finally:
            hitl_repository.close()

    async def test_center_approval_uses_same_hitl_resume_path(self):
        """A center click updates the approval and resumes the paused graph."""

        app = FastAPI()
        app.include_router(router)
        app.state.publishing_service = self.service

        hitl_repository = HITLRepository(
            Path(self.temp_dir.name) / "hitl-center-resume.sqlite",
        )
        hitl_repository.initialize()
        hitl_service = HITLService(hitl_repository)
        app.state.hitl_service = hitl_service
        app.state.hitl_decision_service = HITLDecisionService(
            hitl_service,
            self.service,
        )

        article = await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id=ARTIFACT_ID,
            slug="center-resume-article",
        )
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.XIAOHONGSHU,
            actor="agent",
        )
        interaction = hitl_service.create_or_get_interaction(
            thread_id=THREAD_ID,
            run_id="center-resume-run",
            action=HITLAction.APPROVE,
            target_type="publication_approval",
            target_id=approval.approval_id,
            target_version=approval.article_version,
        )

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(
                    config={
                        "configurable": {"checkpoint_id": "center-resume-cp"},
                    },
                    values={},
                    interrupts=(
                        SimpleNamespace(
                            id="center-resume-interrupt",
                            value={
                                "kind": "publication_approval",
                                "interaction_id": interaction.interaction_id,
                            },
                        ),
                    ),
                    tasks=(),
                )

            async def ainvoke(self, value, *, config, context):
                return {
                    "messages": [
                        SimpleNamespace(content="发布中心审批已同步，Graph 已继续。"),
                    ],
                }

        app.state.agent = FakeAgent()
        try:
            with TestClient(app) as client:
                response = client.post(
                    f"/publishing/approval-requests/{approval.approval_id}/approve",
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "approved")
            persisted = hitl_service.get_interaction(interaction.interaction_id)
            self.assertEqual(persisted.status, HITLStatus.RESUMED)
            self.assertEqual(persisted.interrupt_id, "center-resume-interrupt")
        finally:
            hitl_repository.close()

    async def test_http_api_routes_wechat_channel_to_wechat_publisher(self):
        app = FastAPI()
        app.include_router(router)
        app.state.publishing_service = self.service
        app.state.local_static_publisher = LocalStaticPublisher(
            Path(self.temp_dir.name) / "published-site"
        )
        wechat_publisher = FakeWeChatPublisher()
        app.state.wechat_publisher = wechat_publisher

        article = await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id=ARTIFACT_ID,
            slug="wechat-api-article",
        )
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            actor="agent",
        )
        self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
        )

        with TestClient(app) as client:
            response = client.post(
                f"/publishing/articles/{article.article_id}/publish",
                json={
                    "channel": "wechat_official_account",
                    "idempotency_key": "wechat-api-1",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "publishing")
        self.assertEqual(wechat_publisher.calls, [article.article_id])

    async def test_http_api_reports_unconfigured_wechat_publisher(self):
        app = FastAPI()
        app.include_router(router)
        app.state.publishing_service = self.service
        app.state.local_static_publisher = LocalStaticPublisher(
            Path(self.temp_dir.name) / "published-site"
        )

        article = await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id=ARTIFACT_ID,
            slug="wechat-unconfigured-article",
        )

        with TestClient(app) as client:
            response = client.post(
                f"/publishing/articles/{article.article_id}/publish",
                json={
                    "channel": "wechat_official_account",
                    "idempotency_key": "wechat-api-2",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"]["error_code"],
            "wechat_not_configured",
        )

    async def test_lifespan_initializes_and_closes_publishing_repository(self):
        class FakeAgent:
            pass

        class FakeAgentModule:
            agent = FakeAgent()

            def build_agent(
                self,
                checkpointer,
                rag_service,
                memory_service=None,
                publishing_service=None,
            ):
                self.received_service = publishing_service
                return FakeAgent()

        class FakeRegistry:
            def initialize(self) -> None:
                pass

        class FakeRag:
            def close(self) -> None:
                pass

        @asynccontextmanager
        async def fake_context():
            yield object()

        root = Path(self.temp_dir.name)
        current_settings = SimpleNamespace(
            rag_registry_db_path=str(root / "registry.sqlite"),
            publishing_db_path=str(root / "publishing.sqlite"),
            hitl_db_path=str(root / "hitl.sqlite"),
            published_site_dir=str(root / "published-site"),
            memory_user_id="test-user",
        )
        agent_module = FakeAgentModule()
        app = FastAPI()
        lifespan = build_lifespan(
            settings=current_settings,
            agent_module=agent_module,
            document_registry_factory=lambda path: FakeRegistry(),
            rag_service_factory=lambda settings: FakeRag(),
            checkpointer_factory=fake_context,
            memory_store_factory=fake_context,
            memory_service_factory=lambda store, **kwargs: object(),
            memory_extractor_factory=lambda model: object(),
            model=object(),
        )

        async with lifespan(app):
            repository = app.state.publishing_repository
            hitl_repository = app.state.hitl_repository
            self.assertIsNotNone(app.state.publishing_service)
            self.assertIsNotNone(app.state.image_attachment_service)
            self.assertIsNotNone(hitl_repository)
            self.assertIs(
                agent_module.received_service,
                app.state.publishing_service,
            )
            self.assertTrue(
                app.state.local_static_publisher.channel
                is PublicationChannel.LOCAL_STATIC_SITE
            )

        self.assertIsNone(app.state.publishing_service)
        self.assertIsNone(app.state.image_attachment_service)
        with self.assertRaises(RepositoryClosedError):
            repository.list_articles()
        with self.assertRaises(HITLRepositoryClosedError):
            hitl_repository.list_for_thread("thread-1")

    def test_wechat_publisher_is_constructed_without_network_access(self):
        settings = SimpleNamespace(
            wechat_app_id="test-app-id",
            wechat_app_secret="test-app-secret",
            wechat_thumb_media_id="test-thumb-media-id",
            wechat_author="test-author",
            wechat_publish_mode="draft",
        )

        publisher = _build_wechat_publisher(settings)

        self.assertIsNotNone(publisher)
        self.assertIs(
            publisher.channel,
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )


if __name__ == "__main__":
    unittest.main()
