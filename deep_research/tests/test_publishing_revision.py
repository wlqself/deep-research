import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deep_research.publishing.models import (
    ArticleStatus,
    ArtifactSnapshot,
    InvalidStateTransitionError,
    Publication,
    PublicationChannel,
    PublicationStatus,
)
from deep_research.publishing.repository import PublishingRepository
from deep_research.publishing.service import PublishingService
from deep_research.tools import (
    build_read_article_for_revision_tool,
    build_revise_article_tool,
    build_researcher_tools,
    build_supervisor_tools,
)


THREAD_ID = "11111111-1111-1111-1111-111111111111"
OTHER_THREAD_ID = "22222222-2222-2222-2222-222222222222"
ARTIFACT_ID = "abcdef1234567890abcdef1234567890"
CONTENT = "# Original\n\nOriginal body."


class FakeArtifactService:
    async def read(self, *, thread_id: str, artifact_id: str) -> ArtifactSnapshot:
        if thread_id != THREAD_ID or artifact_id != ARTIFACT_ID:
            raise ValueError("artifact not found")
        return ArtifactSnapshot(
            source_thread_id=THREAD_ID,
            source_artifact_id=ARTIFACT_ID,
            workspace_path="/final/original.md",
            filename="original.md",
            size_bytes=len(CONTENT.encode("utf-8")),
            sha256=hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
            markdown_content=CONTENT,
        )


class PublishingRevisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = PublishingRepository(
            Path(self.temp_dir.name) / "publishing.sqlite"
        )
        self.repository.initialize()
        self.service = PublishingService(
            self.repository,
            FakeArtifactService(),
        )

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    async def create_article(self):
        return await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id=ARTIFACT_ID,
            title="Original",
            slug="original",
        )

    async def test_approved_article_revision_preserves_previous_version(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)

        revision = self.service.revise_article(
            article.article_id,
            markdown_content="# Revised\n\nUpdated body.",
        )

        self.assertEqual(revision.status, ArticleStatus.DRAFT)
        self.assertEqual(revision.version, 2)
        self.assertEqual(revision.markdown_content, "# Revised\n\nUpdated body.\n")
        self.assertEqual(
            self.repository.get_article(article.article_id, version=1).status,
            ArticleStatus.APPROVED,
        )
        self.assertEqual(
            self.repository.get_article(article.article_id).version,
            2,
        )

    async def test_revision_rejects_pending_publication_approval(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
        )

        revision = self.service.revise_article(
            article.article_id,
            markdown_content="# Revised",
            change_reason="第二段需要重写",
        )

        self.assertEqual(revision.status, ArticleStatus.DRAFT)
        rejected = self.repository.get_approval_request(approval.approval_id)
        self.assertEqual(rejected.status.value, "rejected")
        self.assertEqual(rejected.decision_reason, "第二段需要重写")

    async def test_draft_revision_reuses_existing_editor(self):
        article = await self.create_article()

        revised = self.service.revise_article(
            article.article_id,
            title="Edited title",
            markdown_content="# Edited",
        )

        self.assertEqual(revised.status, ArticleStatus.DRAFT)
        self.assertEqual(revised.version, 1)
        self.assertEqual(revised.title, "Edited title")
        self.assertEqual(revised.markdown_content, "# Edited\n")

    async def test_failed_article_can_be_revised_into_new_draft(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)
        article = self.repository.get_article(article.article_id)
        article.start_publishing()
        self.repository.update_article(article)
        article.mark_failed()
        self.repository.update_article(article)

        revised = self.service.revise_article(
            article.article_id,
            title="Recovered title",
            markdown_content="# Recovered\n\nUpdated body.",
            change_reason="Fix the failed publication input",
        )

        self.assertEqual(revised.status, ArticleStatus.DRAFT)
        self.assertEqual(revised.version, 2)
        self.assertEqual(revised.title, "Recovered title")
        self.assertEqual(
            self.repository.get_article(article.article_id, version=1).status,
            ArticleStatus.FAILED,
        )

    async def test_publishing_article_without_inflight_publication_can_be_revised(self):
        """An ambiguous/finished attempt must not permanently lock editing."""

        article = await self.create_article()
        self.service.approve_article(article.article_id)
        article = self.repository.get_article(article.article_id)
        article.start_publishing()
        self.repository.update_article(article)

        revised = self.service.revise_article(
            article.article_id,
            title="Recovered while delivery is unknown",
            markdown_content="# Recovered\n\nUpdated body.",
        )

        self.assertEqual(revised.status, ArticleStatus.DRAFT)
        self.assertEqual(revised.version, 2)
        self.assertEqual(
            self.repository.get_article(article.article_id, version=1).status,
            ArticleStatus.PUBLISHING,
        )

    async def test_publishing_article_with_inflight_publication_is_not_revisable(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)
        article = self.repository.get_article(article.article_id)
        article.start_publishing()
        self.repository.update_article(article)
        self.repository.create_publication(
            Publication(
                publication_id="publication-in-flight",
                article_id=article.article_id,
                article_version=article.version,
                channel=PublicationChannel.XIAOHONGSHU,
                status=PublicationStatus.PUBLISHING,
                idempotency_key="idempotency-in-flight",
                content_sha256=hashlib.sha256(
                    article.markdown_content.encode("utf-8")
                ).hexdigest(),
                attempt_count=1,
            )
        )

        with self.assertRaises(InvalidStateTransitionError):
            self.service.revise_article(
                article.article_id,
                markdown_content="# Must wait",
            )

    async def test_revision_tool_returns_specific_non_retryable_state_error(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)
        article = self.repository.get_article(article.article_id)
        article.start_publishing()
        self.repository.update_article(article)
        self.repository.create_publication(
            Publication(
                publication_id="publication-tool-in-flight",
                article_id=article.article_id,
                article_version=article.version,
                channel=PublicationChannel.XIAOHONGSHU,
                status=PublicationStatus.PUBLISHING,
                idempotency_key="idempotency-tool-in-flight",
                content_sha256=hashlib.sha256(
                    article.markdown_content.encode("utf-8")
                ).hexdigest(),
                attempt_count=1,
            )
        )
        tool = build_revise_article_tool(self.service)
        result = await tool.coroutine(
            article_id=article.article_id,
            markdown_content="# Must wait",
            runtime=SimpleNamespace(
                config={"configurable": {"thread_id": THREAD_ID}},
            ),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "article_revision_in_flight")
        self.assertFalse(result["retryable"])
        self.assertFalse(result["repairable"])

    async def test_read_tool_is_scoped_to_current_thread(self):
        article = await self.create_article()
        tool = build_read_article_for_revision_tool(self.service)

        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": OTHER_THREAD_ID}},
        )
        result = await tool.coroutine(
            article_id=article.article_id,
            runtime=runtime,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "article_not_found")

    async def test_revise_tool_saves_draft_and_never_approves(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)
        tool = build_revise_article_tool(self.service)

        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        result = await tool.coroutine(
            article_id=article.article_id,
            markdown_content="# LLM revision\n\nNew body.",
            runtime=runtime,
            edit_mode="open_ended",
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["approval_required"])
        self.assertFalse(result["published"])
        self.assertEqual(result["article"]["status"], "draft")
        self.assertEqual(
            self.repository.get_article(article.article_id).status,
            ArticleStatus.DRAFT,
        )

    async def test_revision_tools_are_main_only(self):
        supervisor_names = {
            tool.name
            for tool in build_supervisor_tools(
                publishing_service=self.service,
            )
        }
        researcher_names = {
            tool.name for tool in build_researcher_tools()
        }

        self.assertIn("read_article_for_revision", supervisor_names)
        self.assertIn("revise_article_for_publication", supervisor_names)
        self.assertNotIn("read_article_for_revision", researcher_names)
        self.assertNotIn("revise_article_for_publication", researcher_names)


if __name__ == "__main__":
    unittest.main()
