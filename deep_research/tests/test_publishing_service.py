import hashlib
import tempfile
import unittest
from pathlib import Path

from deep_research.publishing.artifacts import ArtifactService
from deep_research.publishing.models import (
    ArticleStatus,
    ArtifactSnapshot,
    InvalidStateTransitionError,
    PublicationChannel,
    PublicationStatus,
)
from deep_research.publishing.repository import PublishingRepository
from deep_research.publishing.service import (
    PublicationDeliveryUnknownError,
    PublicationExecutionError,
    PublicationResult,
    PublicationValidationError,
    PublishingService,
    PublishingServiceError,
)


THREAD_ID = "11111111-1111-1111-1111-111111111111"
CONTENT = "# Research report\n\nEvidence and conclusion."
CONTENT_SHA256 = hashlib.sha256(CONTENT.encode("utf-8")).hexdigest()


class FakeArtifactService:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = []

    async def read(self, *, thread_id, artifact_id):
        self.calls.append((thread_id, artifact_id))
        return self.snapshots[artifact_id]


class FakePublisher:
    channel = PublicationChannel.LOCAL_STATIC_SITE

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.calls = []

    def publish(self, article):
        self.calls.append(article.article_id)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return PublicationResult(
            public_url=f"/articles/{article.slug}/",
        )


class AttachmentPublisher(FakePublisher):
    def __init__(self):
        super().__init__()
        self.attachment_ids = None

    def publish(self, article, *, attachment_ids=()):
        self.calls.append(article.article_id)
        self.attachment_ids = tuple(attachment_ids)
        return PublicationResult(
            public_url=f"/articles/{article.slug}/",
        )


class StatusAwarePublisher(FakePublisher):
    def __init__(self, *, publish_result, status_result):
        super().__init__(outcomes=[publish_result])
        self.status_result = status_result
        self.status_calls = []

    def get_publication_status(self, external_id):
        self.status_calls.append(external_id)
        return self.status_result


class UnknownThenSuccessWeChatPublisher(FakePublisher):
    channel = PublicationChannel.WECHAT_OFFICIAL_ACCOUNT


class UnknownXiaohongshuPublisher(FakePublisher):
    channel = PublicationChannel.XIAOHONGSHU


class DouyinPublisher(FakePublisher):
    channel = PublicationChannel.DOUYIN


def build_snapshot(artifact_id="abcdef1234567890abcdef1234567890", filename="research-report.md"):
    return ArtifactSnapshot(
        source_thread_id=THREAD_ID,
        source_artifact_id=artifact_id,
        workspace_path=f"/final/{filename}",
        filename=filename,
        size_bytes=len(CONTENT.encode("utf-8")),
        sha256=CONTENT_SHA256,
        markdown_content=CONTENT,
    )


class PublishingServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = PublishingRepository(
            Path(self.temp_dir.name) / "publishing.sqlite"
        )
        self.repository.initialize()
        self.artifact_service = FakeArtifactService(
            {
                "abcdef1234567890abcdef1234567890": build_snapshot(),
                "bbcdef1234567890abcdef1234567890": build_snapshot(
                    artifact_id="bbcdef1234567890abcdef1234567890",
                    filename="second-report.md",
                ),
            }
        )
        self.service = PublishingService(
            self.repository,
            self.artifact_service,
        )

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    async def create_article(self, **kwargs):
        return await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id="abcdef1234567890abcdef1234567890",
            **kwargs,
        )

    def approve_for_publication(
        self,
        article,
        channel=PublicationChannel.LOCAL_STATIC_SITE,
    ):
        self.service.approve_article(article.article_id)
        request = self.service.request_publication_approval(
            article.article_id,
            channel=channel,
            actor="agent",
        )
        return self.service.approve_publication_request(
            request.approval_id,
            decision_actor="user",
            decision_reason="reviewed",
        )

    async def test_artifact_becomes_independent_article_draft(self):
        article = await self.create_article(
            title="A research article",
            slug="a-research-article",
            excerpt="A concise excerpt",
            tags=["research", "evidence"],
        )

        self.assertEqual(article.status, ArticleStatus.DRAFT)
        self.assertEqual(article.version, 1)
        self.assertEqual(article.markdown_content, CONTENT + "\n")
        self.assertEqual(article.source_thread_id, THREAD_ID)
        self.assertEqual(article.source_artifact_id, "abcdef1234567890abcdef1234567890")
        self.assertEqual(article.source_artifact_sha256, CONTENT_SHA256)
        self.assertEqual(self.artifact_service.calls, [(THREAD_ID, article.source_artifact_id)])

    async def test_article_markdown_is_normalized_without_mutating_artifact(self):
        raw_content = "\ufeff#Title###\r\n\r\n* item  \r\n\r\n```text\nvalue\n\n\nvalue\n```\r\n"
        artifact_id = "cccdef1234567890abcdef1234567890"
        snapshot = ArtifactSnapshot(
            source_thread_id=THREAD_ID,
            source_artifact_id=artifact_id,
            workspace_path="/final/formatting.md",
            filename="formatting.md",
            size_bytes=len(raw_content.encode("utf-8")),
            sha256=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
            markdown_content=raw_content,
        )
        self.artifact_service.snapshots[artifact_id] = snapshot

        article = await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id=artifact_id,
        )

        self.assertEqual(
            article.markdown_content,
            "# Title\n\n- item\n\n```text\nvalue\n\n\nvalue\n```\n",
        )
        self.assertEqual(snapshot.markdown_content, raw_content)

    async def test_invalid_markdown_is_blocked_before_publication_approval(self):
        article = await self.create_article()
        self.service.edit_article(
            article.article_id,
            markdown_content="# Title\n\n```python\nprint('unfinished')",
        )

        with self.assertRaises(PublicationValidationError) as context:
            self.service.request_publication_approval(
                article.article_id,
                channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            )

        self.assertEqual(
            context.exception.error_code,
            "article_markdown_unclosed_code_fence",
        )
        self.assertEqual(self.repository.list_approval_requests(), [])

    async def test_reimporting_same_artifact_returns_existing_article(self):
        first = await self.create_article()
        second = await self.create_article(
            title="A different title",
            slug="different-slug",
        )

        self.assertEqual(second.article_id, first.article_id)
        self.assertEqual(second.title, first.title)
        self.assertEqual(len(self.repository.list_articles()), 1)

    async def test_slug_collision_gets_unique_suffix(self):
        first = await self.create_article(slug="same-slug")
        second = await self.service.create_article_from_artifact(
            thread_id=THREAD_ID,
            artifact_id="bbcdef1234567890abcdef1234567890",
            slug="same-slug",
        )

        self.assertEqual(first.slug, "same-slug")
        self.assertEqual(second.slug, "same-slug-2")

    async def test_path_traversal_slug_is_rejected(self):
        with self.assertRaises(PublishingServiceError):
            await self.create_article(slug="../outside")

    async def test_only_drafts_can_be_edited(self):
        article = await self.create_article()
        updated = self.service.edit_article(
            article.article_id,
            title="Edited title",
            markdown_content="# Edited",
        )
        self.assertEqual(updated.title, "Edited title")
        self.assertEqual(updated.markdown_content, "# Edited\n")

        self.service.approve_article(article.article_id)
        with self.assertRaises(InvalidStateTransitionError):
            self.service.edit_article(
                article.article_id,
                title="Must not edit approved article",
            )

    async def test_article_must_be_approved_before_publishing(self):
        article = await self.create_article()
        publisher = FakePublisher()

        with self.assertRaises(InvalidStateTransitionError):
            self.service.publish_article(
                article.article_id,
                channel=PublicationChannel.LOCAL_STATIC_SITE,
                idempotency_key="request-1",
                publisher=publisher,
            )

        self.assertEqual(publisher.calls, [])
        self.assertEqual(
            self.repository.get_article(article.article_id).status,
            ArticleStatus.DRAFT,
        )

    async def test_publication_requires_an_approved_publication_request(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)
        publisher = FakePublisher()

        with self.assertRaises(InvalidStateTransitionError):
            self.service.publish_article(
                article.article_id,
                channel=PublicationChannel.LOCAL_STATIC_SITE,
                idempotency_key="request-without-approval",
                publisher=publisher,
            )

        self.assertEqual(publisher.calls, [])
        self.assertEqual(
            self.repository.get_article(article.article_id).status,
            ArticleStatus.APPROVED,
        )

    async def test_publication_approval_is_version_pinned_and_idempotent(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)

        first = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
        )
        second = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
        )

        self.assertEqual(first.approval_id, second.approval_id)
        self.assertEqual(first.article_version, article.version)
        self.assertEqual(
            first.content_sha256,
            hashlib.sha256(article.markdown_content.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(len(self.repository.list_approval_requests()), 1)

        approved = self.service.approve_publication_request(
            first.approval_id,
            decision_actor="user",
            decision_reason="looks good",
        )
        self.assertEqual(approved.status.value, "approved")
        self.assertEqual(approved.decision_actor, "user")

        with self.assertRaises(InvalidStateTransitionError):
            self.service.approve_publication_request(
                first.approval_id,
                decision_actor="user",
            )

    async def test_draft_channel_approval_only_approves_article_after_user_decision(self):
        article = await self.create_article()

        request = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )

        self.assertEqual(
            self.service.get_article(article.article_id).status,
            ArticleStatus.DRAFT,
        )
        approved = self.service.approve_publication_request(
            request.approval_id,
            decision_actor="user",
        )
        self.assertEqual(approved.status.value, "approved")
        self.assertEqual(
            self.service.get_article(article.article_id).status,
            ArticleStatus.APPROVED,
        )

    async def test_rejecting_draft_channel_approval_keeps_article_editable(self):
        article = await self.create_article()
        request = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )

        rejected = self.service.reject_publication_request(
            request.approval_id,
            decision_actor="user",
            decision_reason="revise first",
        )

        self.assertEqual(rejected.status.value, "rejected")
        self.assertEqual(
            self.service.get_article(article.article_id).status,
            ArticleStatus.DRAFT,
        )

    async def test_publish_is_idempotent(self):
        article = await self.create_article()
        self.approve_for_publication(article)
        publisher = FakePublisher()

        first = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-1",
            publisher=publisher,
        )
        second = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-1",
            publisher=publisher,
        )

        self.assertEqual(first.publication_id, second.publication_id)
        self.assertEqual(first.status, PublicationStatus.PUBLISHED)
        self.assertEqual(len(publisher.calls), 1)
        self.assertEqual(
            self.repository.get_article(article.article_id).status,
            ArticleStatus.PUBLISHED,
        )

    async def test_publish_uses_immutable_approval_attachment_snapshot(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            attachment_ids=("image-a", "image-b", "image-a"),
        )
        self.assertEqual(approval.attachment_ids, ("image-a", "image-b"))
        self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
        )

        publisher = AttachmentPublisher()
        publication = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="attachment-snapshot-1",
            publisher=publisher,
        )

        self.assertEqual(publication.attachment_ids, ("image-a", "image-b"))
        self.assertEqual(publisher.attachment_ids, ("image-a", "image-b"))

    async def test_publish_rejects_attachment_selection_different_from_approval(self):
        article = await self.create_article()
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            attachment_ids=("image-a",),
        )
        self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
        )

        with self.assertRaises(PublicationValidationError) as error:
            self.service.publish_article(
                article.article_id,
                channel=PublicationChannel.LOCAL_STATIC_SITE,
                idempotency_key="attachment-mismatch-1",
                publisher=AttachmentPublisher(),
                attachment_ids=("image-b",),
            )

        self.assertEqual(error.exception.error_code, "publication_attachment_mismatch")

    async def test_async_publication_stays_in_progress_after_acceptance(self):
        article = await self.create_article()
        self.approve_for_publication(article)
        publisher = FakePublisher(
            outcomes=[
                PublicationResult(
                    status=PublicationStatus.PUBLISHING,
                    external_id="wechat-publish-1",
                )
            ]
        )

        publication = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-async-1",
            publisher=publisher,
        )

        self.assertEqual(publication.status, PublicationStatus.PUBLISHING)
        self.assertEqual(publication.external_id, "wechat-publish-1")
        self.assertEqual(
            self.repository.get_article(article.article_id).status,
            ArticleStatus.PUBLISHING,
        )
        self.assertEqual(len(publisher.calls), 1)

    async def test_replaying_async_publication_queries_status_without_republishing(self):
        article = await self.create_article()
        self.approve_for_publication(article)
        publisher = StatusAwarePublisher(
            publish_result=PublicationResult(
                status=PublicationStatus.PUBLISHING,
                external_id="external-publish-1",
            ),
            status_result=PublicationResult(
                status=PublicationStatus.PUBLISHED,
                external_id="external-article-1",
                public_url="https://example.test/articles/one",
            ),
        )

        first = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-async-replay-1",
            publisher=publisher,
        )
        second = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-async-replay-1",
            publisher=publisher,
        )

        self.assertEqual(first.status, PublicationStatus.PUBLISHING)
        self.assertEqual(second.status, PublicationStatus.PUBLISHED)
        self.assertEqual(publisher.calls, [article.article_id])
        self.assertEqual(publisher.status_calls, ["external-publish-1"])
        self.assertEqual(second.public_url, "https://example.test/articles/one")

    async def test_failed_publication_can_be_retried_with_new_request(self):
        article = await self.create_article()
        self.approve_for_publication(article)
        publisher = FakePublisher(
            outcomes=[
                PublicationExecutionError("static_write_failed"),
                PublicationResult(public_url="/articles/retry/")
            ]
        )

        failed = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-1",
            publisher=publisher,
        )
        self.assertEqual(failed.status, PublicationStatus.FAILED)
        self.assertEqual(failed.error_code, "static_write_failed")
        self.assertEqual(
            self.repository.get_article(article.article_id).status,
            ArticleStatus.FAILED,
        )

        published = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-2",
            publisher=publisher,
        )
        self.assertEqual(published.status, PublicationStatus.PUBLISHED)
        self.assertEqual(len(self.service.list_publications(article.article_id)), 2)
        self.assertEqual(len(publisher.calls), 2)

    async def test_publication_center_can_refresh_an_inflight_publication(self):
        article = await self.create_article()
        self.approve_for_publication(article)
        publisher = StatusAwarePublisher(
            publish_result=PublicationResult(
                status=PublicationStatus.PUBLISHING,
                external_id="external-publish-refresh",
            ),
            status_result=PublicationResult(
                status=PublicationStatus.PUBLISHED,
                external_id="external-article-refresh",
                public_url="https://example.test/articles/refresh",
            ),
        )

        pending = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="request-refresh-1",
            publisher=publisher,
        )
        refreshed = self.service.refresh_publication(
            pending.publication_id,
            publisher=publisher,
        )

        self.assertEqual(refreshed.status, PublicationStatus.PUBLISHED)
        self.assertEqual(refreshed.attempt_count, 1)
        self.assertEqual(publisher.calls, [article.article_id])
        self.assertEqual(publisher.status_calls, ["external-publish-refresh"])

    async def test_delivery_unknown_can_only_be_retried_after_confirmation(self):
        article = await self.create_article()
        self.approve_for_publication(
            article,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )
        publisher = UnknownThenSuccessWeChatPublisher(
            outcomes=[
                PublicationDeliveryUnknownError(
                    "wechat_publish_request_failed",
                    external_id="draft-media-1",
                ),
                PublicationResult(
                    status=PublicationStatus.PUBLISHED,
                    external_id="wechat-article-1",
                    public_url="https://example.test/articles/wechat",
                ),
            ]
        )

        unknown = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            idempotency_key="request-unknown-1",
            publisher=publisher,
        )
        self.assertEqual(unknown.status, PublicationStatus.DELIVERY_UNKNOWN)

        with self.assertRaises(PublicationValidationError):
            self.service.retry_publication(
                unknown.publication_id,
                idempotency_key="request-unknown-retry",
                publisher=publisher,
            )

        published = self.service.retry_publication(
            unknown.publication_id,
            idempotency_key="request-unknown-retry",
            confirm_delivery_unknown=True,
            publisher=publisher,
        )
        self.assertEqual(published.status, PublicationStatus.PUBLISHED)
        self.assertEqual(published.attempt_count, 2)
        self.assertEqual(len(publisher.calls), 2)

    async def test_inflight_channel_does_not_block_another_channel_approval(self):
        article = await self.create_article(
            title="跨平台文章",
        )
        self.service.approve_article(article.article_id)
        xhs_approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.XIAOHONGSHU,
        )
        self.service.approve_publication_request(
            xhs_approval.approval_id,
            decision_actor="user",
        )

        xhs_publisher = UnknownXiaohongshuPublisher(
            outcomes=[
                PublicationDeliveryUnknownError(
                    "xiaohongshu_publish_receipt_missing",
                )
            ]
        )
        unknown = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.XIAOHONGSHU,
            idempotency_key="cross-channel-xhs",
            publisher=xhs_publisher,
        )
        self.assertEqual(unknown.status, PublicationStatus.DELIVERY_UNKNOWN)
        self.assertEqual(
            self.service.get_article(article.article_id).status,
            ArticleStatus.PUBLISHING,
        )

        douyin_approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.DOUYIN,
        )
        self.service.approve_publication_request(
            douyin_approval.approval_id,
            decision_actor="user",
        )
        published = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.DOUYIN,
            idempotency_key="cross-channel-douyin",
            publisher=DouyinPublisher(),
        )

        self.assertEqual(published.status, PublicationStatus.PUBLISHED)
        self.assertEqual(
            self.service.get_article(article.article_id).status,
            ArticleStatus.PUBLISHED,
        )
        self.assertEqual(
            self.service.list_publications(article.article_id)[0].channel,
            PublicationChannel.DOUYIN,
        )


if __name__ == "__main__":
    unittest.main()
