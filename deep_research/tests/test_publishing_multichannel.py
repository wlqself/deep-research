import hashlib
import tempfile
import unittest
from pathlib import Path

from deep_research.publishing.intent_resolver import (
    PublishingIntentResolutionStatus,
    PublishingIntentResolver,
)
from deep_research.publishing.intents import PublishingIntent
from deep_research.publishing.models import (
    ArticleStatus,
    ArtifactSnapshot,
    InvalidStateTransitionError,
    PublicationChannel,
    PublicationStatus,
)
from deep_research.publishing.repository import PublishingRepository
from deep_research.publishing.service import (
    PublicationExecutionError,
    PublicationResult,
    PublishingService,
)


class _ArtifactService:
    async def read(self, *, thread_id, artifact_id):
        content = "# Multi-channel article\n\nBody."
        encoded = content.encode("utf-8")
        return ArtifactSnapshot(
            source_thread_id=thread_id,
            source_artifact_id=artifact_id,
            workspace_path="/final/multi-channel.md",
            filename="multi-channel.md",
            size_bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
            markdown_content=content,
        )


class _Publisher:
    def __init__(self, channel, outcome):
        self.channel = channel
        self.outcome = outcome
        self.calls = 0

    def publish(self, article):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class MultiChannelPublishingTests(unittest.IsolatedAsyncioTestCase):
    THREAD_ID = "thread-multi-channel"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = PublishingRepository(
            Path(self.temp_dir.name) / "publishing.sqlite"
        )
        self.repository.initialize()
        self.service = PublishingService(
            self.repository,
            _ArtifactService(),
        )

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    async def _publish_locally(self):
        article = await self.service.create_article_from_artifact(
            thread_id=self.THREAD_ID,
            artifact_id="artifact-multi-channel",
            title="Multi-channel article",
            slug="multi-channel-article",
        )
        self.service.approve_article(article.article_id)
        approval = self.service.request_publication_approval(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
        )
        self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
        )
        local = _Publisher(
            PublicationChannel.LOCAL_STATIC_SITE,
            PublicationResult(public_url="/articles/multi-channel/v1/index.md"),
        )
        publication = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.LOCAL_STATIC_SITE,
            idempotency_key="local-v1",
            publisher=local,
        )
        self.assertEqual(publication.status, PublicationStatus.PUBLISHED)
        return self.service.get_article(article.article_id), publication

    async def _approve_wechat(self, article_id):
        approval = self.service.request_publication_approval(
            article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
        )
        return self.service.approve_publication_request(
            approval.approval_id,
            decision_actor="user",
        )

    async def test_published_article_resolves_for_an_unpublished_channel(self):
        article, _ = await self._publish_locally()

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(
                action="request_publication",
                target_hint=article.title,
                channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            ),
            thread_id=self.THREAD_ID,
        )

        self.assertEqual(
            resolution.status,
            PublishingIntentResolutionStatus.RESOLVED,
        )
        self.assertEqual(resolution.target.article_id, article.article_id)

    async def test_second_channel_pending_keeps_article_published(self):
        article, local_publication = await self._publish_locally()
        await self._approve_wechat(article.article_id)
        wechat = _Publisher(
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            PublicationResult(
                status=PublicationStatus.PUBLISHING,
                external_id="wechat-draft-1",
            ),
        )

        publication = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            idempotency_key="wechat-v1",
            publisher=wechat,
        )

        self.assertEqual(publication.status, PublicationStatus.PUBLISHING)
        self.assertEqual(
            self.service.get_article(article.article_id).status,
            ArticleStatus.PUBLISHED,
        )
        self.assertEqual(
            self.repository.get_publication(local_publication.publication_id).status,
            PublicationStatus.PUBLISHED,
        )

    async def test_second_channel_failure_keeps_first_channel_published(self):
        article, local_publication = await self._publish_locally()
        await self._approve_wechat(article.article_id)
        wechat = _Publisher(
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            PublicationExecutionError("wechat_draft_rejected"),
        )

        publication = self.service.publish_article(
            article.article_id,
            channel=PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
            idempotency_key="wechat-failure-v1",
            publisher=wechat,
        )

        self.assertEqual(publication.status, PublicationStatus.FAILED)
        self.assertEqual(publication.error_code, "wechat_draft_rejected")
        self.assertEqual(
            self.service.get_article(article.article_id).status,
            ArticleStatus.PUBLISHED,
        )
        self.assertEqual(
            self.repository.get_publication(local_publication.publication_id).status,
            PublicationStatus.PUBLISHED,
        )

    async def test_same_published_channel_cannot_request_another_approval(self):
        article, _ = await self._publish_locally()

        resolution = PublishingIntentResolver(self.service).resolve(
            PublishingIntent(
                action="request_publication",
                target_hint=article.title,
                channel=PublicationChannel.LOCAL_STATIC_SITE,
            ),
            thread_id=self.THREAD_ID,
        )
        self.assertEqual(
            resolution.status,
            PublishingIntentResolutionStatus.NOT_FOUND,
        )
        with self.assertRaises(InvalidStateTransitionError):
            self.service.request_publication_approval(
                article.article_id,
                channel=PublicationChannel.LOCAL_STATIC_SITE,
            )


if __name__ == "__main__":
    unittest.main()
