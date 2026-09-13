import hashlib
import tempfile
import unittest
from pathlib import Path

from deep_research.publishing.models import Article, ArticleStatus, WeChatCoverAsset
from deep_research.publishing.publishers.wechat import WeChatOfficialAccountPublisher
from deep_research.publishing.repository import PublishingRepository
from deep_research.publishing.wechat_covers import WeChatCoverService


class FakeCoverClient:
    def __init__(self) -> None:
        self.calls = 0

    def upload_cover(self, content: bytes, *, filename: str, content_type: str) -> str:
        self.calls += 1
        return f"remote-{self.calls}"

    def add_draft(self, payload):
        self.last_payload = payload
        return "draft-1"

    def submit_draft(self, media_id):
        return "publish-1"


class WeChatCoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = PublishingRepository(
            Path(self.temp_dir.name) / "publishing.sqlite"
        )
        self.repository.initialize()
        self.client = FakeCoverClient()
        self.service = WeChatCoverService(self.repository, self.client)

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_dir.cleanup()

    def test_same_cover_is_uploaded_once_and_can_be_rotated(self):
        first = self.service.upload_cover(b"cover-a", filename="a.jpg")
        repeated = self.service.upload_cover(b"cover-a", filename="a.jpg")
        second = self.service.upload_cover(b"cover-b", filename="b.jpg")

        self.assertEqual(self.client.calls, 2)
        self.assertEqual(first.remote_media_id, repeated.remote_media_id)
        first_current = self.repository.get_wechat_cover_asset_by_hash(
            first.content_sha256
        )
        self.assertIsNotNone(first_current)
        self.assertFalse(first_current.is_active)
        self.assertTrue(second.is_active)
        self.assertEqual(
            self.service.active_media_id(),
            second.remote_media_id,
        )

    def test_registering_existing_remote_id_does_not_store_content(self):
        digest = hashlib.sha256(b"cover").hexdigest()
        asset = self.service.register_uploaded_cover(
            content_sha256=digest,
            remote_media_id="remote-cover",
        )
        restored = self.repository.get_wechat_cover_asset_by_hash(digest)
        self.assertIsInstance(restored, WeChatCoverAsset)
        self.assertEqual(restored.remote_media_id, asset.remote_media_id)

    def test_publisher_reads_rotated_active_cover(self):
        first = self.service.upload_cover(b"cover-a", filename="a.jpg")
        self.service.upload_cover(b"cover-b", filename="b.jpg")
        publisher = WeChatOfficialAccountPublisher(
            self.client,
            thumb_media_id=None,
            cover_media_id_resolver=lambda: self.service.active_media_id(),
        )
        article = Article(
            article_id="article-1",
            source_thread_id="thread-1",
            source_artifact_id="artifact-1",
            source_artifact_sha256="a" * 64,
            title="Article",
            slug="article",
            markdown_content="# Article",
            excerpt="Excerpt",
            status=ArticleStatus.PUBLISHING,
        )
        publisher.publish(article)
        self.assertEqual(
            self.client.last_payload["articles"][0]["thumb_media_id"],
            "remote-2",
        )


if __name__ == "__main__":
    unittest.main()
