import tempfile
import unittest
from pathlib import Path

from deep_research.publishing.models import Article, ArticleStatus, PublicationStatus
from deep_research.publishing.publishers.douyin.publisher import DouyinPublisher
from deep_research.publishing.service_support import PublicationExecutionError


class _FakeReceipt:
    def __init__(self, status, external_id="douyin-task-1", public_url=None):
        self.status = status
        self.external_id = external_id
        self.public_url = public_url


class _FakeClient:
    def __init__(self):
        self.publish_payload = None

    def publish_image_note(self, **kwargs):
        self.publish_payload = kwargs
        return _FakeReceipt(PublicationStatus.PUBLISHING)

    def get_publication_status(self, external_id):
        return _FakeReceipt(
            PublicationStatus.PUBLISHED,
            external_id=external_id,
            public_url="https://example.test/douyin/1",
        )


def _article():
    return Article(
        article_id="article-1",
        source_thread_id="thread-1",
        source_artifact_id="artifact-1",
        source_artifact_sha256="0" * 64,
        title="抖音图文测试标题",
        slug="douyin-test",
        markdown_content="# 小标题\n\n正文 [链接](https://example.test)",
        excerpt="摘要",
        tags=["AI", "效率"],
        status=ArticleStatus.PUBLISHING,
    )


class DouyinPublisherTests(unittest.TestCase):
    def test_formats_and_passes_selected_local_images(self):
        client = _FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "cover.jpg"
            image.write_bytes(b"image")
            publisher = DouyinPublisher(
                client,
                image_paths_resolver=lambda article, attachment_ids: [
                    str(image)
                ],
            )

            result = publisher.publish(
                _article(),
                attachment_ids=("attachment-1",),
            )

        self.assertEqual(result.status, PublicationStatus.PUBLISHING)
        self.assertEqual(client.publish_payload["title"], "抖音图文测试标题")
        self.assertEqual(client.publish_payload["images"], [str(image.resolve())])
        self.assertEqual(client.publish_payload["tags"], ["AI", "效率"])

    def test_requires_at_least_one_image(self):
        publisher = DouyinPublisher(
            _FakeClient(),
            image_paths_resolver=lambda article, attachment_ids: [],
        )
        with self.assertRaises(PublicationExecutionError) as context:
            publisher.publish(_article())
        self.assertEqual(context.exception.error_code, "douyin_image_required")

    def test_inline_attachment_images_are_added_to_gallery(self):
        client = _FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "inline.jpg"
            image.write_bytes(b"image")
            requested_ids = []
            article = _article()
            article.markdown_content = "正文\n\n![配图](attachment://inline-id)"
            publisher = DouyinPublisher(
                client,
                image_paths_resolver=lambda article, attachment_ids: (
                    requested_ids.extend(attachment_ids) or [str(image)]
                ),
            )
            publisher.publish(article)

        self.assertEqual(requested_ids, ["inline-id"])
        self.assertEqual(client.publish_payload["images"], [str(image.resolve())])

    def test_reads_external_status(self):
        publisher = DouyinPublisher(_FakeClient())
        result = publisher.get_publication_status("douyin-task-1")
        self.assertEqual(result.status, PublicationStatus.PUBLISHED)
        self.assertEqual(result.public_url, "https://example.test/douyin/1")


if __name__ == "__main__":
    unittest.main()
