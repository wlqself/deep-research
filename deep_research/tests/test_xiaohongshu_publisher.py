import tempfile
import unittest
from pathlib import Path

from deep_research.publishing.models import Article, ArticleStatus, PublicationStatus
from deep_research.publishing.service_support import PublicationValidationError
from deep_research.publishing.publishers.xiaohongshu.publisher import (
    XiaohongshuPublisher,
)
from deep_research.publishing.publishers.xiaohongshu.formatter import (
    format_xiaohongshu_content,
)


class _FakeReceipt:
    def __init__(self, status, external_id="xhs-task-1", public_url=None):
        self.status = status
        self.external_id = external_id
        self.public_url = public_url


class _FakeClient:
    def __init__(self):
        self.publish_payload = None

    def publish_note(self, **kwargs):
        self.publish_payload = kwargs
        return _FakeReceipt(PublicationStatus.PUBLISHING)

    def get_publication_status(self, external_id):
        return _FakeReceipt(
            PublicationStatus.PUBLISHED,
            external_id=external_id,
            public_url="https://example.test/note/1",
        )


def _article():
    return Article(
        article_id="article-1",
        source_thread_id="thread-1",
        source_artifact_id="artifact-1",
        source_artifact_sha256="0" * 64,
        title="小红书测试标题",
        slug="xiaohongshu-test",
        markdown_content="# 小标题\n\n正文 [链接](https://example.test)",
        excerpt="摘要",
        tags=["AI", "效率"],
        status=ArticleStatus.PUBLISHING,
    )


class XiaohongshuPublisherTests(unittest.TestCase):
    def test_body_has_no_duplicate_hashtags_when_tags_are_submitted_separately(self):
        article = _article()
        article.markdown_content = (
            "正文内容\n\n#情侣日常 #干饭人 #披萨 #西瓜 #周末快乐"
            "#情侣日常 #干饭人"
        )

        formatted = format_xiaohongshu_content(article)

        self.assertNotIn("#情侣日常", formatted.content)
        self.assertNotIn("#干饭人", formatted.content)
        self.assertEqual(
            formatted.tags,
            ("AI", "效率", "情侣日常", "干饭人", "披萨", "西瓜", "周末快乐"),
        )

    def test_formats_and_passes_selected_local_images(self):
        client = _FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "cover.jpg"
            image.write_bytes(b"image")
            publisher = XiaohongshuPublisher(
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
        self.assertEqual(client.publish_payload["title"], "小红书测试标题")
        self.assertEqual(client.publish_payload["images"], [str(image.resolve())])
        self.assertEqual(client.publish_payload["tags"], ["AI", "效率"])

    def test_sanitizes_platform_unsafe_topic_punctuation(self):
        article = _article()
        article.tags = ["DeepSeek", "V4.1", "V4-1", "#AI"]
        article.markdown_content = "正文内容\n\n#DeepSeek #V4.1"

        formatted = format_xiaohongshu_content(article)

        self.assertEqual(formatted.tags, ("DeepSeek", "V41", "AI"))
        self.assertNotIn("#V4.1", formatted.content)
        self.assertNotIn(".1", formatted.content)

    def test_inline_attachment_images_are_added_in_body_order(self):
        client = _FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jpg"
            second = Path(directory) / "second.jpg"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            requested_ids = []

            def resolve(article, attachment_ids):
                requested_ids.extend(attachment_ids)
                return [str({"first-id": first, "second-id": second}[value]) for value in attachment_ids]

            article = _article()
            article.markdown_content = (
                "正文\n\n![第一张](attachment://first-id)\n\n"
                "![第二张](attachment://second-id)"
            )
            publisher = XiaohongshuPublisher(
                client,
                image_paths_resolver=resolve,
            )
            publisher.publish(article)

        self.assertEqual(requested_ids, ["first-id", "second-id"])
        self.assertEqual(
            client.publish_payload["images"],
            [str(first.resolve()), str(second.resolve())],
        )

    def test_rejects_content_over_platform_limit(self):
        article = _article()
        article.markdown_content = "x" * 1001

        with self.assertRaises(PublicationValidationError) as context:
            format_xiaohongshu_content(article)
        self.assertEqual(context.exception.error_code, "xiaohongshu_content_too_long")

    def test_reads_external_status(self):
        publisher = XiaohongshuPublisher(_FakeClient())
        result = publisher.get_publication_status("xhs-task-1")
        self.assertEqual(result.status, PublicationStatus.PUBLISHED)
        self.assertEqual(result.public_url, "https://example.test/note/1")


if __name__ == "__main__":
    unittest.main()
