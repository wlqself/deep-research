import unittest
import tempfile
from pathlib import Path

from deep_research.publishing.models import (
    Article,
    ArticleStatus,
    PublicationStatus,
)
from deep_research.publishing.publishers.wechat import (
    WeChatOfficialAccountPublisher,
)
from deep_research.publishing.publishers.wechat_api import (
    WeChatPublishStatus,
    WeChatTokenError,
)
from deep_research.publishing.service import PublicationExecutionError


class FakeWeChatClient:
    def __init__(self):
        self.drafts = []
        self.submitted_media_ids = []
        self.cover_uploads = []
        self.inline_uploads = []
        self.publish_status = WeChatPublishStatus(
            publish_id="publish-id-1",
            publish_status=1,
        )

    def add_draft(self, payload):
        self.drafts.append(payload)
        return "media-id-1"

    def submit_draft(self, media_id):
        self.submitted_media_ids.append(media_id)
        return "publish-id-1"

    def get_publish_status(self, publish_id):
        return self.publish_status

    def upload_cover(self, content, *, filename, content_type="image/jpeg"):
        self.cover_uploads.append((content, filename, content_type))
        return "uploaded-thumb-id"

    def upload_inline_image(self, content, *, filename, content_type="image/jpeg"):
        self.inline_uploads.append((content, filename, content_type))
        return f"https://mmbiz.example/{filename}"


class TokenFailingWeChatClient(FakeWeChatClient):
    def add_draft(self, payload):
        raise WeChatTokenError(
            "wechat_token_rejected",
            "WeChat rejected the access token request",
            provider_code=40164,
        )

    def get_publish_status(self, publish_id):
        raise WeChatTokenError(
            "wechat_token_request_failed",
            "WeChat access token request failed",
        )


def build_article(*, status=ArticleStatus.PUBLISHING):
    return Article(
        article_id="article-1",
        source_thread_id="thread-1",
        source_artifact_id="abcdef1234567890abcdef1234567890",
        source_artifact_sha256="a" * 64,
        title="WeChat article",
        slug="wechat-article",
        markdown_content="# Heading\n\n<script>alert(1)</script>",
        excerpt="A short excerpt",
        tags=["research"],
        status=status,
    )


class WeChatPublisherTests(unittest.TestCase):
    def test_draft_mode_creates_draft_and_is_drafted(self):
        client = FakeWeChatClient()
        publisher = WeChatOfficialAccountPublisher(
            client,
            thumb_media_id="thumb-id-1",
            author="Author",
            publish_mode="draft",
        )

        result = publisher.publish(build_article())

        self.assertEqual(result.status, PublicationStatus.DRAFTED)
        self.assertEqual(result.external_id, "media-id-1")
        self.assertEqual(client.submitted_media_ids, [])
        content = client.drafts[0]["articles"][0]["content"]
        self.assertNotIn("<script>", content)

    def test_publish_mode_submits_created_draft(self):
        client = FakeWeChatClient()
        publisher = WeChatOfficialAccountPublisher(
            client,
            thumb_media_id="thumb-id-1",
            publish_mode="publish",
        )

        result = publisher.publish(build_article())

        self.assertEqual(result.status, PublicationStatus.PUBLISHING)
        self.assertEqual(result.external_id, "publish-id-1")
        self.assertEqual(client.submitted_media_ids, ["media-id-1"])

    def test_selected_attachments_become_cover_and_inline_images(self):
        client = FakeWeChatClient()
        with tempfile.TemporaryDirectory() as directory:
            cover = Path(directory) / "cover.jpg"
            body = Path(directory) / "body.png"
            cover.write_bytes(b"cover")
            body.write_bytes(b"body")
            publisher = WeChatOfficialAccountPublisher(
                client,
                thumb_media_id=None,
                attachment_resolver=lambda article, ids: [
                    (str(cover), "image/jpeg"),
                    (str(body), "image/png"),
                ],
                publish_mode="draft",
            )

            result = publisher.publish(
                build_article(),
                attachment_ids=("cover-id", "body-id"),
            )

        self.assertEqual(result.status, PublicationStatus.DRAFTED)
        self.assertEqual(client.cover_uploads[0][1], "cover.jpg")
        self.assertEqual(client.inline_uploads[0][1], "body.png")
        content = client.drafts[0]["articles"][0]["content"]
        self.assertIn("https://mmbiz.example/body.png", content)

    def test_inline_attachment_is_uploaded_at_its_markdown_position(self):
        client = FakeWeChatClient()
        with tempfile.TemporaryDirectory() as directory:
            cover = Path(directory) / "cover.jpg"
            body = Path(directory) / "body.png"
            cover.write_bytes(b"cover")
            body.write_bytes(b"body")
            paths = {
                "cover-id": (str(cover), "image/jpeg"),
                "body-id": (str(body), "image/png"),
            }
            article = build_article()
            article.markdown_content = (
                "Before\n\n![正文图片](attachment://body-id)\n\nAfter"
            )
            publisher = WeChatOfficialAccountPublisher(
                client,
                thumb_media_id=None,
                attachment_resolver=lambda _article, ids: [paths[value] for value in ids],
                publish_mode="draft",
            )
            publisher.publish(article, attachment_ids=("cover-id",))

        content = client.drafts[0]["articles"][0]["content"]
        self.assertEqual(len(client.inline_uploads), 1)
        self.assertIn("https://mmbiz.example/body.png", content)
        self.assertLess(content.index("Before"), content.index("body.png"))
        self.assertLess(content.index("body.png"), content.index("After"))

    def test_published_article_can_target_wechat_as_an_additional_channel(self):
        client = FakeWeChatClient()
        publisher = WeChatOfficialAccountPublisher(
            client,
            thumb_media_id="thumb-id-1",
            publish_mode="draft",
        )

        result = publisher.publish(
            build_article(status=ArticleStatus.PUBLISHED)
        )

        self.assertEqual(result.status, PublicationStatus.DRAFTED)
        self.assertEqual(result.external_id, "media-id-1")
        self.assertEqual(len(client.drafts), 1)

    def test_non_publishing_article_is_rejected(self):
        publisher = WeChatOfficialAccountPublisher(
            FakeWeChatClient(),
            thumb_media_id="thumb-id-1",
        )

        with self.assertRaises(PublicationExecutionError) as context:
            publisher.publish(build_article(status=ArticleStatus.APPROVED))

        self.assertEqual(context.exception.error_code, "article_not_publishing")

    def test_publish_status_maps_publishing_and_success(self):
        client = FakeWeChatClient()
        publisher = WeChatOfficialAccountPublisher(
            client,
            thumb_media_id="thumb-id-1",
            publish_mode="publish",
        )

        pending = publisher.get_publication_status("publish-id-1")
        self.assertEqual(pending.status, PublicationStatus.PUBLISHING)

        client.publish_status = WeChatPublishStatus(
            publish_id="publish-id-1",
            publish_status=0,
            article_id="article-id-1",
            article_url="https://mp.weixin.qq.com/s/article-1",
        )
        published = publisher.get_publication_status("publish-id-1")
        self.assertEqual(published.status, PublicationStatus.PUBLISHED)
        self.assertEqual(published.external_id, "article-id-1")
        self.assertEqual(
            published.public_url,
            "https://mp.weixin.qq.com/s/article-1",
        )

    def test_publish_status_failure_uses_safe_error_code(self):
        client = FakeWeChatClient()
        client.publish_status = WeChatPublishStatus(
            publish_id="publish-id-1",
            publish_status=4,
        )
        publisher = WeChatOfficialAccountPublisher(
            client,
            thumb_media_id="thumb-id-1",
            publish_mode="publish",
        )

        with self.assertRaises(PublicationExecutionError) as context:
            publisher.get_publication_status("publish-id-1")

        self.assertEqual(context.exception.error_code, "wechat_audit_rejected")

    def test_token_failure_preserves_specific_error_code(self):
        publisher = WeChatOfficialAccountPublisher(
            TokenFailingWeChatClient(),
            thumb_media_id="thumb-id-1",
            publish_mode="draft",
        )

        with self.assertRaises(PublicationExecutionError) as context:
            publisher.publish(build_article())

        self.assertEqual(context.exception.error_code, "wechat_token_rejected")

    def test_token_failure_during_status_check_preserves_specific_error_code(self):
        publisher = WeChatOfficialAccountPublisher(
            TokenFailingWeChatClient(),
            thumb_media_id="thumb-id-1",
            publish_mode="publish",
        )

        with self.assertRaises(PublicationExecutionError) as context:
            publisher.get_publication_status("publish-id-1")

        self.assertEqual(
            context.exception.error_code,
            "wechat_token_request_failed",
        )


if __name__ == "__main__":
    unittest.main()
