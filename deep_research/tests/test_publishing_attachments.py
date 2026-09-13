import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_research.handlers.publishing import router
from deep_research.publishing.attachments import ImageAttachmentService
from deep_research.publishing.models import ImageAttachmentAnalysis
from deep_research.publishing.repository import PublishingRepository
from deep_research.publishing.wechat_covers import WeChatCoverService
from deep_research.tools.publishing import build_set_wechat_cover_tool


class FakeWeChatClient:
    def __init__(self) -> None:
        self.calls = 0

    def upload_cover(self, content, *, filename, content_type):
        self.calls += 1
        return f"media-{self.calls}"


class TrackingImageAnalysisService:
    def __init__(self) -> None:
        self.ensure_pending_calls = []
        self.schedule_calls = []

    def ensure_pending(self, attachment_id):
        self.ensure_pending_calls.append(attachment_id)

    def schedule(self, attachment_id):
        self.schedule_calls.append(attachment_id)


class PublishingAttachmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.repository = PublishingRepository(root / "publishing.sqlite")
        self.repository.initialize()
        self.attachments = ImageAttachmentService(
            self.repository,
            root / "uploads",
        )
        self.client = FakeWeChatClient()
        self.covers = WeChatCoverService(self.repository, self.client)

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_dir.cleanup()

    async def test_uploaded_image_can_be_promoted_to_active_cover(self):
        attachment = self.attachments.save(
            thread_id="thread-1",
            filename="cover.png",
            content_type="image/png",
            content=b"image-bytes",
        )
        tool = build_set_wechat_cover_tool(self.attachments, self.covers)
        runtime = SimpleNamespace(config={"configurable": {"thread_id": "thread-1"}})

        result = await tool.coroutine(
            attachment_id=attachment.attachment_id,
            runtime=runtime,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "active")
        self.assertEqual(self.covers.active_media_id(), "media-1")
        self.assertNotIn("storage_path", result)

    async def test_attachment_can_be_used_from_another_thread(self):
        attachment = self.attachments.save(
            thread_id="thread-1",
            filename="cover.jpg",
            content_type="image/jpeg",
            content=b"image-bytes",
        )
        tool = build_set_wechat_cover_tool(self.attachments, self.covers)
        runtime = SimpleNamespace(config={"configurable": {"thread_id": "thread-2"}})

        result = await tool.coroutine(
            attachment_id=attachment.attachment_id,
            runtime=runtime,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "active")
        self.assertEqual(self.client.calls, 1)

    def test_uploading_identical_content_is_idempotent(self):
        first = self.attachments.save(
            thread_id="thread-1",
            filename="first.png",
            content_type="image/png",
            content=b"same-image",
        )
        second = self.attachments.save(
            thread_id="thread-2",
            filename="renamed.png",
            content_type="image/png",
            content=b"same-image",
        )

        self.assertEqual(second.attachment_id, first.attachment_id)
        self.assertEqual(
            [item.attachment_id for item in self.repository.list_image_attachments()],
            [first.attachment_id],
        )
        self.assertFalse(
            any(
                path.name.startswith(second.attachment_id)
                for path in (Path(self.temp_dir.name) / "uploads" / "thread-2").glob("*")
            )
        )

    def _http_app(self) -> FastAPI:
        app = FastAPI()
        app.include_router(router)
        app.state.image_attachment_service = self.attachments
        app.state.wechat_cover_service = self.covers
        app.state.wechat_publisher = SimpleNamespace(publish_mode="draft")
        return app

    def test_http_upload_does_not_start_image_analysis_before_message_send(self):
        analysis = TrackingImageAnalysisService()
        app = self._http_app()
        app.state.image_analysis_service = analysis

        with TestClient(app) as client:
            response = client.post(
                "/publishing/attachments/images",
                data={"thread_id": "thread-1"},
                files={"file": ("photo.jpg", b"image-bytes", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["analysis_status"], "pending")
        self.assertEqual(analysis.ensure_pending_calls, [])
        self.assertEqual(analysis.schedule_calls, [])

    def test_idempotent_reupload_exposes_existing_analysis_without_starting_new_job(self):
        attachment = self.attachments.save(
            thread_id="thread-1",
            filename="photo.jpg",
            content_type="image/jpeg",
            content=b"image-bytes",
        )
        self.repository.save_image_attachment_analysis(
            ImageAttachmentAnalysis(
                analysis_id="analysis-1",
                attachment_id=attachment.attachment_id,
                status="completed",
                image_type="photo",
                summary="已有分析",
            )
        )
        analysis = TrackingImageAnalysisService()
        app = self._http_app()
        app.state.image_analysis_service = analysis

        with TestClient(app) as client:
            response = client.post(
                "/publishing/attachments/images",
                data={"thread_id": "thread-2"},
                files={"file": ("renamed.jpg", b"image-bytes", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["analysis_status"], "completed")
        self.assertEqual(response.json()["analysis_summary"], "已有分析")
        self.assertEqual(analysis.ensure_pending_calls, [])
        self.assertEqual(analysis.schedule_calls, [])

    def test_http_upload_rejects_type_and_size_with_safe_error_codes(self):
        app = self._http_app()
        with TestClient(app) as client:
            invalid = client.post(
                "/publishing/attachments/images",
                data={"thread_id": "thread-1"},
                files={"file": ("secret.txt", b"not-image", "text/plain")},
            )
            self.assertEqual(invalid.status_code, 400)
            self.assertEqual(
                invalid.json()["detail"]["error_code"],
                "image_attachment_type_unsupported",
            )

            webp = client.post(
                "/publishing/attachments/images",
                data={"thread_id": "thread-1"},
                files={"file": ("cover.webp", b"webp", "image/webp")},
            )
            self.assertEqual(webp.status_code, 400)
            self.assertEqual(
                webp.json()["detail"]["error_code"],
                "image_attachment_type_unsupported",
            )

            self.attachments.max_bytes = 3
            oversized = client.post(
                "/publishing/attachments/images",
                data={"thread_id": "thread-1"},
                files={"file": ("large.png", b"1234", "image/png")},
            )
            self.assertEqual(oversized.status_code, 400)
            self.assertEqual(
                oversized.json()["detail"]["error_code"],
                "image_attachment_too_large",
            )
            self.assertNotIn("storage_path", oversized.text)
            self.assertNotIn("secret", oversized.text)

    def test_http_cover_route_allows_shared_images_and_supports_delete(self):
        attachment = self.attachments.save(
            thread_id="thread-1",
            filename="cover.png",
            content_type="image/png",
            content=b"image-bytes",
        )
        app = self._http_app()
        with TestClient(app) as client:
            shared_thread = client.post(
                f"/publishing/attachments/{attachment.attachment_id}/wechat-cover",
                json={"thread_id": "thread-2"},
            )
            self.assertEqual(shared_thread.status_code, 200)

            response = client.post(
                f"/publishing/attachments/{attachment.attachment_id}/wechat-cover",
                json={"thread_id": "thread-1"},
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "active")
            self.assertEqual(body["attachment_id"], attachment.attachment_id)
            self.assertIn("cover_asset_id", body)
            for forbidden in ("storage_path", "access_token", "app_secret", "errmsg", "image-bytes"):
                self.assertNotIn(forbidden, response.text)

            listed = client.get(
                "/publishing/attachments/images",
                params={"thread_id": "thread-1"},
            )
            self.assertEqual(listed.status_code, 200)
            self.assertTrue(listed.json()[0]["is_active"])
            self.assertEqual(listed.json()[0]["cover_asset_id"], body["cover_asset_id"])

            deleted = client.delete(
                f"/publishing/attachments/{attachment.attachment_id}"
            )
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(deleted.json()["status"], "deleted")
            self.assertIsNone(self.covers.active_media_id())
            self.assertEqual(
                client.get(
                    "/publishing/attachments/images",
                    params={"thread_id": "thread-2"},
                ).json(),
                [],
            )


if __name__ == "__main__":
    unittest.main()
