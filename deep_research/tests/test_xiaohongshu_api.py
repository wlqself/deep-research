import unittest

import httpx

from deep_research.publishing.models import PublicationStatus
from deep_research.publishing.publishers.xiaohongshu.client import (
    XiaohongshuClient,
)


class XiaohongshuClientTests(unittest.TestCase):
    def test_matches_reference_service_login_and_publish_envelopes(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/api/v1/login/status":
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {"is_logged_in": True, "username": "demo"},
                    },
                )
            if request.url.path == "/api/v1/publish":
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {
                            "status": "published",
                            "post_id": "post-123",
                        },
                        "message": "发布成功",
                    },
                )
            raise AssertionError(f"unexpected request: {request.url}")

        client = XiaohongshuClient(
            "http://127.0.0.1:18060",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        login = client.get_login_status()
        receipt = client.publish_note(
            title="标题",
            content="正文",
            images=["C:/images/cover.jpg"],
            tags=["AI"],
        )

        self.assertTrue(login.logged_in)
        self.assertEqual(receipt.external_id, "post-123")
        self.assertEqual(receipt.status, PublicationStatus.PUBLISHED)
        self.assertEqual(requests[1].url.path, "/api/v1/publish")
        self.assertEqual(
            httpx.Request(
                "POST",
                "http://test",
                json={
                    "title": "标题",
                    "content": "正文",
                    "images": ["C:/images/cover.jpg"],
                    "tags": ["AI"],
                },
            ).content,
            requests[1].content,
        )

    def test_prefers_nested_note_id_and_builds_note_url(self):
        payload = {
            "success": True,
            "data": {
                "note_id": "note-456",
                "status": "发布成功",
                "id": "unrelated-operation-id",
            },
        }

        receipt = XiaohongshuClient._receipt_from_payload(payload)

        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.external_id, "note-456")
        self.assertEqual(receipt.status, PublicationStatus.PUBLISHED)
        self.assertEqual(
            receipt.public_url,
            "https://www.xiaohongshu.com/explore/note-456",
        )

    def test_supports_camel_case_note_id_in_nested_response(self):
        payload = {
            "success": True,
            "data": {"result": {"noteId": "note-789", "status": "published"}},
        }

        receipt = XiaohongshuClient._receipt_from_payload(payload)

        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.external_id, "note-789")

    def test_falls_back_to_feed_for_reference_service_status_checks(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/publish/post-123":
                return httpx.Response(404, json={"error": "not found", "code": "NOT_FOUND"})
            if request.url.path == "/api/v1/feeds/list":
                return httpx.Response(
                    200,
                    json={"success": True, "data": {"feeds": [{"id": "post-123"}]}},
                )
            raise AssertionError(f"unexpected request: {request.url}")

        client = XiaohongshuClient(
            "http://127.0.0.1:18060",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        receipt = client.get_publication_status("post-123")

        self.assertEqual(receipt.external_id, "post-123")
        self.assertEqual(receipt.status, PublicationStatus.PUBLISHED)


if __name__ == "__main__":
    unittest.main()
