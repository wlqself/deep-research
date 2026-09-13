import unittest

import httpx

from deep_research.publishing.models import PublicationStatus
from deep_research.publishing.publishers.douyin.client import DouyinClient


class DouyinClientTests(unittest.TestCase):
    def test_publish_and_status_use_stable_service_contract(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/api/v1/publish":
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "task_id": "task-1",
                            "status": "publishing",
                        }
                    },
                )
            return httpx.Response(
                200,
                json={
                    "data": {
                        "item_id": "item-1",
                        "status": "published",
                        "item_url": "https://example.test/item-1",
                    }
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        api = DouyinClient("http://douyin.test", http_client=client)
        try:
            receipt = api.publish_image_note(
                title="标题",
                content="正文",
                images=["C:/images/a.jpg"],
                tags=["AI"],
            )
            status = api.get_publication_status("task-1/with space")
        finally:
            client.close()

        self.assertEqual(receipt.status, PublicationStatus.PUBLISHED)
        self.assertEqual(receipt.external_id, "task-1")
        self.assertEqual(status.status, PublicationStatus.PUBLISHED)
        self.assertEqual(status.external_id, "task-1/with space")
        self.assertEqual(status.public_url, "https://example.test/item-1")
        self.assertEqual(requests[0].url.path, "/api/v1/publish")
        self.assertEqual(requests[0].content.find(b'"mode":"image"') >= 0, True)
        self.assertEqual(
            requests[-1].url.raw_path.decode("ascii"),
            "/api/v1/publish/task-1%2Fwith%20space",
        )


if __name__ == "__main__":
    unittest.main()
