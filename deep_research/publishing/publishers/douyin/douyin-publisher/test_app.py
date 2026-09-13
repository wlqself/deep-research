import sys
import unittest
from unittest.mock import AsyncMock
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402


class DouyinPublisherTests(unittest.TestCase):
    def test_health_contract(self):
        response = TestClient(app.app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "douyin-publisher")

    def test_hashtags_are_added_once(self):
        self.assertEqual(app._with_hashtags("正文 #AI", ["AI", "抖音"]), "正文 #AI\n\n#抖音")

    def test_validation_success_message_is_not_publication_ack(self):
        self.assertFalse(
            app.DouyinBrowser._publication_was_acknowledged("作品未见异常，请继续填写发布信息")
        )

    def test_post_submit_success_message_is_publication_ack(self):
        self.assertTrue(app.DouyinBrowser._publication_was_acknowledged("作品审核中"))

    def test_content_management_navigation_is_publication_ack(self):
        self.assertTrue(
            app.DouyinBrowser._publication_was_acknowledged(
                "", 
                initial_url="https://creator.douyin.com/creator-micro/content/upload",
                current_url="https://creator.douyin.com/creator-micro/content/manage",
            )
        )

    def test_health_exposes_browser_diagnostics(self):
        response = TestClient(app.app).get("/health")
        payload = response.json()
        self.assertIn(payload["browser_context"], {"stopped", "ready", "closed"})
        self.assertTrue(payload["profile_dir"])

    def test_publish_returns_task_id(self):
        original_dir, original_file = app.DATA_DIR, app.TASKS_FILE
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                app.DATA_DIR = tmp_path
                app.TASKS_FILE = tmp_path / "tasks.json"
                app.store._tasks = {}
                response = TestClient(app.app).post(
                    "/api/v1/publish",
                    json={
                        "mode": "image",
                        "title": "测试作品",
                        "content": "测试正文",
                        "images": ["C:/not-used-by-queue-test.jpg"],
                        "tags": ["测试"],
                    },
                )
                self.assertEqual(response.status_code, 202)
                payload = response.json()
                self.assertTrue(payload["success"])
                self.assertEqual(payload["status"], "queued")
                self.assertTrue(payload["task_id"])
        finally:
                app.DATA_DIR, app.TASKS_FILE = original_dir, original_file


class ClosedPage:
    def is_closed(self):
        return True


class ClosedContext:
    pages = []

    async def new_page(self):
        raise RuntimeError("BrowserContext.new_page: Target page, context or browser has been closed")

    async def close(self):
        return None


class LivePage:
    def is_closed(self):
        return False


class LiveContext:
    pages = [LivePage()]


class BrowserLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_closed_context_is_rebuilt_once(self):
        browser = app.DouyinBrowser()
        browser.ensure = AsyncMock(side_effect=[ClosedContext(), LiveContext()])
        browser.reset = AsyncMock()

        page = await browser.page()

        self.assertIsInstance(page, LivePage)
        self.assertEqual(browser.ensure.await_count, 2)
        browser.reset.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
