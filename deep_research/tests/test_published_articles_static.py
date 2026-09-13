import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_research.main import mount_published_articles


class PublishedArticlesStaticTests(unittest.TestCase):
    def test_generated_markdown_is_served_from_its_public_url(self):
        with tempfile.TemporaryDirectory() as directory:
            site_dir = Path(directory) / "published-site"
            app = FastAPI()
            articles_dir = mount_published_articles(app, site_dir)
            target = articles_dir / "example" / "v1" / "index.md"
            target.parent.mkdir(parents=True)
            target.write_text("# Published article", encoding="utf-8")

            with TestClient(app) as client:
                response = client.get("/articles/example/v1/index.md")
                write_response = client.post(
                    "/articles/example/v1/index.md",
                    content="replacement",
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "# Published article")
            self.assertIn("text/markdown", response.headers["content-type"])
            self.assertEqual(write_response.status_code, 405)
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "# Published article",
            )

    def test_mount_creates_missing_articles_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            site_dir = Path(directory) / "missing-site"
            app = FastAPI()

            articles_dir = mount_published_articles(app, site_dir)

            self.assertEqual(
                articles_dir,
                (site_dir / "articles").resolve(),
            )
            self.assertTrue(articles_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
