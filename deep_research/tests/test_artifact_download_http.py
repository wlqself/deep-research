# 本步验证：
# - /artifacts/... 路由已经注册；
# - 同线程可以下载 Markdown；
# - 不同线程返回 404；
# - 非法 artifact ID 返回 404。
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import deep_research.agent as agent_module
import deep_research.main as main_module
from deep_research.config import settings

THREAD_ID = (
    "11111111-1111-1111-1111-111111111111"
)

ARTIFACT_ID = (
    "abcdef1234567890abcdef1234567890"
)

CONTENT = "# Test report"
FILENAME = "Test-report-abcdef123456.md"
WORKSPACE_PATH = f"/final/{FILENAME}"

def artifact_values():
    content_bytes = CONTENT.encode("utf-8")

    return {
        "messages": [
            {
                "role": "user",
                "content": "test",
            }
        ],
        "artifacts": {
            ARTIFACT_ID: {
                "artifact_id": ARTIFACT_ID,
                "filename": FILENAME,
                "workspace_path": WORKSPACE_PATH,
                "created_at": (
                    "2026-08-20T12:00:00+00:00"
                ),
                "size_bytes": len(content_bytes),
                "sha256": hashlib.sha256(
                    content_bytes,
                ).hexdigest(),
            }
        },
        "files": {
            WORKSPACE_PATH: {
                "content": CONTENT,
                "encoding": "utf-8",
            }
        },
    }

class FakeSnapshot:
    def __init__(self, values):
        self.values = values


class FakeAgent:
    async def aget_state(self, config):
        thread_id = config[
            "configurable"
        ]["thread_id"]

        if thread_id == THREAD_ID:
            return FakeSnapshot(
                artifact_values()
            )

        return FakeSnapshot(
            {
                "messages": [],
                "artifacts": {},
                "files": {},
            }
        )


class FakeRagService:
    def close(self) -> None:
        pass


class ArtifactDownloadHttpTests(unittest.TestCase):
    def test_http_download_is_thread_scoped(self):
        original_path = settings.checkpoint_db_path
        original_agent = agent_module.agent

        with tempfile.TemporaryDirectory() as temp_dir:
            settings.checkpoint_db_path = str(
                Path(temp_dir) / "artifact-http.sqlite"
            )

            def fake_build_agent(
                checkpointer,
                rag_service,
                memory_service=None,
            ):
                return FakeAgent()

            try:
                with patch.object(
                    agent_module,
                    "build_agent",
                    side_effect=fake_build_agent,
                ), patch.object(
                    main_module,
                    "build_rag_service",
                    return_value=FakeRagService(),
                ):
                    with TestClient(
                        main_module.app
                    ) as client:
                        response = client.get(
                            f"/artifacts/{THREAD_ID}/"
                            f"{ARTIFACT_ID}"
                        )

                        other_thread = client.get(
                            "/artifacts/"
                            "22222222-2222-2222-2222-222222222222/"
                            f"{ARTIFACT_ID}"
                        )

                        invalid_artifact = client.get(
                            f"/artifacts/{THREAD_ID}/"
                            "not-a-valid-artifact-id"
                        )

                        wrong_artifact = client.get(
                            f"/artifacts/{THREAD_ID}/"
                            "bbcdef1234567890abcdef1234567890"
                        )

                        artifact_list = client.get(
                            f"/threads/{THREAD_ID}/artifacts"
                        )

                        other_artifact_list = client.get(
                            "/threads/"
                            "22222222-2222-2222-2222-222222222222/"
                            "artifacts"
                        )

                self.assertEqual(
                    response.status_code,
                    200,
                )

                self.assertEqual(
                    response.text,
                    "# Test report",
                )

                self.assertTrue(
                    response.headers[
                        "content-type"
                    ].startswith("text/markdown")
                )

                self.assertEqual(
                    response.headers[
                        "cache-control"
                    ],
                    "private, no-store",
                )

                self.assertEqual(
                    response.headers[
                        "x-content-type-options"
                    ],
                    "nosniff",
                )

                self.assertIn(
                    "Test-report-abcdef123456.md",
                    response.headers[
                        "content-disposition"
                    ],
                )

                self.assertEqual(
                    other_thread.status_code,
                    404,
                )

                self.assertEqual(
                    invalid_artifact.status_code,
                    404,
                )

                self.assertEqual(
                    wrong_artifact.status_code,
                    404,
                )

                self.assertEqual(
                    artifact_list.status_code,
                    200,
                )

                artifact_payload = artifact_list.json()

                self.assertEqual(
                    artifact_payload["thread_id"],
                    THREAD_ID,
                )
                self.assertEqual(
                    len(artifact_payload["artifacts"]),
                    1,
                )
                self.assertEqual(
                    artifact_payload["artifacts"][0][
                        "artifact_id"
                    ],
                    ARTIFACT_ID,
                )

                self.assertEqual(
                    other_artifact_list.status_code,
                    404,
                )

            finally:
                settings.checkpoint_db_path = (
                    original_path
                )
                agent_module.agent = original_agent


if __name__ == "__main__":
    unittest.main()
