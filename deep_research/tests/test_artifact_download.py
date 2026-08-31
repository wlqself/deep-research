import asyncio
import unittest
import hashlib
from uuid import UUID

from fastapi import HTTPException

import deep_research.handlers.artifacts as artifacts_module

THREAD_ID = (
    "11111111-1111-1111-1111-111111111111"
)

ARTIFACT_ID = (
    "abcdef1234567890abcdef1234567890"
)

CONTENT = "# Test report"
FILENAME = "Test-report-abcdef123456.md"
WORKSPACE_PATH = f"/final/{FILENAME}"

def artifact_values(
    content=CONTENT,
    filename=FILENAME,
):
    content_bytes = content.encode("utf-8")
    workspace_path = f"/final/{filename}"

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
                "filename": filename,
                "workspace_path": workspace_path,
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
            workspace_path: {
                "content": content,
                "encoding": "utf-8",
            }
        },
    }

class FakeSnapshot:
    def __init__(self, values):
        self.values = values


class FakeAgent:
    def __init__(self, values=None):
        self.values = values

    async def aget_state(self, config):
        thread_id = config[
            "configurable"
        ]["thread_id"]

        if thread_id == THREAD_ID:
            return FakeSnapshot(
                self.values or artifact_values()
            )

        return FakeSnapshot(
            {
                "messages": [],
                "artifacts": {},
                "files": {},
            }
        )


class ArtifactDownloadTests(unittest.TestCase):
    def test_download_returns_markdown_for_same_thread(self):
        original_agent = (
            artifacts_module.agent_module.agent
        )

        artifacts_module.agent_module.agent = (
            FakeAgent()
        )

        try:
            response = asyncio.run(
                artifacts_module.download_artifact(
                    UUID(THREAD_ID),
                    ARTIFACT_ID,
                )
            )

            self.assertEqual(
                response.status_code,
                200,
            )

            self.assertEqual(
                response.body,
                b"# Test report",
            )

            self.assertIn(
                "Test-report-abcdef123456.md",
                response.headers[
                    "content-disposition"
                ],
            )

            self.assertEqual(
                response.media_type,
                "text/markdown",
            )

        finally:
            artifacts_module.agent_module.agent = (
                original_agent
            )

    def test_other_thread_cannot_download_artifact(self):
        original_agent = (
            artifacts_module.agent_module.agent
        )

        artifacts_module.agent_module.agent = (
            FakeAgent()
        )

        try:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(
                    artifacts_module.download_artifact(
                        UUID(
                            "22222222-2222-2222-2222-222222222222"
                        ),
                        ARTIFACT_ID,
                    )
                )

            self.assertEqual(
                error.exception.status_code,
                404,
            )

        finally:
            artifacts_module.agent_module.agent = (
                original_agent
            )

    def test_one_character_wrong_artifact_id_returns_404(self):
        original_agent = (
            artifacts_module.agent_module.agent
        )

        artifacts_module.agent_module.agent = (
            FakeAgent()
        )

        try:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(
                    artifacts_module.download_artifact(
                        UUID(THREAD_ID),
                        "bbcdef1234567890abcdef1234567890",
                    )
                )

            self.assertEqual(
                error.exception.status_code,
                404,
            )

        finally:
            artifacts_module.agent_module.agent = (
                original_agent
            )

    def test_artifact_list_is_thread_scoped(self):
        original_agent = (
            artifacts_module.agent_module.agent
        )

        artifacts_module.agent_module.agent = (
            FakeAgent()
        )

        try:
            response = asyncio.run(
                artifacts_module.list_artifacts(
                    UUID(THREAD_ID)
                )
            )

            self.assertEqual(
                response["thread_id"],
                THREAD_ID,
            )
            self.assertEqual(
                len(response["artifacts"]),
                1,
            )
            self.assertEqual(
                response["artifacts"][0]["artifact_id"],
                ARTIFACT_ID,
            )

            with self.assertRaises(HTTPException) as error:
                asyncio.run(
                    artifacts_module.list_artifacts(
                        UUID(
                            "22222222-2222-2222-2222-222222222222"
                        )
                    )
                )

            self.assertEqual(
                error.exception.status_code,
                404,
            )

        finally:
            artifacts_module.agent_module.agent = (
                original_agent
            )

    def test_download_rejects_size_mismatch(self):
        original_agent = (
            artifacts_module.agent_module.agent
        )

        values = artifact_values()
        values["artifacts"][ARTIFACT_ID][
            "size_bytes"
        ] += 1

        artifacts_module.agent_module.agent = (
            FakeAgent(values)
        )

        try:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(
                    artifacts_module.download_artifact(
                        UUID(THREAD_ID),
                        ARTIFACT_ID,
                    )
                )

            self.assertEqual(
                error.exception.status_code,
                409,
            )
        finally:
            artifacts_module.agent_module.agent = (
                original_agent
            )

    def test_download_rejects_sha256_mismatch(self):
        original_agent = (
            artifacts_module.agent_module.agent
        )

        values = artifact_values()
        values["artifacts"][ARTIFACT_ID][
            "sha256"
        ] = "0" * 64

        artifacts_module.agent_module.agent = (
            FakeAgent(values)
        )

        try:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(
                    artifacts_module.download_artifact(
                        UUID(THREAD_ID),
                        ARTIFACT_ID,
                    )
                )

            self.assertEqual(
                error.exception.status_code,
                409,
            )
        finally:
            artifacts_module.agent_module.agent = (
                original_agent
            )

    def test_download_preserves_utf8_content_and_filename(self):
        original_agent = (
            artifacts_module.agent_module.agent
        )

        values = artifact_values(
            content="# 中文报告\n\n你好，世界。",
            filename="中文报告.md",
        )

        artifacts_module.agent_module.agent = (
            FakeAgent(values)
        )

        try:
            response = asyncio.run(
                artifacts_module.download_artifact(
                    UUID(THREAD_ID),
                    ARTIFACT_ID,
                )
            )

            self.assertEqual(
                response.body.decode("utf-8"),
                "# 中文报告\n\n你好，世界。",
            )
            self.assertIn(
                "%E4%B8%AD%E6%96%87%E6%8A%A5%E5%91%8A.md",
                response.headers[
                    "content-disposition"
                ],
            )
        finally:
            artifacts_module.agent_module.agent = (
                original_agent
            )


if __name__ == "__main__":
    unittest.main()
