import hashlib
import unittest

from deep_research.publishing.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactService,
)


THREAD_ID = "11111111-1111-1111-1111-111111111111"
OTHER_THREAD_ID = "22222222-2222-2222-2222-222222222222"
ARTIFACT_ID = "abcdef1234567890abcdef1234567890"
CONTENT = "# Test report\n\nVerified content."
SHA256 = hashlib.sha256(CONTENT.encode("utf-8")).hexdigest()


class FakeThreadValuesReader:
    def __init__(self, values_by_thread):
        self.values_by_thread = values_by_thread
        self.calls = []

    async def __call__(self, agent, thread_id):
        self.calls.append((agent, thread_id))
        return self.values_by_thread.get(thread_id, {})


def artifact_values(
    *,
    workspace_path=f"/final/report-{ARTIFACT_ID[:12]}.md",
    filename=f"report-{ARTIFACT_ID[:12]}.md",
    size_bytes=len(CONTENT.encode("utf-8")),
    sha256=SHA256,
    content=CONTENT,
):
    return {
        "artifacts": {
            ARTIFACT_ID: {
                "artifact_id": ARTIFACT_ID,
                "filename": filename,
                "workspace_path": workspace_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        },
        "files": {
            workspace_path: {
                "content": content,
                "encoding": "utf-8",
            }
        },
    }


class PublishingArtifactServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = object()

    def build_service(self, values):
        reader = FakeThreadValuesReader(values)
        return ArtifactService(
            self.agent,
            thread_values_reader=reader,
        ), reader

    async def test_reads_and_returns_verified_snapshot(self):
        service, reader = self.build_service(
            {THREAD_ID: artifact_values()}
        )

        snapshot = await service.read(
            thread_id=THREAD_ID,
            artifact_id=ARTIFACT_ID,
        )

        self.assertEqual(snapshot.source_thread_id, THREAD_ID)
        self.assertEqual(snapshot.source_artifact_id, ARTIFACT_ID)
        self.assertEqual(snapshot.markdown_content, CONTENT)
        self.assertEqual(snapshot.size_bytes, len(CONTENT.encode("utf-8")))
        self.assertEqual(snapshot.sha256, SHA256)
        self.assertEqual(reader.calls, [(self.agent, THREAD_ID)])

    async def test_cross_thread_access_is_rejected(self):
        service, _ = self.build_service(
            {THREAD_ID: artifact_values()}
        )

        with self.assertRaises(ArtifactNotFoundError):
            await service.read(
                thread_id=OTHER_THREAD_ID,
                artifact_id=ARTIFACT_ID,
            )

    async def test_size_mismatch_is_rejected(self):
        service, _ = self.build_service(
            {
                THREAD_ID: artifact_values(
                    size_bytes=len(CONTENT.encode("utf-8")) + 1,
                )
            }
        )

        with self.assertRaises(ArtifactIntegrityError):
            await service.read(
                thread_id=THREAD_ID,
                artifact_id=ARTIFACT_ID,
            )

    async def test_hash_mismatch_is_rejected(self):
        service, _ = self.build_service(
            {
                THREAD_ID: artifact_values(
                    sha256="0" * 64,
                )
            }
        )

        with self.assertRaises(ArtifactIntegrityError):
            await service.read(
                thread_id=THREAD_ID,
                artifact_id=ARTIFACT_ID,
            )

    async def test_workspace_path_traversal_is_rejected(self):
        service, _ = self.build_service(
            {
                THREAD_ID: artifact_values(
                    workspace_path="/final/../outside.md",
                    filename="../outside.md",
                )
            }
        )

        with self.assertRaises(ArtifactNotFoundError):
            await service.read(
                thread_id=THREAD_ID,
                artifact_id=ARTIFACT_ID,
            )

    async def test_backslash_path_is_rejected(self):
        service, _ = self.build_service(
            {
                THREAD_ID: artifact_values(
                    workspace_path="/final/..\\outside.md",
                    filename="..\\outside.md",
                )
            }
        )

        with self.assertRaises(ArtifactNotFoundError):
            await service.read(
                thread_id=THREAD_ID,
                artifact_id=ARTIFACT_ID,
            )

    async def test_invalid_artifact_id_is_rejected(self):
        service, _ = self.build_service(
            {THREAD_ID: artifact_values()}
        )

        with self.assertRaises(ArtifactNotFoundError):
            await service.read(
                thread_id=THREAD_ID,
                artifact_id="../artifact",
            )


if __name__ == "__main__":
    unittest.main()
