import hashlib
import tempfile
import unittest
from pathlib import Path

from deep_research.rag.storage import (
    build_document_path,
    finalize_document_file,
    resolve_document_path,
    validate_upload_type,
    write_upload_to_temp,
)


class FakeUpload:
    def __init__(
        self,
        content: bytes,
        chunk_size: int = 3,
    ) -> None:
        self.content = content
        self.chunk_size = chunk_size
        self.offset = 0

    async def read(self, size: int) -> bytes:
        end = min(
            self.offset + min(size, self.chunk_size),
            len(self.content),
        )
        chunk = self.content[self.offset:end]
        self.offset = end
        return chunk


class StorageTests(unittest.IsolatedAsyncioTestCase):
    def test_allowed_upload_types(self):
        cases = [
            ("guide.pdf", "application/pdf", ".pdf"),
            ("guide.md", "text/markdown", ".md"),
            ("guide.markdown", "text/markdown", ".markdown"),
            ("guide.txt", "text/plain", ".txt"),
        ]

        for filename, content_type, expected_extension in cases:
            with self.subTest(filename=filename):
                self.assertEqual(
                    validate_upload_type(
                        filename,
                        content_type,
                    ),
                    expected_extension,
                )

    def test_content_type_parameters_are_ignored(self):
        self.assertEqual(
            validate_upload_type(
                "guide.pdf",
                "application/pdf; charset=binary",
            ),
            ".pdf",
        )

    def test_mismatched_mime_type_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_upload_type(
                "guide.pdf",
                "text/plain",
            )

    def test_unsupported_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_upload_type(
                "guide.exe",
                "application/octet-stream",
            )

    def test_document_path_uses_server_generated_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            safe_filename, path = build_document_path(
                temp_dir,
                "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                ".pdf",
            )

            root = Path(temp_dir).resolve()

            self.assertEqual(
                safe_filename,
                "3f2504e0-4f89-41d3-9a0c-0305e82c3301.pdf",
            )
            self.assertTrue(path.is_relative_to(root))
            self.assertNotIn("..", path.parts)

    async def test_upload_is_written_and_hashed_in_chunks(self):
        content = b"abcdefghi"

        with tempfile.TemporaryDirectory() as temp_dir:
            path, size, digest = await write_upload_to_temp(
                FakeUpload(content),
                temp_dir,
                max_bytes=20,
            )

            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(size, len(content))
            self.assertEqual(
                digest,
                hashlib.sha256(content).hexdigest(),
            )
            self.assertTrue(path.name.startswith("upload-"))
            self.assertEqual(path.suffix, ".part")

    async def test_upload_over_limit_is_removed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                await write_upload_to_temp(
                    FakeUpload(b"123456"),
                    temp_dir,
                    max_bytes=5,
                )

            self.assertEqual(
                list(Path(temp_dir).iterdir()),
                [],
            )

    def test_finalize_moves_file_and_creates_target_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_path = Path(temp_dir) / "upload.part"
            document_path = (
                Path(temp_dir)
                / "documents"
                / "doc-1"
                / "doc-1.pdf"
            )
            content = b"safe document content"
            temporary_path.write_bytes(content)

            result = finalize_document_file(
                temporary_path,
                document_path,
            )

            self.assertEqual(result, document_path)
            self.assertFalse(temporary_path.exists())
            self.assertEqual(document_path.read_bytes(), content)

    def test_finalize_rejects_existing_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_path = Path(temp_dir) / "upload.part"
            document_path = Path(temp_dir) / "document.pdf"
            temporary_path.write_bytes(b"new")
            document_path.write_bytes(b"old")

            with self.assertRaises(FileExistsError):
                finalize_document_file(
                    temporary_path,
                    document_path,
                )

            self.assertEqual(document_path.read_bytes(), b"old")
            self.assertTrue(temporary_path.exists())

    def test_finalize_rejects_missing_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                finalize_document_file(
                    Path(temp_dir) / "missing.part",
                    Path(temp_dir) / "document.pdf",
                )

    def test_finalize_rejects_symlink_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.part"
            symlink_path = Path(temp_dir) / "upload.part"
            target_path = Path(temp_dir) / "document.pdf"
            source_path.write_bytes(b"source")

            try:
                symlink_path.symlink_to(source_path)
            except (OSError, NotImplementedError):
                self.skipTest(
                    "symlink creation is unavailable"
                )

            with self.assertRaises(ValueError):
                finalize_document_file(
                    symlink_path,
                    target_path,
                )

    def test_resolve_document_path_accepts_matching_safe_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document_id = (
                "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
            )

            path = resolve_document_path(
                temp_dir,
                document_id,
                f"{document_id}.pdf",
            )

            self.assertEqual(
                path,
                Path(temp_dir) / document_id / f"{document_id}.pdf",
            )
            self.assertTrue(
                path.is_relative_to(Path(temp_dir).resolve())
            )

    def test_resolve_document_path_rejects_mismatched_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                resolve_document_path(
                    temp_dir,
                    "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                    "other.pdf",
                )

    def test_resolve_document_path_rejects_path_traversal_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                resolve_document_path(
                    temp_dir,
                    "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                    "../other.pdf",
                )

    def test_resolve_document_path_rejects_invalid_id_and_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                resolve_document_path(
                    temp_dir,
                    "not-a-uuid",
                    "not-a-uuid.pdf",
                )

            with self.assertRaises(ValueError):
                resolve_document_path(
                    temp_dir,
                    "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                    "3f2504e0-4f89-41d3-9a0c-0305e82c3301.exe",
                )


if __name__ == "__main__":
    unittest.main()
