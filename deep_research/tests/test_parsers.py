import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from deep_research.rag.parsers import (
    parse_markdown_document,
    parse_document,
    parse_txt_document,
)


class TxtParserTests(unittest.TestCase):
    def test_parse_utf8_txt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "guide.txt"
            path.write_text(
                "第一段内容\n第二段内容",
                encoding="utf-8",
            )

            documents = parse_txt_document(
                path,
                document_id="doc-1",
                collection_id="deep_research_documents",
                filename="guide.txt",
                mime_type="text/plain",
            )

            self.assertEqual(len(documents), 1)
            self.assertEqual(
                documents[0].page_content,
                "第一段内容\n第二段内容",
            )
            self.assertEqual(
                documents[0].metadata["document_id"],
                "doc-1",
            )
            self.assertEqual(
                documents[0].metadata["page_number"],
                1,
            )

    def test_empty_txt_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.txt"
            path.write_text("", encoding="utf-8")

            documents = parse_txt_document(
                path,
                document_id="doc-1",
                collection_id="deep_research_documents",
                filename="empty.txt",
                mime_type="text/plain",
            )

            self.assertEqual(documents, [])

    def test_invalid_utf8_raises_unicode_decode_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.txt"
            path.write_bytes(b"\xff\xfe\xfd")

            with self.assertRaises(UnicodeDecodeError):
                parse_txt_document(
                    path,
                    document_id="doc-1",
                    collection_id="deep_research_documents",
                    filename="invalid.txt",
                    mime_type="text/plain",
                )


class MarkdownParserTests(unittest.TestCase):
    def test_parse_markdown_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "guide.md"
            path.write_text(
                "# 安装\n\n安装依赖。\n\n"
                "## 配置\n\n配置 API Key。",
                encoding="utf-8",
            )

            documents = parse_markdown_document(
                path,
                document_id="doc-1",
                collection_id="deep_research_documents",
                filename="guide.md",
                mime_type="text/markdown",
            )

            self.assertEqual(len(documents), 2)
            self.assertEqual(
                documents[0].metadata["section_title"],
                "安装",
            )
            self.assertEqual(
                documents[0].page_content,
                "安装依赖。",
            )
            self.assertEqual(
                documents[1].metadata["section_title"],
                "配置",
            )
            self.assertEqual(
                documents[1].page_content,
                "配置 API Key。",
            )

    def test_markdown_without_heading_returns_one_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "guide.md"
            path.write_text(
                "没有标题的 Markdown 正文。",
                encoding="utf-8",
            )

            documents = parse_markdown_document(
                path,
                document_id="doc-1",
                collection_id="deep_research_documents",
                filename="guide.md",
                mime_type="text/markdown",
            )

            self.assertEqual(len(documents), 1)
            self.assertEqual(
                documents[0].metadata["section_title"],
                "",
            )

    def test_empty_markdown_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.md"
            path.write_text("", encoding="utf-8")

            documents = parse_markdown_document(
                path,
                document_id="doc-1",
                collection_id="deep_research_documents",
                filename="empty.md",
                mime_type="text/markdown",
            )

            self.assertEqual(documents, [])

    def test_invalid_utf8_markdown_raises_unicode_decode_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.md"
            path.write_bytes(b"\xff\xfe\xfd")

            with self.assertRaises(UnicodeDecodeError):
                parse_markdown_document(
                    path,
                    document_id="doc-1",
                    collection_id="deep_research_documents",
                    filename="invalid.md",
                    mime_type="text/markdown",
                )


class UnifiedParserTests(unittest.TestCase):
    def test_pdf_parser_sanitizes_unpaired_unicode_surrogates(self):
        class FakePage:
            def extract_text(self):
                return "正常文本\ud835之后的文本"

        class FakeReader:
            pages = [FakePage()]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "surrogate.pdf"
            path.write_bytes(b"not-used-by-fake-reader")

            with patch(
                "deep_research.rag.parsers.PdfReader",
                return_value=FakeReader(),
            ):
                documents, page_count = parse_document(
                    path,
                    document_id="doc-1",
                    collection_id="deep_research_documents",
                    filename="surrogate.pdf",
                    mime_type="application/pdf",
                )

        self.assertEqual(page_count, 1)
        self.assertEqual(len(documents), 1)
        self.assertNotIn("\ud835", documents[0].page_content)
        self.assertIn("�", documents[0].page_content)

    def test_txt_dispatch_returns_one_page(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "guide.txt"
            path.write_text("TXT 内容", encoding="utf-8")

            documents, page_count = parse_document(
                path,
                document_id="doc-1",
                collection_id="deep_research_documents",
                filename="guide.txt",
                mime_type="text/plain",
            )

            self.assertEqual(len(documents), 1)
            self.assertEqual(page_count, 1)

    def test_empty_markdown_dispatch_returns_zero_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.md"
            path.write_text("", encoding="utf-8")

            documents, page_count = parse_document(
                path,
                document_id="doc-1",
                collection_id="deep_research_documents",
                filename="empty.md",
                mime_type="text/markdown",
            )

            self.assertEqual(documents, [])
            self.assertEqual(page_count, 0)

    def test_pdf_dispatch_returns_total_page_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.add_blank_page(width=100, height=100)

            with path.open("wb") as target:
                writer.write(target)

            documents, page_count = parse_document(
                path,
                document_id="doc-1",
                collection_id="deep_research_documents",
                filename="empty.pdf",
                mime_type="application/pdf",
            )

            self.assertEqual(documents, [])
            self.assertEqual(page_count, 2)

    def test_unsupported_mime_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "guide.bin"
            path.write_bytes(b"content")

            with self.assertRaises(ValueError):
                parse_document(
                    path,
                    document_id="doc-1",
                    collection_id="deep_research_documents",
                    filename="guide.bin",
                    mime_type="application/octet-stream",
                )


if __name__ == "__main__":
    unittest.main()
