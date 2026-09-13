import hashlib
import unittest

from langchain_core.documents import Document

from deep_research.rag.chunking import chunk_documents


def make_document(
    content: str,
    *,
    document_id: str = "doc-1",
    page_number: int = 1,
    section_title: str = "安装",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "collection_id": "deep_research_documents",
            "filename": "guide.md",
            "mime_type": "text/markdown",
            "page_number": page_number,
            "section_title": section_title,
        },
    )


class ChunkingTests(unittest.TestCase):
    def test_short_document_creates_chunk_with_metadata(self):
        chunks = chunk_documents(
            [make_document("安装依赖。")],
            document_hash="document-hash",
            chunk_size=100,
            chunk_overlap=10,
        )

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]

        self.assertEqual(chunk.page_content, "安装依赖。")
        self.assertEqual(chunk.metadata["document_id"], "doc-1")
        self.assertEqual(chunk.metadata["page_number"], 1)
        self.assertEqual(chunk.metadata["section_title"], "安装")
        self.assertEqual(
            chunk.metadata["document_hash"],
            "document-hash",
        )
        self.assertEqual(
            chunk.metadata["content_hash"],
            hashlib.sha256(
                "安装依赖。".encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(chunk.metadata["chunk_index"], 0)
        self.assertEqual(chunk.metadata["parent_id"], "doc-1:parent:0")
        self.assertEqual(chunk.metadata["parent_child_ids"], [chunk.metadata["chunk_id"]])
        self.assertEqual(chunk.metadata["status"], "pending")

    def test_long_document_is_split_with_overlap(self):
        content = "甲乙丙丁戊己庚辛壬癸" * 4

        chunks = chunk_documents(
            [make_document(content)],
            document_hash="document-hash",
            chunk_size=10,
            chunk_overlap=2,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(
                len(chunk.page_content) <= 10
                for chunk in chunks
            )
        )

        for previous, current in zip(
            chunks,
            chunks[1:],
        ):
            self.assertEqual(
                previous.page_content[-2:],
                current.page_content[:2],
            )

    def test_chunks_do_not_cross_documents(self):
        chunks = chunk_documents(
            [
                make_document(
                    "第一个文档内容。" * 5,
                    document_id="doc-1",
                ),
                make_document(
                    "第二个文档内容。" * 5,
                    document_id="doc-2",
                ),
            ],
            document_hash="document-hash",
            chunk_size=8,
            chunk_overlap=2,
        )

        document_ids = {
            chunk.metadata["document_id"]
            for chunk in chunks
        }

        self.assertEqual(
            document_ids,
            {"doc-1", "doc-2"},
        )
        self.assertTrue(
            all(
                chunk.metadata["document_id"]
                in {"doc-1", "doc-2"}
                for chunk in chunks
            )
        )
        self.assertEqual(
            len({
                chunk.metadata["chunk_id"]
                for chunk in chunks
            }),
            len(chunks),
        )

    def test_chunk_ids_are_stable(self):
        documents = [
            make_document("稳定的内容。" * 5),
        ]

        first = chunk_documents(
            documents,
            document_hash="document-hash",
            chunk_size=8,
            chunk_overlap=2,
        )
        second = chunk_documents(
            documents,
            document_hash="document-hash",
            chunk_size=8,
            chunk_overlap=2,
        )

        self.assertEqual(
            [chunk.metadata["chunk_id"] for chunk in first],
            [chunk.metadata["chunk_id"] for chunk in second],
        )


if __name__ == "__main__":
    unittest.main()
