import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    VectorParams,
)

from deep_research.rag.service import RagService


class FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.delete_calls: list[dict[str, object]] = []
        self.set_payload_calls: list[dict[str, object]] = []

    def close(self) -> None:
        self.closed = True

    def delete(self, **kwargs: object) -> None:
        self.delete_calls.append(kwargs)

    def set_payload(self, **kwargs: object) -> None:
        self.set_payload_calls.append(kwargs)


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Document], list[str]]] = []
        self.search_calls: list[dict[str, object]] = []
        self.search_results: list[tuple[Document, float]] = []

    def add_documents(
        self,
        *,
        documents: list[Document],
        ids: list[str],
    ) -> list[str]:
        self.calls.append((documents, ids))
        return ids

    def similarity_search_with_score(
        self,
        *,
        query: str,
        k: int,
        filter: Filter,
    ) -> list[tuple[Document, float]]:
        self.search_calls.append(
            {
                "query": query,
                "k": k,
                "filter": filter,
            }
        )
        return self.search_results


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.document_calls.append(texts)

        return [
            [
                float(len(text)),
                1.0,
                0.0,
                0.0,
            ]
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [
            float(len(text)),
            1.0,
            0.0,
            0.0,
        ]


def make_service(
    vector_store: FakeVectorStore,
) -> tuple[RagService, FakeClient]:
    client = FakeClient()
    return RagService(
        client=client,
        embeddings=object(),
        vector_store=vector_store,
        collection_name="deep_research_documents",
    ), client


class RagServiceTests(unittest.TestCase):
    def test_empty_chunks_do_not_write(self):
        vector_store = FakeVectorStore()
        service, _ = make_service(vector_store)

        self.assertEqual(service.index_chunks([]), 0)
        self.assertEqual(vector_store.calls, [])

    def test_index_chunks_passes_documents_and_ids(self):
        vector_store = FakeVectorStore()
        service, _ = make_service(vector_store)
        chunks = [
            Document(
                page_content="第一段",
                metadata={"chunk_id": "chunk-1"},
            ),
            Document(
                page_content="第二段",
                metadata={"chunk_id": "chunk-2"},
            ),
        ]

        count = service.index_chunks(chunks)

        self.assertEqual(count, 2)
        self.assertEqual(len(vector_store.calls), 1)
        written_documents, written_ids = vector_store.calls[0]
        self.assertIs(written_documents[0], chunks[0])
        self.assertIs(written_documents[1], chunks[1])
        self.assertEqual(
            written_ids,
            ["chunk-1", "chunk-2"],
        )

    def test_missing_chunk_id_is_rejected_before_write(self):
        vector_store = FakeVectorStore()
        service, _ = make_service(vector_store)
        chunks = [
            Document(
                page_content="没有 ID",
                metadata={},
            )
        ]

        with self.assertRaises(ValueError):
            service.index_chunks(chunks)

        self.assertEqual(vector_store.calls, [])

    def test_search_chunks_uses_indexed_filter_and_document_filter(self):
        vector_store = FakeVectorStore()
        expected = [
            (
                Document(
                    page_content="命中内容",
                    metadata={"chunk_id": "chunk-1"},
                ),
                0.91,
            )
        ]
        vector_store.search_results = expected
        service, _ = make_service(vector_store)

        results = service.search_chunks(
            "  查询内容  ",
            top_k=3,
            document_id="  doc-1  ",
        )

        self.assertEqual(results, expected)
        self.assertEqual(len(vector_store.search_calls), 1)
        call = vector_store.search_calls[0]
        self.assertEqual(call["query"], "查询内容")
        self.assertEqual(call["k"], 3)

        search_filter = call["filter"]
        conditions = search_filter.must
        self.assertEqual(
            conditions[0].key,
            "metadata.status",
        )
        self.assertEqual(
            conditions[0].match.value,
            "indexed",
        )
        self.assertEqual(
            conditions[1].key,
            "metadata.document_id",
        )
        self.assertEqual(
            conditions[1].match.value,
            "doc-1",
        )

    def test_search_chunks_rejects_invalid_query_and_top_k(self):
        vector_store = FakeVectorStore()
        service, _ = make_service(vector_store)

        with self.assertRaises(ValueError):
            service.search_chunks("   ", top_k=3)

        with self.assertRaises(ValueError):
            service.search_chunks("query", top_k=0)

        with self.assertRaises(ValueError):
            service.search_chunks(
                "query",
                top_k=3,
                document_id="   ",
            )

        self.assertEqual(vector_store.search_calls, [])

    def test_delete_document_chunks_uses_metadata_filter(self):
        vector_store = FakeVectorStore()
        service, client = make_service(vector_store)

        service.delete_document_chunks("  doc-123  ")

        self.assertEqual(len(client.delete_calls), 1)
        call = client.delete_calls[0]
        self.assertEqual(
            call["collection_name"],
            "deep_research_documents",
        )
        self.assertTrue(call["wait"])

        selector = call["points_selector"]
        self.assertEqual(
            selector.must[0].key,
            "metadata.document_id",
        )
        self.assertEqual(
            selector.must[0].match.value,
            "doc-123",
        )

    def test_delete_document_chunks_rejects_empty_id(self):
        vector_store = FakeVectorStore()
        service, client = make_service(vector_store)

        with self.assertRaises(ValueError):
            service.delete_document_chunks("   ")

        self.assertEqual(client.delete_calls, [])

    def test_mark_pending_updates_metadata_status(self):
        vector_store = FakeVectorStore()
        service, client = make_service(vector_store)

        service.mark_document_chunks_pending("  doc-123  ")

        self.assertEqual(len(client.set_payload_calls), 1)
        call = client.set_payload_calls[0]
        self.assertEqual(
            call["collection_name"],
            "deep_research_documents",
        )
        self.assertEqual(call["payload"], {"status": "pending"})
        self.assertEqual(call["key"], "metadata")
        self.assertTrue(call["wait"])
        selector = call["points"]
        self.assertEqual(
            selector.must[0].key,
            "metadata.document_id",
        )
        self.assertEqual(
            selector.must[0].match.value,
            "doc-123",
        )

    def test_mark_pending_rejects_empty_id(self):
        vector_store = FakeVectorStore()
        service, client = make_service(vector_store)

        with self.assertRaises(ValueError):
            service.mark_document_chunks_pending("   ")

        self.assertEqual(client.set_payload_calls, [])

    def test_index_chunks_vectorizes_and_writes_to_qdrant(self):
        client = QdrantClient(":memory:")

        try:
            client.create_collection(
                collection_name="test_documents",
                vectors_config=VectorParams(
                    size=4,
                    distance=Distance.COSINE,
                ),
            )

            embeddings = FakeEmbeddings()
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="test_documents",
                embedding=embeddings,
                validate_collection_config=False,
            )
            service = RagService(
                client=client,
                embeddings=embeddings,
                vector_store=vector_store,
                collection_name="test_documents",
            )

            chunks = [
                Document(
                    page_content="第一段内容",
                    metadata={
                        "chunk_id": (
                            "00000000-0000-0000-0000-000000000001"
                        ),
                        "document_id": "doc-1",
                        "status": "pending",
                    },
                ),
                Document(
                    page_content="第二段内容",
                    metadata={
                        "chunk_id": (
                            "00000000-0000-0000-0000-000000000002"
                        ),
                        "document_id": "doc-1",
                        "status": "pending",
                    },
                ),
            ]

            count = service.index_chunks(chunks)

            self.assertEqual(count, 2)
            self.assertTrue(embeddings.document_calls)
            self.assertEqual(
                embeddings.document_calls[0],
                ["第一段内容", "第二段内容"],
            )

            stored = client.retrieve(
                collection_name="test_documents",
                ids=[
                    "00000000-0000-0000-0000-000000000001",
                    "00000000-0000-0000-0000-000000000002",
                ],
                with_vectors=True,
                with_payload=True,
            )

            self.assertEqual(len(stored), 2)
            self.assertEqual(len(stored[0].vector), 4)
            self.assertEqual(
                stored[0].payload["metadata"]["document_id"],
                "doc-1",
            )

            self.assertEqual(
                service.search_chunks(
                    "第一段内容",
                    top_k=2,
                ),
                [],
            )

            service.mark_document_chunks_pending("doc-1")
            pending = client.retrieve(
                collection_name="test_documents",
                ids=[
                    "00000000-0000-0000-0000-000000000001",
                ],
                with_payload=True,
            )[0]
            self.assertEqual(
                pending.payload["metadata"]["status"],
                "pending",
            )
            self.assertNotIn("status", pending.payload)

            service.mark_document_chunks_indexed("doc-1")
            indexed = client.retrieve(
                collection_name="test_documents",
                ids=[
                    "00000000-0000-0000-0000-000000000001",
                ],
                with_payload=True,
            )[0]
            self.assertEqual(
                indexed.payload["metadata"]["status"],
                "indexed",
            )
            self.assertNotIn("status", indexed.payload)
            results = service.search_chunks(
                "第一段内容",
                top_k=2,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(
                all(
                    document.metadata["document_id"] == "doc-1"
                    for document, _score in results
                )
            )
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
