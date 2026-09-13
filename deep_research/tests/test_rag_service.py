import unittest
from types import SimpleNamespace

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
from deep_research.rag.lexical import SqliteBm25Index


class FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.delete_calls: list[dict[str, object]] = []
        self.set_payload_calls: list[dict[str, object]] = []
        self.retrieve_records: list[object] = []
        self.retrieve_by_id: dict[str, object] = {}
        self.retrieve_error: Exception | None = None
        self.scroll_pages: list[tuple[list[object], object]] = []
        self.scroll_calls: list[dict[str, object]] = []

    def close(self) -> None:
        self.closed = True

    def delete(self, **kwargs: object) -> None:
        self.delete_calls.append(kwargs)

    def set_payload(self, **kwargs: object) -> None:
        self.set_payload_calls.append(kwargs)

    def retrieve(self, **kwargs: object) -> list[object]:
        if self.retrieve_error is not None:
            raise self.retrieve_error
        ids = kwargs.get("ids")
        if self.retrieve_by_id and isinstance(ids, list):
            return [
                self.retrieve_by_id[chunk_id]
                for chunk_id in ids
                if chunk_id in self.retrieve_by_id
            ]
        return self.retrieve_records

    def scroll(self, **kwargs: object) -> tuple[list[object], object]:
        self.scroll_calls.append(kwargs)
        if not self.scroll_pages:
            return [], None
        page_index = len(self.scroll_calls) - 1
        if page_index >= len(self.scroll_pages):
            return [], None
        return self.scroll_pages[page_index]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Document], list[str]]] = []
        self.search_calls: list[dict[str, object]] = []
        self.search_results: list[tuple[Document, float]] = []
        self.search_results_by_query: dict[str, list[tuple[Document, float]]] = {}

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
        return self.search_results_by_query.get(query, self.search_results)


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
    *,
    bm25_index=None,
    query_rewriter=None,
    candidate_k: int = 32,
    rrf_k: int = 60,
) -> tuple[RagService, FakeClient]:
    client = FakeClient()
    return RagService(
        client=client,
        embeddings=object(),
        vector_store=vector_store,
        collection_name="deep_research_documents",
        bm25_index=bm25_index,
        query_rewriter=query_rewriter,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    ), client


class FakeBm25Index:
    def __init__(self, ids=None, error: Exception | None = None) -> None:
        self.ids = list(ids or [])
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.status_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []

    def search_ids(self, query: str, *, top_k: int, document_id=None):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "document_id": document_id,
            }
        )
        if self.error is not None:
            raise self.error
        return self.ids[:top_k]

    def update_document_status(self, document_id: str, status: str) -> int:
        self.status_calls.append((document_id, status))
        return 0

    def delete_document(self, document_id: str) -> int:
        self.delete_calls.append(document_id)
        return 0


class FakeQueryRewriter:
    def __init__(self, terms=None, error: Exception | None = None) -> None:
        self.terms = list(terms or [])
        self.error = error
        self.queries: list[str] = []

    def rewrite(self, query: str):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(terms=self.terms)


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
        self.assertEqual(written_documents[0].page_content, chunks[0].page_content)
        self.assertEqual(written_documents[1].page_content, chunks[1].page_content)
        self.assertEqual(written_documents[0].metadata["status"], "pending")
        self.assertEqual(written_documents[1].metadata["status"], "pending")
        self.assertEqual(
            written_ids,
            ["chunk-1", "chunk-2"],
        )

    def test_metadata_weight_is_used_only_for_embedding_text(self):
        vector_store = FakeVectorStore()
        service, _ = make_service(vector_store)
        chunk = Document(
            page_content="原始 chunk 内容",
            metadata={
                "chunk_id": "chunk-1",
                "title": "CVIU paper",
                "filename": "CVIU.pdf",
            },
        )

        service.index_chunks([chunk])
        written_documents, _written_ids = vector_store.calls[0]
        self.assertIsNot(written_documents[0], chunk)
        self.assertIn("CVIU.pdf", written_documents[0].page_content)

        vector_store.search_results = [
            (written_documents[0], 0.9),
        ]
        results = service.search_chunks("CVIU", top_k=1)
        self.assertEqual(results[0][0].page_content, "原始 chunk 内容")

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

    def test_expand_results_to_parents_restores_ordered_parent_context(self):
        vector_store = FakeVectorStore()
        service, client = make_service(vector_store)
        client.retrieve_records = [
            SimpleNamespace(
                payload={
                    "page_content": "second child",
                    "metadata": {"chunk_index": 1},
                }
            ),
            SimpleNamespace(
                payload={
                    "page_content": "first child",
                    "metadata": {"chunk_index": 0},
                }
            ),
        ]

        results = service.expand_results_to_parents(
            [
                (
                    Document(
                        page_content="first child",
                        metadata={
                            "chunk_id": "c1",
                            "parent_id": "p1",
                            "parent_child_ids": ["c1", "c2"],
                        },
                    ),
                    0.8,
                )
            ]
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0][0].page_content,
            "first child\n\nsecond child",
        )
        self.assertEqual(results[0][0].metadata["parent_id"], "p1")

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

    def test_search_contexts_fuses_children_then_collapses_parents(self):
        vector_store = FakeVectorStore()
        dense_documents = [
            Document(
                page_content="first child",
                metadata={
                    "chunk_id": "c1",
                    "document_id": "doc-1",
                    "filename": "guide.pdf",
                    "status": "indexed",
                    "parent_id": "p1",
                    "parent_child_ids": ["c1", "c2"],
                    "chunk_index": 0,
                },
            ),
            Document(
                page_content="second child",
                metadata={
                    "chunk_id": "c2",
                    "document_id": "doc-1",
                    "filename": "guide.pdf",
                    "status": "indexed",
                    "parent_id": "p1",
                    "parent_child_ids": ["c1", "c2"],
                    "chunk_index": 1,
                },
            ),
        ]
        vector_store.search_results = [(dense_documents[0], 0.9), (dense_documents[1], 0.8)]
        bm25 = FakeBm25Index(ids=["c3", "c2"])
        rewriter = FakeQueryRewriter(terms=["expanded term"])
        service, client = make_service(
            vector_store,
            bm25_index=bm25,
            query_rewriter=rewriter,
            candidate_k=1,
            rrf_k=60,
        )
        c3 = Document(
            page_content="third child",
            metadata={
                "chunk_id": "c3",
                "document_id": "doc-1",
                "filename": "guide.pdf",
                "status": "indexed",
                "parent_id": "p2",
                "parent_child_ids": ["c3"],
                "chunk_index": 2,
            },
        )
        client.retrieve_by_id = {
            "c1": SimpleNamespace(payload={"page_content": "first child", "metadata": dense_documents[0].metadata}),
            "c2": SimpleNamespace(payload={"page_content": "second child", "metadata": dense_documents[1].metadata}),
            "c3": SimpleNamespace(payload={"page_content": "third child", "metadata": c3.metadata}),
        }

        results = service.search_contexts(
            "  raw query  ",
            top_k=2,
            document_id="  doc-1  ",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [document.metadata["parent_id"] for document, _score in results],
            ["p1", "p2"],
        )
        self.assertEqual(results[0][0].page_content, "first child\n\nsecond child")
        self.assertEqual(results[1][0].page_content, "third child")
        self.assertEqual(vector_store.search_calls[0]["query"], "raw query")
        self.assertEqual(vector_store.search_calls[0]["k"], 2)
        self.assertEqual(rewriter.queries, ["raw query"])
        self.assertIn("expanded term", bm25.calls[0]["query"])
        self.assertEqual(bm25.calls[0]["top_k"], 2)
        self.assertEqual(bm25.calls[0]["document_id"], "doc-1")

    def test_rewrite_failure_falls_back_to_original_lexical_query(self):
        vector_store = FakeVectorStore()
        vector_store.search_results = []
        bm25 = FakeBm25Index(ids=[])
        rewriter = FakeQueryRewriter(error=RuntimeError("rewrite unavailable"))
        service, _client = make_service(
            vector_store,
            bm25_index=bm25,
            query_rewriter=rewriter,
        )

        service.search_contexts("Why CVIU?", top_k=1)

        self.assertEqual(bm25.calls[0]["query"], "cviu")

    def test_bm25_failure_returns_dense_contexts(self):
        vector_store = FakeVectorStore()
        document = Document(
            page_content="dense result",
            metadata={
                "chunk_id": "c1",
                "document_id": "doc-1",
                "filename": "guide.pdf",
                "status": "indexed",
            },
        )
        vector_store.search_results = [(document, 0.9)]
        service, _client = make_service(
            vector_store,
            bm25_index=FakeBm25Index(error=RuntimeError("bm25 unavailable")),
        )

        results = service.search_contexts("query", top_k=1)

        self.assertEqual(results, [(document, 0.9)])

    def test_parent_refetch_failure_falls_back_to_triggering_child(self):
        vector_store = FakeVectorStore()
        document = Document(
            page_content="trigger child",
            metadata={
                "chunk_id": "c1",
                "document_id": "doc-1",
                "filename": "guide.pdf",
                "status": "indexed",
                "parent_id": "p1",
                "parent_child_ids": ["c1", "c2"],
            },
        )
        vector_store.search_results = [(document, 0.9)]
        client = FakeClient()
        bm25 = FakeBm25Index(ids=["c1"])
        service = RagService(
            client=client,
            embeddings=object(),
            vector_store=vector_store,
            collection_name="deep_research_documents",
            bm25_index=bm25,
        )
        client.retrieve_error = RuntimeError("qdrant unavailable")

        results = service.search_contexts("query", top_k=1)

        self.assertIs(results[0][0], document)
        self.assertAlmostEqual(results[0][1], 2 / 61)
        self.assertFalse(results[0][0].metadata.get("parent_content", False))

    def test_status_transitions_update_qdrant_and_bm25(self):
        vector_store = FakeVectorStore()
        bm25 = FakeBm25Index()
        service, client = make_service(vector_store, bm25_index=bm25)

        service.mark_document_chunks_pending(" doc-1 ")
        service.mark_document_chunks_indexed(" doc-1 ")
        service.delete_document_chunks(" doc-1 ")

        self.assertEqual(
            bm25.status_calls,
            [("doc-1", "pending"), ("doc-1", "indexed")],
        )
        self.assertEqual(bm25.delete_calls, ["doc-1"])
        self.assertEqual(len(client.set_payload_calls), 2)
        self.assertEqual(len(client.delete_calls), 1)

    def test_rebuild_lexical_index_scrolls_all_pages_and_ignores_non_indexed_payloads(self):
        vector_store = FakeVectorStore()
        bm25 = SqliteBm25Index()
        service, client = make_service(vector_store, bm25_index=bm25)
        client.scroll_pages = [
            (
                [
                    SimpleNamespace(
                        payload={
                            "page_content": "prefix\n__DEEP_RESEARCH_ORIGINAL_CONTENT__\nfirstalpha",
                            "metadata": {
                                "chunk_id": "c1",
                                "document_id": "doc-1",
                                "title": "Guide",
                                "status": "indexed",
                            },
                        }
                    ),
                    SimpleNamespace(
                        payload={
                            "page_content": "pending",
                            "metadata": {
                                "chunk_id": "pending",
                                "document_id": "doc-1",
                                "title": "Guide",
                                "status": "pending",
                            },
                        }
                    ),
                ],
                "next-page",
            ),
            (
                [
                    SimpleNamespace(
                        payload={
                            "page_content": "secondbeta",
                            "metadata": {
                                "chunk_id": "c2",
                                "document_id": "doc-1",
                                "title": "Guide",
                                "status": "indexed",
                            },
                        }
                    ),
                    SimpleNamespace(payload={"metadata": {"status": "indexed"}}),
                ],
                None,
            ),
        ]

        service.rebuild_lexical_index()

        self.assertEqual(len(client.scroll_calls), 2)
        self.assertEqual(bm25.search_ids("firstalpha", top_k=5), ["c1"])
        self.assertEqual(bm25.search_ids("secondbeta", top_k=5), ["c2"])
        self.assertEqual(bm25.search_ids("pending", top_k=5), [])
        bm25.close()

    def test_rebuild_failure_keeps_previous_bm25_contents(self):
        vector_store = FakeVectorStore()
        bm25 = SqliteBm25Index()
        bm25.upsert_chunks(
            [{
                "_id": "old",
                "title": "Old",
                "text": "old stable content",
                "metadata": {"document_id": "doc-old", "status": "indexed"},
            }]
        )
        service, client = make_service(vector_store, bm25_index=bm25)
        client.scroll_pages = [([], None)]

        from unittest.mock import patch
        with patch.object(bm25, "_upsert_rows_locked", side_effect=RuntimeError("rebuild failed")):
            with self.assertRaises(RuntimeError):
                service.rebuild_lexical_index()

        self.assertEqual(bm25.search_ids("old stable", top_k=1), ["old"])
        bm25.close()

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
