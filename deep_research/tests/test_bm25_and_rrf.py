import asyncio
import unittest

from deep_research.rag.lexical import (
    SqliteBm25Index,
    build_lexical_query,
)
from deep_research.rag.fusion import reciprocal_rank_fusion


class Bm25AndRrfTests(unittest.TestCase):
    def setUp(self):
        self.index = SqliteBm25Index()
        self.index.upsert_chunks(
            [
                {
                    "_id": "c1",
                    "title": "CVIU paper",
                    "text": "C1-C8 is a matched control experiment.",
                    "metadata": {
                        "document_id": "doc-cviu",
                        "status": "indexed",
                    },
                },
                {
                    "_id": "c2",
                    "title": "YOLO26 report",
                    "text": "Mamba attribution results are reported.",
                    "metadata": {
                        "document_id": "doc-yolo",
                        "status": "indexed",
                    },
                },
            ]
        )

    def tearDown(self):
        self.index.close()

    def test_bm25_matches_exact_terms_and_document_filter(self):
        self.assertEqual(
            self.index.search_ids("C1-C8 matched control", top_k=1),
            ["c1"],
        )
        self.assertEqual(
            self.index.search_ids("CVIU", top_k=1),
            ["c1"],
        )
        self.assertEqual(
            self.index.search_ids(
                "Mamba attribution",
                top_k=1,
                document_id="doc-yolo",
            ),
            ["c2"],
        )

    def test_lexical_query_keeps_entities_and_removes_question_words(self):
        lexical_query = build_lexical_query(
            "为什么 CVIU 论文对 YOLO26 中 Mamba 的总体结论是什么？"
        )
        self.assertIn("cviu", lexical_query)
        self.assertIn("yolo26", lexical_query)
        self.assertIn("mamba", lexical_query)
        self.assertNotIn("为什么", lexical_query)
        self.assertNotIn("总体", lexical_query)
        self.assertNotIn("结论", lexical_query)

    def test_rrf_rewards_agreement_and_deduplicates(self):
        fused = reciprocal_rank_fusion(
            [["a", "b", "c"], ["b", "a", "d"]],
            top_k=4,
            rrf_k=0,
        )
        self.assertEqual(fused[:2], ["a", "b"])
        self.assertEqual(len(fused), len(set(fused)))

    def test_pending_chunks_are_not_searchable_until_status_is_indexed(self):
        self.index.upsert_chunks(
            [
                {
                    "_id": "pending",
                    "title": "Pending",
                    "text": "pending-only term",
                    "metadata": {
                        "document_id": "doc-pending",
                        "status": "pending",
                    },
                }
            ]
        )
        self.assertEqual(
            self.index.search_ids("pending-only", top_k=5),
            [],
        )
        self.index.update_document_status("doc-pending", "indexed")
        self.assertEqual(
            self.index.search_ids("pending-only", top_k=5),
            ["pending"],
        )

    def test_missing_status_is_safe_and_not_indexed_by_default(self):
        self.index.upsert_chunks(
            [{
                "_id": "missing-status",
                "title": "Missing status",
                "text": "must stay hidden",
                "metadata": {"document_id": "doc-missing"},
            }]
        )
        self.assertEqual(
            self.index.search_ids("must stay hidden", top_k=5),
            [],
        )

    def test_search_from_asyncio_to_thread_works_after_main_thread_write(self):
        self.index.upsert_chunks(
            [
                {
                    "_id": "c3",
                    "title": "Third",
                    "text": "cross thread regression",
                    "metadata": {
                        "document_id": "doc-cviu",
                        "status": "indexed",
                    },
                }
            ]
        )

        async def search_in_worker() -> list[str]:
            return await asyncio.to_thread(
                self.index.search_ids,
                "cross thread",
                top_k=1,
            )

        self.assertEqual(asyncio.run(search_in_worker()), ["c3"])

    def test_concurrent_search_and_write_are_serialized(self):
        async def run_concurrently() -> None:
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self.index.search_ids,
                        "CVIU matched",
                        top_k=2,
                    )
                    for _ in range(8)
                ),
                asyncio.to_thread(
                    self.index.update_document_status,
                    "doc-cviu",
                    "pending",
                ),
                asyncio.to_thread(
                    self.index.update_document_status,
                    "doc-cviu",
                    "indexed",
                ),
            )

        asyncio.run(run_concurrently())
        self.index.update_document_status("doc-cviu", "indexed")
        self.assertEqual(
            self.index.search_ids("CVIU matched", top_k=2),
            ["c1"],
        )

    def test_close_is_idempotent(self):
        self.index.close()
        self.index.close()


if __name__ == "__main__":
    unittest.main()
