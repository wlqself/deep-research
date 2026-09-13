import unittest

from deep_research.evaluation.rag_metrics import (
    RetrievalCase,
    average_precision_at_k,
    evaluate_retrieval,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


class RagMetricsTests(unittest.TestCase):
    def setUp(self):
        self.ranked = ["noise", "core", "support"]
        self.relevance = {"core": 2, "support": 1}

    def test_basic_metrics_measure_rank_and_coverage(self):
        self.assertEqual(hit_rate_at_k(self.ranked, self.relevance, 1), 0.0)
        self.assertEqual(recall_at_k(self.ranked, self.relevance, 1), 0.0)
        self.assertEqual(recall_at_k(self.ranked, self.relevance, 3), 1.0)
        self.assertAlmostEqual(
            precision_at_k(self.ranked, self.relevance, 3),
            2 / 3,
        )
        self.assertAlmostEqual(
            reciprocal_rank_at_k(self.ranked, self.relevance, 3),
            1 / 2,
        )
        self.assertAlmostEqual(
            average_precision_at_k(self.ranked, self.relevance, 3),
            ((1 / 2) + (2 / 3)) / 2,
        )

    def test_ndcg_uses_relevance_grades(self):
        self.assertAlmostEqual(
            ndcg_at_k(["core", "support"], self.relevance, 2),
            1.0,
        )
        self.assertLess(
            ndcg_at_k(["support", "core"], self.relevance, 2),
            1.0,
        )

    def test_aggregate_excludes_unanswerable_queries(self):
        result = evaluate_retrieval(
            [
                RetrievalCase(
                    query_id="q1",
                    ranked_chunk_ids=["core"],
                    relevance={"core": 2},
                ),
                RetrievalCase(
                    query_id="q2",
                    ranked_chunk_ids=["other"],
                    relevance={},
                ),
            ],
            k_values=(1,),
        )

        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["answerable_query_count"], 1)
        self.assertEqual(result["unanswerable_query_count"], 1)
        self.assertEqual(result["aggregate"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
