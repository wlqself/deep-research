import json
import tempfile
import unittest
from pathlib import Path

from deep_research.evaluation.rag_runner import (
    evaluate_retriever,
    load_eval_queries,
    load_qrels,
)


class RagRunnerTests(unittest.TestCase):
    def test_loaders_and_runner_share_query_and_qrels_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            queries_path = root / "queries.jsonl"
            queries_path.write_text(
                json.dumps({"_id": "q1", "text": "find alpha"}) + "\n"
                + json.dumps({"_id": "q2", "text": "find beta"}) + "\n",
                encoding="utf-8",
            )
            qrels_path = root / "qrels.tsv"
            qrels_path.write_text(
                "query-id\tcorpus-id\tscore\n"
                "q1\tc1\t2\n",
                encoding="utf-8",
            )

            queries = load_eval_queries(queries_path)
            qrels = load_qrels(qrels_path)
            result = evaluate_retriever(
                queries,
                qrels,
                lambda text, top_k: ["c1"] if "alpha" in text else ["noise"],
                top_k=1,
                k_values=(1,),
                retriever_name="fake",
            )

        self.assertEqual(result["retriever"], "fake")
        self.assertEqual(result["answerable_query_count"], 1)
        self.assertEqual(result["aggregate"]["recall@1"], 1.0)
        self.assertEqual(
            result["per_query"][0]["retrieved_chunk_ids"],
            ["c1"],
        )
        self.assertIsInstance(result["per_query"][0]["latency_ms"], float)


if __name__ == "__main__":
    unittest.main()
