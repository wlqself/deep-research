import json
import unittest
from pathlib import Path


CORPUS_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "rag_eval_sources_v1"
    / "corpus.jsonl"
)


class RagEvaluationCorpusTests(unittest.TestCase):
    def test_generated_corpus_has_stable_beir_compatible_rows(self):
        rows = [
            json.loads(line)
            for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertGreaterEqual(len(rows), 100)
        self.assertEqual(
            len({row["_id"] for row in rows}),
            len(rows),
        )

        for row in rows:
            with self.subTest(chunk_id=row["_id"]):
                self.assertTrue(row["_id"].strip())
                self.assertTrue(row["title"].strip())
                self.assertTrue(row["text"].strip())
                self.assertEqual(row["_id"], row["metadata"]["chunk_id"])
                self.assertNotIn("created_at", row["metadata"])
                self.assertTrue(row["metadata"]["document_id"].strip())
                self.assertGreaterEqual(row["metadata"]["chunk_index"], 0)


if __name__ == "__main__":
    unittest.main()
