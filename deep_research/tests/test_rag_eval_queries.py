import csv
import json
import unittest
from pathlib import Path


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
CORPUS_PATH = SOURCE_DIR / "corpus.jsonl"
QUERIES_PATH = SOURCE_DIR / "queries.jsonl"
REFERENCES_PATH = SOURCE_DIR / "references.jsonl"
QRELS_PATH = SOURCE_DIR / "qrels.tsv"


class RagEvaluationQueryTests(unittest.TestCase):
    def test_queries_and_qrels_reference_generated_corpus(self):
        corpus_ids = {
            json.loads(line)["_id"]
            for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        queries = [
            json.loads(line)
            for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        query_by_id = {query["_id"]: query for query in queries}
        references = [
            json.loads(line)
            for line in REFERENCES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reference_by_id = {
            reference["query_id"]: reference
            for reference in references
        }

        self.assertEqual(len(queries), 40)
        self.assertEqual(len(query_by_id), len(queries))
        self.assertEqual(len(references), len(queries))
        self.assertEqual(set(reference_by_id), set(query_by_id))

        qrels = []
        with QRELS_PATH.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            self.assertEqual(
                reader.fieldnames,
                ["query-id", "corpus-id", "score"],
            )
            qrels = list(reader)

        qrel_pairs = set()
        for qrel in qrels:
            pair = (qrel["query-id"], qrel["corpus-id"])
            self.assertNotIn(pair, qrel_pairs)
            qrel_pairs.add(pair)
            self.assertIn(qrel["query-id"], query_by_id)
            self.assertIn(qrel["corpus-id"], corpus_ids)
            self.assertIn(int(qrel["score"]), {1, 2})

        for query in queries:
            query_id = query["_id"]
            reference = reference_by_id[query_id]
            with self.subTest(query_id=query_id):
                self.assertTrue(query["text"].strip())
                self.assertTrue(query["metadata"]["type"].strip())
                self.assertTrue(reference["expected_answer"].strip())
                self.assertGreaterEqual(len(query["metadata"]["tags"]), 1)
                self.assertNotIn("relevant_chunk_ids", query)
                self.assertNotIn("expected_answer", query)

                actual_ids = {
                    corpus_id
                    for query_id, corpus_id in qrel_pairs
                    if query_id == query["_id"]
                }
                if reference["answerable"]:
                    self.assertGreaterEqual(len(actual_ids), 1)
                else:
                    self.assertEqual(actual_ids, set())


if __name__ == "__main__":
    unittest.main()
