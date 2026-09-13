import json
import unittest
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rag_eval_v1.json"


class RagEvaluationDatasetTests(unittest.TestCase):
    def test_fixture_has_stable_retrieval_contract(self):
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], "rag-eval-v1")
        self.assertTrue(payload["corpus_id"].strip())
        self.assertTrue(payload["description"].strip())

        documents = payload["documents"]
        queries = payload["queries"]
        self.assertGreaterEqual(len(documents), 6)
        self.assertGreaterEqual(len(queries), 12)

        document_ids = [document["document_id"] for document in documents]
        self.assertEqual(len(document_ids), len(set(document_ids)))

        chunks = [chunk for document in documents for chunk in document["chunks"]]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.assertEqual(len(chunks_by_id), len(chunks))

        query_ids = [query["query_id"] for query in queries]
        self.assertEqual(len(query_ids), len(set(query_ids)))

        for document in documents:
            with self.subTest(document=document["document_id"]):
                self.assertTrue(document["filename"].strip())
                self.assertGreaterEqual(len(document["chunks"]), 1)
                for chunk in document["chunks"]:
                    self.assertTrue(
                        chunk["chunk_id"].startswith(document["document_id"] + ":")
                    )
                    self.assertTrue(chunk["section_title"].strip())
                    self.assertTrue(chunk["text"].strip())

        for query in queries:
            with self.subTest(query=query["query_id"]):
                self.assertTrue(query["query"].strip())
                self.assertGreaterEqual(len(query["relevant_chunk_ids"]), 1)
                self.assertGreaterEqual(len(query["tags"]), 1)
                for chunk_id in query["relevant_chunk_ids"]:
                    self.assertIn(chunk_id, chunks_by_id)


if __name__ == "__main__":
    unittest.main()
