import json
import unittest
from pathlib import Path


SOURCE_DIR = Path(__file__).parent / "fixtures" / "rag_eval_sources_v1"
MANIFEST_PATH = SOURCE_DIR / "pdf_sources.json"


class RagEvaluationPdfManifestTests(unittest.TestCase):
    def test_manifest_contains_only_non_personal_eval_sources(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        documents = manifest["documents"]

        self.assertEqual(manifest["version"], "rag-eval-pdf-sources-v1")
        self.assertEqual(len(documents), 6)
        self.assertNotIn("personal-document", {
            document["category"] for document in documents
        })

        document_ids = [document["document_id"] for document in documents]
        filenames = [document["filename"] for document in documents]
        self.assertEqual(len(document_ids), len(set(document_ids)))
        self.assertEqual(len(filenames), len(set(filenames)))

        for document in documents:
            with self.subTest(filename=document["filename"]):
                self.assertEqual(document["mime_type"], "application/pdf")
                self.assertTrue(document["category"].strip())
                self.assertTrue((SOURCE_DIR / document["filename"]).is_file())


if __name__ == "__main__":
    unittest.main()
