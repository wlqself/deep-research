import unittest

from langchain_core.messages import AIMessage

from deep_research.rag.query_rewrite import (
    QueryRewriter,
    RetrievalQueryRewrite,
    build_rewritten_lexical_query,
)
from deep_research.evaluation.reranker import build_reranker_query


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.response


class FakeModel:
    def __init__(self, structured_model):
        self.structured_model = structured_model

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_model


class PlainFakeModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.response)


class QueryRewriteTests(unittest.TestCase):
    def test_structured_rewriter_normalizes_and_deduplicates_terms(self):
        structured = FakeStructuredModel(
            {
                "terms": ["CVIU", "training seeds", "CVIU", "   "],
            }
        )
        rewriter = QueryRewriter(FakeModel(structured))

        result = rewriter.rewrite("CVIU 实验使用了多少个随机种子？")

        self.assertIsInstance(result, RetrievalQueryRewrite)
        self.assertEqual(result.terms, ["CVIU", "training seeds"])
        self.assertEqual(len(structured.messages), 2)

    def test_rewritten_query_preserves_original_lexical_signals(self):
        rewritten = build_rewritten_lexical_query(
            "为什么 CVIU 论文对 YOLO26 中 Mamba 的总体结论是什么？",
            ["matched control", "CVIU"],
        )

        self.assertIn("cviu", rewritten)
        self.assertIn("yolo26", rewritten)
        self.assertIn("mamba", rewritten)
        self.assertIn("matched control", rewritten)

    def test_reranker_query_keeps_original_query_and_rewrite_terms(self):
        reranker_query = build_reranker_query(
            "为什么 C1-C8 是 matched control？",
            ["matched control", "near-parameter-matched"],
        )

        self.assertIn("原始问题：为什么 C1-C8 是 matched control？", reranker_query)
        self.assertIn("检索扩展词：matched control；near-parameter-matched", reranker_query)

    def test_rewriter_rejects_empty_query(self):
        rewriter = QueryRewriter(FakeModel(FakeStructuredModel({"terms": []})))
        with self.assertRaises(ValueError):
            rewriter.rewrite("   ")

    def test_plain_json_mode_is_validated_locally(self):
        rewriter = QueryRewriter(
            PlainFakeModel('```json\n{"terms": ["CVIU", "seed"]}\n```'),
            structured_output=False,
        )

        result = rewriter.rewrite("CVIU 实验使用了随机种子")

        self.assertEqual(result.terms, ["CVIU", "seed"])


if __name__ == "__main__":
    unittest.main()
