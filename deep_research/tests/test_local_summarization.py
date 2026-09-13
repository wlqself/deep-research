import unittest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from deep_research.middleware.local_summarization import (
    LocalTokenSummarizationMiddleware,
)


class LocalTokenSummarizationTests(unittest.TestCase):
    def _middleware(self, threshold=1000):
        model = GenericFakeChatModel(messages=iter(["summary"]))
        return LocalTokenSummarizationMiddleware(
            model=model,
            trigger=("tokens", threshold),
            keep=("messages", 2),
        )

    def test_large_provider_total_does_not_trigger_for_short_messages(self):
        middleware = self._middleware()
        messages = [
            HumanMessage(content="short question"),
            AIMessage(
                content="short answer",
                usage_metadata={
                    "input_tokens": 678_000,
                    "output_tokens": 518,
                    "total_tokens": 678_518,
                },
                response_metadata={"model_provider": "openai"},
            ),
        ]
        local_tokens = middleware.token_counter(messages)

        self.assertLess(local_tokens, 1000)
        self.assertFalse(
            middleware._should_summarize(messages, local_tokens)
        )

    def test_local_message_size_still_triggers_summarization(self):
        middleware = self._middleware(threshold=100)
        messages = [HumanMessage(content="local context " * 500)]
        local_tokens = middleware.token_counter(messages)

        self.assertGreaterEqual(local_tokens, 100)
        self.assertTrue(
            middleware._should_summarize(messages, local_tokens)
        )


if __name__ == "__main__":
    unittest.main()
