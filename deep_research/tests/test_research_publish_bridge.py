import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import deep_research.agent.factory as factory_module
from deep_research.context import ResearchContext
from deep_research.publishing.artifacts import ArtifactService
from deep_research.publishing.models import ArticleStatus
from deep_research.publishing.repository import PublishingRepository
from deep_research.publishing.service import PublishingService


THREAD_ID = "research-publish-bridge-thread"
ARTIFACT_ID = "abcdef1234567890abcdef1234567890"
REPORT = "# GPT-6 research\n\nVerified findings for publication."


class _BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class _LiveReader:
    agent = None

    async def aget_state(self, config):
        return await self.agent.aget_state(config)


class ResearchPublishBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_report_artifact_can_immediately_prepare_article(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = PublishingRepository(
                Path(directory) / "publishing.sqlite"
            )
            repository.initialize()
            try:
                reader = _LiveReader()
                service = PublishingService(
                    repository,
                    ArtifactService(reader),
                )
                fake_model = _BindableFakeChatModel(
                    messages=iter(
                        [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "save_report",
                                    "args": {
                                        "title": "GPT-6 research",
                                        "content": REPORT,
                                    },
                                    "id": "save-report-call",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "prepare_article_for_publication",
                                    "args": {
                                        "artifact_id": ARTIFACT_ID,
                                        "title": "GPT-6 latest research",
                                        "slug": "gpt-6-latest-research",
                                        "excerpt": "Verified GPT-6 findings.",
                                        "tags": ["GPT-6", "OpenAI"],
                                    },
                                    "id": "prepare-article-call",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(content="Article draft prepared."),
                        ]
                    )
                )
                metadata = {
                    "artifact_id": ARTIFACT_ID,
                    "filename": "gpt-6-research.md",
                    "workspace_path": "/final/gpt-6-research.md",
                }

                with (
                    patch.object(factory_module, "model", fake_model),
                    patch(
                        "deep_research.tools.save_report.build_artifact_metadata",
                        return_value=metadata,
                    ),
                ):
                    agent = factory_module.build_agent(
                        InMemorySaver(),
                        publishing_service=service,
                    )
                    reader.agent = agent
                    result = await agent.ainvoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Research GPT-6 and publish it.",
                                }
                            ]
                        },
                        config={"configurable": {"thread_id": THREAD_ID}},
                        context=ResearchContext.from_settings(),
                    )

                articles = repository.list_articles()
                self.assertEqual(len(articles), 1)
                self.assertEqual(articles[0].status, ArticleStatus.DRAFT)
                self.assertEqual(articles[0].source_artifact_id, ARTIFACT_ID)
                self.assertEqual(
                    articles[0].source_artifact_sha256,
                    hashlib.sha256(REPORT.encode("utf-8")).hexdigest(),
                )
                self.assertEqual(
                    result["messages"][-1].content,
                    "Article draft prepared.",
                )
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
