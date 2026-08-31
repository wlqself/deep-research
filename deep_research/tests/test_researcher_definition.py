import unittest

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)

from deep_research.agent.sub_agent import (
    RESEARCHER_TOOLS,
    build_researcher,
)


class BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ResearcherDefinitionTests(unittest.TestCase):
    def test_researcher_has_only_research_tools(self):
        researcher = build_researcher(
            model=BindableFakeChatModel(messages=iter([]))
        )

        self.assertEqual(
            researcher["name"],
            "researcher",
        )

        self.assertIn(
            "复杂、多步骤网页研究",
            researcher["description"],
        )

        actual_tool_names = {tool.name for tool in RESEARCHER_TOOLS}

        expected_tool_names = {
            "web_search",
            "read_page",
            "assess_research",
            "record_research_finding",
            "list_research_findings",
        }

        self.assertEqual(
            actual_tool_names,
            expected_tool_names,
        )

        self.assertNotIn(
            "save_report",
            actual_tool_names,
        )

        self.assertNotIn(
            "task",
            actual_tool_names,
        )

        self.assertEqual(
            len(RESEARCHER_TOOLS),
            5,
        )

    def test_researcher_uses_tool_strategy(self):
        researcher = build_researcher(
            model=BindableFakeChatModel(messages=iter([]))
        )

        self.assertIn("runnable", researcher)
        self.assertTrue(hasattr(researcher["runnable"], "ainvoke"))


if __name__ == "__main__":
    unittest.main()
