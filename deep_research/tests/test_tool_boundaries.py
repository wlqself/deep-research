import unittest

from deep_research.tools import (
    RESEARCHER_TOOLS,
    RESEARCH_TOOLS,
    SUPERVISOR_TOOLS,
    build_supervisor_tools,
    build_researcher_tools,
)


class ToolBoundaryTests(unittest.TestCase):
    def test_researcher_tools_are_restricted(self):
        names = {
            tool.name
            for tool in RESEARCHER_TOOLS
        }

        self.assertEqual(
            names,
            {
                "web_search",
                "read_page",
                "assess_research",
                "record_research_finding",
                "list_research_findings",
            },
        )

        self.assertNotIn("save_report", names)

    def test_supervisor_tools_are_restricted(self):
        names = {
            tool.name
            for tool in SUPERVISOR_TOOLS
        }

        self.assertEqual(
            names,
            {
                "list_research_findings",
                "save_report",
            },
        )

        self.assertNotIn("web_search", names)
        self.assertNotIn("read_page", names)
        self.assertNotIn("record_research_finding", names)

    def test_old_research_tools_export_is_preserved(self):
        names = {
            tool.name
            for tool in RESEARCH_TOOLS
        }

        self.assertEqual(
            names,
            {
                "web_search",
                "read_page",
                "assess_research",
                "record_research_finding",
                "list_research_findings",
                "save_report",
            },
        )

    def test_rag_tool_is_added_only_for_researcher_with_service(self):
        without_rag_names = {
            tool.name
            for tool in build_researcher_tools()
        }

        with_rag_names = {
            tool.name
            for tool in build_researcher_tools(
                object()
            )
        }

        self.assertNotIn(
            "search_knowledge_base",
            without_rag_names,
        )
        self.assertIn(
            "search_knowledge_base",
            with_rag_names,
        )
        self.assertNotIn(
            "search_knowledge_base",
            {
                tool.name
                for tool in SUPERVISOR_TOOLS
            },
        )

    def test_memory_tools_are_added_only_to_dynamic_supervisor_tools(self):
        without_memory = {
            tool.name
            for tool in build_supervisor_tools()
        }
        with_memory = {
            tool.name
            for tool in build_supervisor_tools(object())
        }
        researcher_names = {
            tool.name
            for tool in build_researcher_tools(object())
        }

        self.assertEqual(
            without_memory,
            {
                "list_research_findings",
                "save_report",
            },
        )
        self.assertIn("remember_memory", with_memory)
        self.assertIn("recall_memories", with_memory)
        self.assertIn("list_memories", with_memory)
        self.assertIn("forget_memory", with_memory)
        self.assertNotIn("remember_memory", researcher_names)
        self.assertNotIn("recall_memories", researcher_names)
        self.assertNotIn("list_memories", researcher_names)
        self.assertNotIn("forget_memory", researcher_names)

    def test_image_generation_is_main_only_when_injected(self):
        names = {
            tool.name
            for tool in build_supervisor_tools(
                publishing_service=object(),
                image_generation_service=object(),
            )
        }
        self.assertIn("generate_image", names)
        self.assertNotIn("generate_image", {
            tool.name for tool in build_researcher_tools()
        })


if __name__ == "__main__":
    unittest.main()
