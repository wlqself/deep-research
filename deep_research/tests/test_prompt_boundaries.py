import unittest

from deep_research.prompts.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
)
from deep_research.prompts.supervisor import (
    SUPERVISOR_SYSTEM_PROMPT,
)


class PromptBoundaryTests(unittest.TestCase):
    def test_supervisor_prompt_describes_coordination(self):
        self.assertIn("Main Agent", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("task", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("save_report", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn(
            "list_research_findings",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "两个相互独立",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "同一条 AIMessage",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "等待全部对应 ToolMessage",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "不要为了展示多 Agent",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Main 不直接调用 web_search",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "已上传的简历、文档或本地知识库",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertIn(
            "search_knowledge_base",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertNotIn(
            "不要使用长期记忆、RAG、向量数据库",
            SUPERVISOR_SYSTEM_PROMPT,
        )

    def test_researcher_prompt_describes_research_execution(self):
        self.assertIn("Researcher", RESEARCHER_SYSTEM_PROMPT)
        self.assertIn("搜索网页", RESEARCHER_SYSTEM_PROMPT)
        self.assertIn("阅读来源", RESEARCHER_SYSTEM_PROMPT)
        self.assertIn("findings", RESEARCHER_SYSTEM_PROMPT)
        self.assertIn("ResearcherResult", RESEARCHER_SYSTEM_PROMPT)
        self.assertIn(
            "第一步必须调用 search_knowledge_base",
            RESEARCHER_SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
