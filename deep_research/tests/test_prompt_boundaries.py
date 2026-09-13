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

    def test_supervisor_prompt_describes_publication_request_boundary(self):
        self.assertIn(
            "request_publication_approval",
            SUPERVISOR_SYSTEM_PROMPT,
        )
        self.assertIn("local_static_site", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("wechat_official_account", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("如果用户没有说明渠道", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("不能把这句话当成自动批准指令", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("不得绕过 HITL 审批直接发布", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("langchain-1-0-updates", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("retryable=false", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("不得使用完全相同的参数再次调用", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("Main 必须先调用 save_report", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("发布请求已经授权", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("询问后立即结束本轮", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("不得反复叙述", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("系统才会同时批准该 Article 版本", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("最多根据明确错误修正一次", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("repairable=true", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("publication_repair_exhausted", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("retryable=true", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("正文控制在 850 个字符以内", SUPERVISOR_SYSTEM_PROMPT)
        self.assertIn("正文（不含 Markdown 标记）不得超过 1000 个字符", SUPERVISOR_SYSTEM_PROMPT)

    def test_todo_prompt_covers_article_approval_boundary(self):
        from deep_research.prompts.todo import (
            TODO_SYSTEM_PROMPT,
            TODO_TOOL_DESCRIPTION,
        )

        self.assertIn("整理并保存 Article 草稿", TODO_SYSTEM_PROMPT)
        self.assertIn("等待用户人工审批", TODO_SYSTEM_PROMPT)
        self.assertIn(
            "Agent 不得把批准或发布伪装成已完成的 Todo",
            TODO_SYSTEM_PROMPT,
        )
        self.assertIn("正式 Markdown Artifact", TODO_SYSTEM_PROMPT)
        self.assertIn("Activity", TODO_SYSTEM_PROMPT)
        self.assertIn(
            "article_id、文件路径、数据库路径",
            TODO_TOOL_DESCRIPTION,
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
