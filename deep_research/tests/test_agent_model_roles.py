import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

import deep_research.agent.factory as factory_module
from deep_research.agent.model import model, researcher_model
from deep_research.config import settings


class _BindableFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentModelRoleTests(unittest.TestCase):
    def test_main_agent_disables_thinking(self):
        self.assertEqual(model.model_name, settings.model_name)
        self.assertEqual(
            model.extra_body,
            {"enable_thinking": False},
        )

    def test_researcher_keeps_thinking_enabled(self):
        self.assertEqual(researcher_model.model_name, settings.model_name)
        self.assertEqual(
            researcher_model.extra_body,
            {"enable_thinking": True},
        )

    def test_patched_main_model_is_reused_by_researcher_in_tests(self):
        fake_model = _BindableFakeChatModel(messages=iter([]))

        with (
            patch.object(factory_module, "model", fake_model),
            patch.object(
                factory_module,
                "build_researcher",
                wraps=factory_module.build_researcher,
            ) as build_researcher,
        ):
            factory_module.build_agent(InMemorySaver())

        self.assertIs(
            build_researcher.call_args.args[0],
            fake_model,
        )


if __name__ == "__main__":
    unittest.main()
