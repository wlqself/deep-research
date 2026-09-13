import asyncio
import unittest
from types import SimpleNamespace

from deep_research.agent.image_context import image_context_instruction
from deep_research.context import ResearchContext
from deep_research.state.access import context_for_thread
from deep_research.state.research import (
    ResearchState,
    merge_image_attachment_ids,
)


class ImageContextStateTests(unittest.TestCase):
    def test_image_ids_are_deduplicated_without_being_compressed(self):
        self.assertEqual(
            merge_image_attachment_ids(
                ["img-1", "img-1"],
                [" img-2 ", "img-1", ""],
            ),
            ["img-1", "img-2"],
        )

    def test_image_ids_are_declared_in_research_state(self):
        self.assertIn("image_attachment_ids", ResearchState.__annotations__)

    def test_context_recovers_historical_ids_from_checkpoint(self):
        class Agent:
            async def aget_state(self, _config):
                return SimpleNamespace(
                    values={"image_attachment_ids": ["img-1", "img-2"]}
                )

        context = asyncio.run(
            context_for_thread(
                Agent(),
                "thread-1",
                selected_attachment_ids=("img-3",),
            )
        )
        self.assertEqual(context.selected_attachment_ids, ("img-3",))
        self.assertEqual(
            context.conversation_attachment_ids,
            ("img-1", "img-2"),
        )

    def test_instruction_distinguishes_current_and_historical_images(self):
        context = ResearchContext.from_settings(
            selected_attachment_ids=("img-3",),
            conversation_attachment_ids=("img-1", "img-3"),
        )
        instruction = image_context_instruction(context)
        self.assertIn("img-3", instruction)
        self.assertIn("img-1", instruction)
        self.assertIn('mode="cached"', instruction)
        self.assertIn('mode="fresh"', instruction)


if __name__ == "__main__":
    unittest.main()
