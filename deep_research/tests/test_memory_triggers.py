import unittest

from pydantic import ValidationError

from deep_research.memory.triggers import (
    MemoryTriggerInput,
    decide_memory_trigger,
    has_confirmed_project_decision,
    has_explicit_correction,
    has_memory_rule_signal,
)


def make_input(**overrides: object) -> MemoryTriggerInput:
    data: dict[str, object] = {
        "success": True,
        "answer": "A successful answer.",
        "successful_turn_count": 1,
        "review_interval": 10,
    }
    data.update(overrides)
    return MemoryTriggerInput.model_validate(data)


class MemoryTriggerTests(unittest.TestCase):
    def test_successful_turn_without_signal_does_not_trigger(self):
        decision = decide_memory_trigger(make_input())

        self.assertFalse(decision.should_review)
        self.assertEqual(decision.reasons, [])

    def test_each_explicit_signal_triggers_review(self):
        for field, reason in (
            ("rule_triggered", "rule"),
            ("summary_changed", "summary_changed"),
            ("report_saved", "report_saved"),
            (
                "project_decision_confirmed",
                "project_decision_confirmed",
            ),
            ("explicit_correction", "explicit_correction"),
        ):
            decision = decide_memory_trigger(
                make_input(**{field: True})
            )

            self.assertTrue(decision.should_review)
            self.assertEqual(decision.reasons, [reason])

    def test_multiple_signals_are_merged_once_in_stable_order(self):
        decision = decide_memory_trigger(
            make_input(
                rule_triggered=True,
                summary_changed=True,
                report_saved=True,
                project_decision_confirmed=True,
                explicit_correction=True,
                successful_turn_count=10,
            )
        )

        self.assertEqual(
            decision.reasons,
            [
                "rule",
                "summary_changed",
                "report_saved",
                "project_decision_confirmed",
                "explicit_correction",
                "interval",
            ],
        )

    def test_failed_or_empty_answer_never_triggers(self):
        for overrides in (
            {
                "success": False,
                "rule_triggered": True,
                "summary_changed": True,
            },
            {
                "answer": "   ",
                "report_saved": True,
                "explicit_correction": True,
                "successful_turn_count": 10,
            },
        ):
            decision = decide_memory_trigger(make_input(**overrides))

            self.assertFalse(decision.should_review)
            self.assertEqual(decision.reasons, [])

    def test_interval_triggers_only_on_positive_multiple(self):
        self.assertFalse(
            decide_memory_trigger(
                make_input(successful_turn_count=9)
            ).should_review
        )
        self.assertTrue(
            decide_memory_trigger(
                make_input(successful_turn_count=10)
            ).should_review
        )
        self.assertTrue(
            decide_memory_trigger(
                make_input(successful_turn_count=20)
            ).should_review
        )

    def test_trigger_configuration_rejects_invalid_counts(self):
        with self.assertRaises(ValidationError):
            make_input(successful_turn_count=-1)

        with self.assertRaises(ValidationError):
            make_input(review_interval=0)

    def test_explicit_correction_rule_accepts_strong_correction_markers(self):
        for text in (
            "刚才不对，应该使用 Main 的记忆工具。",
            "请更正为项目级约束。",
            "That is wrong; it should be Main-only.",
            "Correction: change it to the project scope.",
        ):
            with self.subTest(text=text):
                self.assertTrue(has_explicit_correction(text))

    def test_explicit_correction_rule_rejects_preferences_and_hypotheticals(self):
        for text in (
            "以后回答简洁一点。",
            "如果不对请告诉我。",
            "",
            "   ",
        ):
            with self.subTest(text=text):
                self.assertFalse(has_explicit_correction(text))

    def test_project_decision_rule_accepts_explicit_decisions(self):
        for text in (
            "确认采用这个方案，按 Main-only 设计实现。",
            "项目架构就按这个定了。",
            "We confirm this architecture.",
        ):
            with self.subTest(text=text):
                self.assertTrue(has_confirmed_project_decision(text))

    def test_project_decision_rule_rejects_acknowledgements_and_unrelated_text(self):
        for text in (
            "确认收到方案。",
            "请确认会议时间。",
            "这个项目很重要。",
            "以后回答简单一点。",
        ):
            with self.subTest(text=text):
                self.assertFalse(has_confirmed_project_decision(text))

    def test_memory_rule_signal_accepts_durable_preferences_and_constraints(self):
        for text in (
            "请记住以后回答简洁一点。",
            "我偏好使用简洁的代码。",
            "以后请默认使用 Main-only 设计。",
            "不要再把记忆传给 Researcher。",
            "Please remember my preference.",
        ):
            with self.subTest(text=text):
                self.assertTrue(has_memory_rule_signal(text))

    def test_memory_rule_signal_rejects_ordinary_temporary_and_hypothetical_text(self):
        for text in (
            "什么是长期记忆？",
            "请告诉我这个方案的优缺点。",
            "这次回答简单一点。",
            "如果以后需要再处理。",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(has_memory_rule_signal(text))


if __name__ == "__main__":
    unittest.main()
