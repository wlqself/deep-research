import unittest

from deep_research.agent.repetition_guard import StreamOutputGuard


class StreamOutputGuardTests(unittest.TestCase):
    def test_detects_three_repeated_long_blocks_across_chunks(self):
        guard = StreamOutputGuard(
            max_chars=10_000,
            repetition_min_chars=80,
            repetition_count=3,
        )
        block = "".join(
            f"segment-{index:03d}:unique-content;"
            for index in range(12)
        )

        self.assertIsNone(guard.add(block[:70]))
        self.assertIsNone(guard.add(block[70:] + block[:25]))
        self.assertIsNone(guard.add(block[25:] + block[:130]))
        self.assertEqual(
            guard.add(block[130:]),
            "model_repetition_detected",
        )

    def test_enforces_visible_output_limit(self):
        guard = StreamOutputGuard(
            max_chars=20,
            repetition_min_chars=10,
            repetition_count=3,
        )

        self.assertIsNone(guard.add("1234567890"))
        self.assertEqual(
            guard.add("12345678901"),
            "model_output_limit_exceeded",
        )


if __name__ == "__main__":
    unittest.main()
