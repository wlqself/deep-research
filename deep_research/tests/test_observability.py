import asyncio
import unittest

from deep_research.agent.observability import LifecycleTracker, safe_model_error_code
from deep_research.handlers.research_stream import _stream_with_heartbeat


class ObservabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_token_is_emitted_once_and_done_has_counters(self):
        tracker = LifecycleTracker(thread_id="thread-1", run_id="run-1")
        self.assertIsNotNone(tracker.first_token())
        self.assertIsNone(tracker.first_token())
        tracker.heartbeat()
        done = tracker.done(goal_status="completed", error_codes=[])
        self.assertTrue(done["first_token_received"])
        self.assertEqual(done["heartbeat_count"], 1)
        self.assertEqual(done["goal_status"], "completed")

    async def test_error_codes_are_safe_and_stable(self):
        self.assertEqual(safe_model_error_code(TimeoutError()), "model_timeout")
        self.assertEqual(safe_model_error_code(ConnectionError()), "model_connection_failed")
        self.assertEqual(safe_model_error_code(ValueError()), "model_failed")
        self.assertNotIn("secret", str(safe_model_error_code(ValueError("secret"))))

    async def test_heartbeat_stream_stops_when_source_finishes(self):
        tracker = LifecycleTracker(thread_id="thread-1")

        async def source():
            yield {"type": "done", "goal_status": "completed"}

        events = [event async for event in _stream_with_heartbeat(source(), tracker)]
        self.assertEqual(events[0]["type"], "done")
        await asyncio.sleep(0)
        self.assertEqual(tracker.heartbeat_count, 0)


if __name__ == "__main__":
    unittest.main()
