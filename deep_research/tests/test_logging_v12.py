import asyncio
import io
import json
import logging
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from deep_research.config import (
    configure_logging,
    settings,
    shutdown_logging,
)
from deep_research.handlers import research as research_handler
from deep_research.log.log_audit import (
    AuditFormatter,
    AuditQueueListener,
    audit_event,
)
from deep_research.log.log_context import (
    reset_request_id,
    reset_thread_id,
    set_request_id,
    set_thread_id,
)
from deep_research.log.logging_utils import (
    ContextFilter,
    JsonFormatter,
    SafeQueueHandler,
    SafeQueueListener,
    STRUCTURED_FIELDS,
    log_event,
)
from deep_research.middleware.request_context import (
    RequestContextMiddleware,
)


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _FailingLogger:
    def log(self, *args: object, **kwargs: object) -> None:
        raise OSError("logging backend unavailable")


class _FailingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        raise OSError("audit file unavailable")


class _LoggerStateIsolationMixin:
    _LOGGER_NAMES = (
        "deep_research",
        "deep_research.audit",
        "deep_research.chain",
    )

    def setUp(self) -> None:
        super().setUp()
        self._logger_snapshots: dict[str, dict[str, object]] = {}
        for logger_name in self._LOGGER_NAMES:
            logger = logging.getLogger(logger_name)
            managed = {
                key: value
                for key, value in vars(logger).items()
                if key.startswith("_deep_research")
            }
            self._logger_snapshots[logger_name] = {
                "handlers": list(logger.handlers),
                "filters": list(logger.filters),
                "level": logger.level,
                "propagate": logger.propagate,
                "disabled": logger.disabled,
                "managed": managed,
            }
        self.addCleanup(self._restore_logger_state)

    def _restore_logger_state(self) -> None:
        # Stop only listeners created or replaced by this test.  This keeps
        # cleanup safe even when a caller had configured logging beforehand.
        for logger_name, snapshot in self._logger_snapshots.items():
            logger = logging.getLogger(logger_name)
            before = snapshot["managed"]
            current = {
                key: value
                for key, value in vars(logger).items()
                if key.startswith("_deep_research")
            }
            for attr_name, state in current.items():
                if (
                    attr_name in before
                    and state is before[attr_name]
                ) or not isinstance(state, dict):
                    continue
                listener = state.get("listener")
                if listener is not None:
                    try:
                        listener.stop()
                    except Exception:
                        pass
                queue_handler = state.get("queue_handler")
                if queue_handler is not None:
                    try:
                        logger.removeHandler(queue_handler)
                    except Exception:
                        pass
                for sink_key in ("sink_handlers", "file_handler"):
                    sinks = state.get(sink_key, ())
                    if sink_key == "file_handler":
                        sinks = (sinks,)
                    for sink in sinks or ():
                        try:
                            sink.close()
                        except Exception:
                            pass
                try:
                    delattr(logger, attr_name)
                except Exception:
                    pass

        for logger_name, snapshot in self._logger_snapshots.items():
            logger = logging.getLogger(logger_name)
            logger.handlers = list(snapshot["handlers"])
            logger.filters = list(snapshot["filters"])
            logger.setLevel(snapshot["level"])
            logger.propagate = bool(snapshot["propagate"])
            logger.disabled = bool(snapshot["disabled"])

            managed = snapshot["managed"]
            for key in list(vars(logger)):
                if key.startswith("_deep_research") and key not in managed:
                    delattr(logger, key)
            for key, value in managed.items():
                setattr(logger, key, value)


class _LoggerCaptureMixin(_LoggerStateIsolationMixin):
    def capture_logger(
        self,
        logger_name: str,
        *,
        add_context_filter: bool = False,
    ) -> _RecordHandler:
        logger = logging.getLogger(logger_name)
        previous_handlers = list(logger.handlers)
        previous_level = logger.level
        previous_propagate = logger.propagate
        previous_disabled = logger.disabled

        handler = _RecordHandler()

        if add_context_filter:
            handler.addFilter(ContextFilter())

        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.disabled = False

        def restore() -> None:
            logger.handlers = previous_handlers
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate
            logger.disabled = previous_disabled

        self.addCleanup(restore)
        return handler


class JsonLoggingTests(_LoggerCaptureMixin, unittest.TestCase):
    def test_json_schema_is_stable_and_hides_message_content(self) -> None:
        secret = "sk-test-secret"
        record = logging.LogRecord(
            name="deep_research.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=f"question={secret}",
            args=(),
            exc_info=None,
        )
        record.event = "research.completed"
        record.request_id = "request-1"
        record.thread_id = "thread-1"
        record.task_id = "task-1"
        record.agent_name = "researcher"
        record.tool_name = "web_search"
        record.document_id = "document-1"
        record.memory_id = "memory-1"
        record.artifact_id = "artifact-1"
        record.status = "completed"
        record.elapsed_ms = 12.3456
        record.error_code = None
        record.exception_type = None
        record.method = "POST"
        record.route_template = "/research"
        record.status_code = 200

        payload = json.loads(JsonFormatter().format(record))

        self.assertEqual(
            set(payload),
            {
                "timestamp",
                "level",
                "logger",
                "event",
                *STRUCTURED_FIELDS,
            },
        )
        self.assertEqual(payload["request_id"], "request-1")
        self.assertEqual(payload["elapsed_ms"], 12.346)
        self.assertNotIn(secret, json.dumps(payload))
        self.assertNotIn("message", payload)

    def test_traceback_keeps_no_secret_or_exception_message(self) -> None:
        secret = "Authorization: Bearer secret-token"

        try:
            raise ValueError(secret)
        except ValueError:
            record = logging.LogRecord(
                name="deep_research.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="unexpected failure",
                args=(),
                exc_info=sys.exc_info(),
            )

        record.event = "research.failed"
        payload = json.loads(
            JsonFormatter(
                include_traceback=True
            ).format(record)
        )

        self.assertEqual(payload["exception_type"], "ValueError")
        self.assertEqual(
            payload["error_code"],
            "unclassified_error",
        )
        self.assertNotIn(secret, payload["traceback"])
        self.assertIn("[REDACTED]", payload["traceback"])

    def test_log_event_failure_does_not_raise(self) -> None:
        with patch("sys.stderr", io.StringIO()) as stderr:
            log_event(
                _FailingLogger(),
                logging.INFO,
                "research.completed",
                status="completed",
            )
        self.assertEqual(
            stderr.getvalue(),
            "CRITICAL business logging failed\n",
        )

    def test_task_and_tool_ids_are_preserved(self) -> None:
        handler = self.capture_logger(
            "deep_research.test.association",
            add_context_filter=True,
        )
        logger = logging.getLogger(
            "deep_research.test.association"
        )
        request_token = set_request_id("request-2")
        thread_token = set_thread_id("thread-2")

        try:
            log_event(
                logger,
                logging.INFO,
                "tool.completed",
                task_id="task-2",
                tool_name="save_report",
                document_id="document-2",
                memory_id="memory-2",
                artifact_id="artifact-2",
                status="completed",
            )
        finally:
            reset_thread_id(thread_token)
            reset_request_id(request_token)

        payload = json.loads(JsonFormatter().format(handler.records[0]))

        self.assertEqual(payload["request_id"], "request-2")
        self.assertEqual(payload["thread_id"], "thread-2")
        self.assertEqual(payload["task_id"], "task-2")
        self.assertEqual(payload["tool_name"], "save_report")
        self.assertEqual(payload["document_id"], "document-2")
        self.assertEqual(payload["memory_id"], "memory-2")
        self.assertEqual(payload["artifact_id"], "artifact-2")


class SafeQueueChainTests(_LoggerStateIsolationMixin, unittest.TestCase):
    def test_handler_queue_listener_formatter_chain_is_safe_and_reusable(self) -> None:
        logger = logging.getLogger("deep_research.chain")
        logger.handlers = []
        logger.filters = []
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.disabled = False

        records: list[logging.LogRecord] = []
        collecting_sink = _RecordHandler()
        failing_sink = _FailingHandler()
        log_queue: queue.Queue[logging.LogRecord] = queue.Queue()
        queue_handler = SafeQueueHandler(
            log_queue,
            include_traceback=True,
        )
        queue_handler.addFilter(ContextFilter())
        logger.addHandler(queue_handler)

        listener = SafeQueueListener(
            log_queue,
            failing_sink,
            collecting_sink,
            respect_handler_level=True,
        )
        listener.start()

        secret = "sk-chain-secret"
        with patch("sys.stderr", io.StringIO()) as stderr:
            try:
                try:
                    raise ValueError(f"Authorization: Bearer {secret}")
                except ValueError:
                    logger.error(
                        f"question={secret}",
                        extra={"event": "research.failed"},
                        exc_info=sys.exc_info(),
                    )
                logger.info(
                    f"follow-up={secret}",
                    extra={"event": "research.completed"},
                )
            finally:
                listener.stop()
            self.assertNotIn(secret, stderr.getvalue())

        records.extend(collecting_sink.records)
        self.assertEqual(len(records), 2)
        self.assertEqual(
            [record.event for record in records],
            ["research.failed", "research.completed"],
        )
        for record in records:
            self.assertNotIn(secret, repr(record.__dict__))

        payload = json.loads(
            JsonFormatter(include_traceback=True).format(records[0])
        )
        self.assertNotIn(secret, json.dumps(payload))
        self.assertNotIn(secret, payload.get("traceback", ""))
        self.assertIsNone(getattr(records[0], "exc_info", None))
        self.assertFalse(listener._thread and listener._thread.is_alive())


class ContextIsolationTests(_LoggerStateIsolationMixin, unittest.IsolatedAsyncioTestCase):
    async def test_contextvars_are_isolated_between_tasks(self) -> None:
        async def worker(
            request_id: str,
            thread_id: str,
        ) -> tuple[str | None, str | None]:
            request_token = set_request_id(request_id)
            thread_token = set_thread_id(thread_id)

            try:
                await asyncio.sleep(0)
                record = logging.LogRecord(
                    name="deep_research.test",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=1,
                    msg="event",
                    args=(),
                    exc_info=None,
                )
                ContextFilter().filter(record)
                return record.request_id, record.thread_id
            finally:
                reset_thread_id(thread_token)
                reset_request_id(request_token)

        first, second = await asyncio.gather(
            worker("request-a", "thread-a"),
            worker("request-b", "thread-b"),
        )

        self.assertEqual(first, ("request-a", "thread-a"))
        self.assertEqual(second, ("request-b", "thread-b"))


class RequestContextMiddlewareTests(_LoggerStateIsolationMixin, unittest.IsolatedAsyncioTestCase):
    async def _invoke(
        self,
        headers: list[tuple[bytes, bytes]],
    ) -> str:
        async def app(scope, receive, send) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"ok",
                    "more_body": False,
                }
            )

        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return {"type": "http.request"}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        middleware = RequestContextMiddleware(app)
        await middleware(
            {
                "type": "http",
                "method": "GET",
                "path": "/health",
                "headers": headers,
            },
            receive,
            send,
        )

        start = sent[0]
        response_headers = start["headers"]
        self.assertIsInstance(response_headers, list)

        for name, value in response_headers:
            if name.lower() == b"x-request-id":
                return value.decode("ascii")

        self.fail("response did not include X-Request-ID")

    async def test_request_id_passthrough_and_replacement(self) -> None:
        valid = await self._invoke(
            [(b"x-request-id", b"request-123")]
        )
        invalid = await self._invoke(
            [(b"x-request-id", b"not valid!")]
        )
        too_long = await self._invoke(
            [(b"x-request-id", b"a" * 129)]
        )

        self.assertEqual(valid, "request-123")
        self.assertNotEqual(invalid, "not valid!")
        self.assertNotEqual(too_long, "a" * 129)
        UUID(invalid)
        UUID(too_long)


class AuditAndConfigurationTests(_LoggerCaptureMixin, unittest.TestCase):
    def test_audit_formatter_is_separate_and_content_free(self) -> None:
        handler = self.capture_logger("deep_research.audit")
        request_token = set_request_id("request-audit")

        try:
            audit_event(
                "document.upload",
                "completed",
                document_id="document-audit",
            )
        finally:
            reset_request_id(request_token)

        payload = json.loads(
            AuditFormatter().format(handler.records[0])
        )

        self.assertEqual(
            set(payload),
            {
                "timestamp",
                "request_id",
                "operation",
                "result",
                "thread_id",
                "document_id",
                "memory_id",
                "artifact_id",
            },
        )
        self.assertEqual(payload["request_id"], "request-audit")
        self.assertEqual(payload["operation"], "document.upload")
        self.assertEqual(payload["document_id"], "document-audit")
        self.assertNotIn("message", payload)

    def test_audit_failure_does_not_raise(self) -> None:
        logger = logging.getLogger("deep_research.audit")

        with patch("sys.stderr", io.StringIO()) as stderr:
            with patch.object(
                logger,
                "info",
                side_effect=OSError("audit file unavailable"),
            ):
                audit_event(
                    "thread.delete",
                    "failed",
                    thread_id="thread-audit",
                )

        self.assertIn(
            "CRITICAL audit logging failed",
            stderr.getvalue(),
        )

    def test_audit_listener_failure_emits_security_alert(self) -> None:
        listener = AuditQueueListener(
            queue.Queue(),
            _FailingHandler(),
        )
        record = logging.LogRecord(
            name="deep_research.audit",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="document.upload",
            args=(),
            exc_info=None,
        )

        with patch("sys.stderr", io.StringIO()) as stderr:
            listener.handle(record)

        self.assertIn(
            "CRITICAL audit logging failed",
            stderr.getvalue(),
        )

    def test_configuration_is_idempotent_and_separates_files(self) -> None:
        shutdown_logging()

        original_values = {
            "log_dir": settings.log_dir,
            "log_file": settings.log_file,
            "log_audit_file": settings.log_audit_file,
        }

        with tempfile.TemporaryDirectory() as directory:
            settings.log_dir = directory
            settings.log_file = "application.log"
            settings.log_audit_file = "audit.log"

            try:
                with patch("sys.stderr", io.StringIO()):
                    configure_logging()
                    app_logger = logging.getLogger("deep_research")
                    audit_logger = logging.getLogger(
                        "deep_research.audit"
                    )
                    first_app_handler = app_logger.handlers[0]
                    first_audit_handler = audit_logger.handlers[0]

                    configure_logging()

                    self.assertIs(
                        app_logger.handlers[0],
                        first_app_handler,
                    )
                    self.assertIs(
                        audit_logger.handlers[0], first_audit_handler)

                    log_event(
                        app_logger,
                        logging.INFO,
                        "research.completed",
                        status="completed",
                    )
                    audit_event(
                        "document.upload",
                        "completed",
                        document_id="document-config",
                    )

                    shutdown_logging()

                application_log = Path(directory) / "application.log"
                audit_log = Path(directory) / "audit.log"
                application_text = application_log.read_text(
                    encoding="utf-8"
                )
                audit_text = audit_log.read_text(encoding="utf-8")

                self.assertIn("research.completed", application_text)
                self.assertNotIn("document.upload", application_text)
                self.assertIn("document.upload", audit_text)
                self.assertNotIn("research.completed", audit_text)
            finally:
                settings.log_dir = original_values["log_dir"]
                settings.log_file = original_values["log_file"]
                settings.log_audit_file = (
                    original_values["log_audit_file"]
                )
                shutdown_logging()
                configure_logging()


class StreamLifecycleLoggingTests(_LoggerCaptureMixin, unittest.IsolatedAsyncioTestCase):
    async def test_stream_completion_failure_and_cancellation_log_terminals(
        self,
    ) -> None:
        handler = self.capture_logger(
            "deep_research.handlers.research"
        )

        async def completed_events(question: str, thread_id: str):
            yield {"type": "done"}

        async def failed_events(question: str, thread_id: str):
            raise RuntimeError("provider response contains secret")
            yield  # pragma: no cover

        async def cancelled_events(question: str, thread_id: str):
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        with patch(
            "deep_research.handlers.research.stream_research_events",
            completed_events,
        ):
            _ = [
                chunk
                async for chunk in research_handler._stream_answer(
                    "question must not be logged",
                    "stream-thread",
                )
            ]

        with patch(
            "deep_research.handlers.research.stream_research_events",
            failed_events,
        ):
            with self.assertRaises(RuntimeError):
                _ = [
                    chunk
                    async for chunk in research_handler._stream_answer(
                        "question must not be logged",
                        "stream-thread",
                    )
                ]

        with patch(
            "deep_research.handlers.research.stream_research_events",
            cancelled_events,
        ):
            with self.assertRaises(asyncio.CancelledError):
                _ = [
                    chunk
                    async for chunk in research_handler._stream_answer(
                        "question must not be logged",
                        "stream-thread",
                    )
                ]

        events = [
            getattr(record, "event", None)
            for record in handler.records
        ]

        self.assertIn("research.completed", events)
        self.assertIn("research.failed", events)
        self.assertIn("research.cancelled", events)


if __name__ == "__main__":
    unittest.main()
