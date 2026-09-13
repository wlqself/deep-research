import copy
import logging
import sys
from typing import Any
from logging.handlers import QueueHandler, QueueListener

from .log_context import get_request_id, get_thread_id
from .log_safety import (
    _EVENT_PATTERN,
    _exception_type,
    sanitize_traceback,
)


# 在日志系统自身出问题时发出告警。
def _listener_failure_alert() -> None:
    try:
        sys.stderr.write(
            "CRITICAL logging sink failed\n"
        )
        sys.stderr.flush() # 立即刷新缓冲区，强制把刚才那行字推送到 stderr 输出流里。
    except Exception:
        pass

class ContextFilter(logging.Filter):
    """
    在业务协程中把 ContextVar 快照写入 LogRecord。

    必须在 QueueHandler 入队之前执行，
    因为 QueueListener 后台线程无法读取原请求协程的 ContextVar。
    """
    # 日志上下文注入器
    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        # 如果业务代码打日志时已经手动传了 request_id，就不会被覆盖；没传就自动补上。
        if getattr(record, "request_id", None) is None:
            record.request_id = get_request_id()
        # 与上面同理
        if getattr(record, "thread_id", None) is None:
            record.thread_id = get_thread_id()

        return True

class SafeQueueHandler(QueueHandler):
    def __init__(
        self,
        queue: Any,
        *,
        include_traceback: bool = False,
    ) -> None:
        super().__init__(queue)
        self.include_traceback = include_traceback

    def prepare(
        self,
        record: logging.LogRecord,
    ) -> logging.LogRecord:
        # 浅拷贝
        prepared = copy.copy(record)

        event = getattr(record, "event", None)

        if (
            not isinstance(event, str)
            or not _EVENT_PATTERN.fullmatch(event)
        ):
            event = "legacy.log"

        #  提取异常类型
        exception_type = _exception_type(record)
        #  清理并统一字段
        prepared.event = event
        prepared.msg = event
        prepared.message = event
        prepared.args = ()
        prepared.exc_info = None
        prepared.exc_text = None
        prepared.stack_info = None

        if exception_type is not None:
            prepared.exception_type = exception_type

        prepared.safe_traceback = None
        
        # 只有开启了 include_traceback 且确实有异常时，才格式化 traceback
        if (
            self.include_traceback
            and isinstance(record.exc_info, tuple)
        ):
            try:
                raw_traceback = logging.Formatter().formatException(
                    record.exc_info
                )
                prepared.safe_traceback = sanitize_traceback(
                    raw_traceback
                )
            except Exception:
                prepared.safe_traceback = None

        return prepared
    
    def handleError(
        self,
        record: logging.LogRecord,
    ) -> None:
        try:
            sys.stderr.write(
                "CRITICAL business logging failed\n"
            )
            sys.stderr.flush()
        except Exception:
            pass


class SafeQueueListener(QueueListener):
    def handle(
        self,
        record: logging.LogRecord,
    ) -> None:
        # 重写父类 QueueListener 的 handle 方法。这个方法是后台线程从队列里取出一条日志（LogRecord）后调用的核心处理方法，每取一条就调用一次。
        for handler in self.handlers:
            if (
                self.respect_handler_level
                and record.levelno < handler.level # 如果日志级别比 handler 要求的级别低（比如 handler 设的 ERROR，但日志是 INFO），就 continue 跳过，不交给这个 handler 处理
            ):
                continue

            try:
                handler.handle(record)
            except Exception:
                _listener_failure_alert()