import json
import logging
import sys

from .log_context import get_request_id
from .log_handlers import SafeQueueListener
from .log_safety import _sanitize_field, _utc_timestamp

# 在审计日志系统自身出问题时发出告警。
def _security_alert() -> None:
    try:
        sys.stderr.write("CRITICAL audit logging failed\n")
        sys.stderr.flush() # 立即刷新缓冲区，强制把刚才那行字推送到 stderr。防止缓冲区没满导致信息迟迟不输出
    except Exception:
        pass


class AuditQueueListener(SafeQueueListener):
    # 后台线程从队列里取出一条日志（LogRecord）后调用的核心处理方法，每取一条就调用一次。
    """
    取日志：QueueListener 的后台线程从内存队列里取出一条 LogRecord。
    调 handle：线程调用 self.handle(record)，也就是上面这个重写的方法。
    遍历分发：逐个遍历所有 sink handler。
    级别检查：如果开启了 respect_handler_level，就检查这条日志够不够级别，不够就跳过。
    尝试写入：级别够了，就调用 handler.handle(record) 真正写文件/写控制台。
    异常兜底：如果写入失败，捕获异常，调用告警函数，继续处理下一个 handler 或下一条日志，线程不死。
    """
    def handle(
        self,
        record: logging.LogRecord,
    ) -> None:
        for handler in self.handlers:
            if (
                self.respect_handler_level
                and record.levelno < handler.level
            ):
                continue

            try:
                handler.handle(record) # 调用 handler 的 handle 方法，真正执行写入操作
            except Exception:
                try:
                    sys.stderr.write(
                        "CRITICAL audit logging failed\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass


class AuditFormatter(logging.Formatter):
    """
    审计日志只保留：
    时间、request_id、操作、结果和内部 ID。
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        
        payload = {
            "timestamp": _utc_timestamp(
                record.created
            ),
            "request_id": _sanitize_field(
                "request_id",
                getattr(
                    record,
                    "request_id",
                    None,
                ),
            ),
            "operation": _sanitize_field(
                "error_code",
                getattr(
                    record,
                    "operation",
                    None,
                ),
            ),
            "result": _sanitize_field(
                "error_code",
                getattr(
                    record,
                    "result",
                    None,
                ),
            ),
            "thread_id": _sanitize_field(
                "thread_id",
                getattr(
                    record,
                    "thread_id",
                    None,
                ),
            ),
            "document_id": _sanitize_field(
                "document_id",
                getattr(
                    record,
                    "document_id",
                    None,
                ),
            ),
            "memory_id": _sanitize_field(
                "memory_id",
                getattr(
                    record,
                    "memory_id",
                    None,
                ),
            ),
            "artifact_id": _sanitize_field(
                "artifact_id",
                getattr(
                    record,
                    "artifact_id",
                    None,
                ),
            ),
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    
# 记录审计事件
def audit_event(
    operation: str,
    result: str,
    *,
    request_id: str | None = None,
    thread_id: str | None = None,
    document_id: str | None = None,
    memory_id: str | None = None,
    artifact_id: str | None = None,
) -> None:
    """
    写入独立 audit logger。

    审计失败不能影响业务操作。
    """
    audit_logger = logging.getLogger(
        "deep_research.audit"
    )
    
    # 组装一个 fields 字典，收集所有审计字段
    fields = {
        "operation": operation,
        "result": result,
        "request_id": (
            request_id
            if request_id is not None
            else get_request_id()
        ),
        "thread_id": thread_id,
        "document_id": document_id,
        "memory_id": memory_id,
        "artifact_id": artifact_id,
    }

    try:
        audit_logger.info(
            operation,
            extra=fields,
        )
    except Exception:
        _security_alert()
