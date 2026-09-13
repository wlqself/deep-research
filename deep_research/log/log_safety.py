import math

import logging
import re

from datetime import datetime, timezone
from typing import Any

STRUCTURED_FIELDS = (
    "request_id",
    "thread_id",
    "task_id",
    "agent_name",
    "tool_name",
    "document_id",
    "memory_id",
    "artifact_id",
    "status",
    "elapsed_ms",
    "error_code",
    "exception_type",
    "method",
    "route_template",
    "status_code",
)

_EVENT_PATTERN = re.compile(
    r"^[a-z0-9]+(?:\.[a-z0-9_]+)+$"
)

_SAFE_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$"
)

_SAFE_CODE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9._:-]{0,63}$"
)

_SAFE_EXCEPTION_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*$"
)

_SAFE_ROUTE_PATTERN = re.compile(
    r"^/[A-Za-z0-9_./:{}-]{0,255}$"
)

_SAFE_METHOD_PATTERN = re.compile(
    r"^[A-Z]+$"
)

_TRACEBACK_FRAME_PATTERN = re.compile(
    r'^\s*File ".*?", line ([0-9]+), '
    r"in ([A-Za-z_][A-Za-z0-9_.]*)"
)

_EXCEPTION_LINE_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*:"
)

# 打上时间戳
def _utc_timestamp(created: float) -> str:
    return (
        datetime.fromtimestamp(
            created,
            tz=timezone.utc,
        )
        .isoformat()
        .replace("+00:00", "Z")
    )

# 把“异常类型”从一个三元组里抠成可结构化、可聚合、可进 ES 的一个短字符串，并且拿不到时安静返回 None，不抛、不猜、不降级成 "Unknown"。
def _exception_type(
    record: logging.LogRecord,
) -> str | None:
    exc_info = getattr(record, "exc_info", None)
    if not isinstance(exc_info, tuple) or len(exc_info) < 1:
        return None

    exception_class = exc_info[0] #  [0] 取的是类，不是实例，不是错误信息。

    if exception_class is None:
        return None

    return exception_class.__name__

def _sanitize_field(
    field_name: str,
    value: Any,
) -> Any:
    # 日志字段缺失是正常状态，null 在 JSON 里合法，不用折腾。
    if value is None:
        return None

    if field_name == "elapsed_ms":
        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            if math.isfinite(float(value)): # math.isfinite 挡 NaN / +inf / -inf
                return round(float(value), 3)

        return None

    if field_name == "status_code":
        if isinstance(value, bool):
            return None
        # 100–599 覆盖 HTTP 的 1xx–5xx，含 WebSocket 握手 101，也容下你们自定义的 499（客户端取消）。
        if isinstance(value, int) and 100 <= value <= 599:
            return value

        return None

    if not isinstance(value, str):
        return None

    value = value.strip()

    if field_name == "method":
        return (
            value
            if _SAFE_METHOD_PATTERN.fullmatch(value)
            else None
        )

    if field_name == "route_template":
        return (
            value
            if _SAFE_ROUTE_PATTERN.fullmatch(value)
            else None
        )

    if field_name == "error_code":
        return (
            value
            if _SAFE_CODE_PATTERN.fullmatch(value)
            else None
        )

    if field_name == "exception_type":
        return (
            value
            if _SAFE_EXCEPTION_PATTERN.fullmatch(value)
            else None
        )

    return (
        value
        if _SAFE_ID_PATTERN.fullmatch(value)
        else None
    )

def sanitize_traceback(
    traceback_text: str,
) -> str:
    """
    保留 traceback 的调用位置和异常类型，
    删除路径、源码行和异常消息。
    """
    safe_lines: list[str] = []

    # 不丢标题，但明确告诉看日志的人：这条堆栈已经被脱敏过，别拿它去本地复现时指望行号能对上源码文件。
    for line in traceback_text.splitlines():
        if line.startswith("Traceback"):
            safe_lines.append(
                "Traceback (sanitized)"
            )
            continue

        frame_match = _TRACEBACK_FRAME_PATTERN.match(line)

        if frame_match is not None:
            line_number, function_name = (
                frame_match.groups()
            )
            safe_lines.append(
                "  File \"<redacted>\", "
                f"line {line_number}, " # line_number 知道哪行出的问题
                f"in {function_name}"
            )
            continue

        # 末行异常行：只留异常类名，消息换 [REDACTED]
        exception_match = _EXCEPTION_LINE_PATTERN.match(
            line
        )

        if (
            exception_match is not None
            and not line.startswith(" ")
        ):
            exception_name = (
                exception_match.group(1)
            )

            if _SAFE_EXCEPTION_PATTERN.fullmatch(
                exception_name
            ):
                safe_lines.append(
                    f"{exception_name}: [REDACTED]"
                )

    return "\n".join(safe_lines)
