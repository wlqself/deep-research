import json
import logging
from typing import Any

from .log_safety import (
    STRUCTURED_FIELDS,
    _exception_type,
    _sanitize_field,
    _utc_timestamp,
    sanitize_traceback,
)

class JsonFormatter(logging.Formatter):
    """
    将 LogRecord 转换成一行 JSON。

    """

    def __init__(
        self,
        *,
        include_traceback: bool = False,
    ) -> None:
        super().__init__()
        self.include_traceback = include_traceback

    # 接收一条日志记录 LogRecord，返回一个 JSON 字符串。这是 logging 模块在输出日志时调用的核心格式化方法。
    """
    取事件名：从 LogRecord 里拿 event 字段，没有就降级为 "legacy.log"；
    组装基础信息：把时间（UTC）、级别、logger 名、事件名放进字典；
    填充结构化字段：遍历预定义字段列表，从记录里取值、清洗、填入；
    处理异常信息：提取异常类型名，有异常但没错误码就给默认值；
    处理堆栈：如果开启了 traceback 且确实有异常，格式化并清洗后加入；
    序列化为紧凑 JSON：转成可读、体积小、容错强的 JSON 字符串返回给 handler 去写文件/控制台。
    """
    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        # 提取 event（事件名），提取不到就降级；兼容新旧两种打日志的方式，下游按 event 字段做告警/聚合时不会缺字段
        event = getattr(
            record,
            "event",
            None,
        )

        if not isinstance(event, str) or not event:
            event = "legacy.log"

        # 组装基础字段
        payload: dict[str, Any] = {
            "timestamp": _utc_timestamp(
                record.created
            ),
            "level": record.levelname, # "DEBUG" / "INFO" / "ERROR" 这种字符串，而不是数字级别。
            "logger": record.name, # logger 的名字，一般就是 __name__（如 app.services.payment）。
            "event": event,
        }

        # 从 record 里取对应的值，用 _sanitize_field 做清洗/校验（比如脱敏、类型检查），然后塞进 payload。
        for field_name in STRUCTURED_FIELDS:
            payload[field_name] = _sanitize_field(
                field_name,
                getattr(
                    record,
                    field_name,
                    None,
                ),
            )

        # 日志收集系统可以按 exception_type 聚合错误类型，而不必去正则解析 traceback。
        payload["exception_type"] = (
            _exception_type(record)
            or payload["exception_type"]
        )

        # 处理之前加入默认安全错误码
        if (
            payload["exception_type"] is not None
            and payload["error_code"] is None
        ):
            payload["error_code"] = (
                "unclassified_error"
            )

        # 只有真有异常时才加 traceback 字段，正常 INFO 日志里不会出现这个大字段
        if self.include_traceback:
            safe_traceback = getattr(
                record,
                "safe_traceback",
                None,
            )

            if isinstance(safe_traceback, str):
                payload["traceback"] = safe_traceback
            elif isinstance(record.exc_info, tuple):
                raw_traceback = self.formatException(
                    record.exc_info
                )
                payload["traceback"] = sanitize_traceback(
                    raw_traceback
                )

        return json.dumps(
            payload,
            ensure_ascii=False, # 中文不转义成 \uXXXX，日志可读
            separators=(",", ":"), # 去掉空格，体积更小：{"a":1,"b":2}
            default=str, # 遇到 Decimal/自定义对象等非 JSON 类型，兜底转字符串
        )