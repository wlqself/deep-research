import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from ..log.log_context import (
    get_thread_id,
    reset_request_id,
    reset_thread_id,
    set_request_id,
    set_thread_id,
)
from ..log.logging_utils import log_event


logger = logging.getLogger(
    "deep_research.http"
)

_REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)


def _request_id_from_scope(
    scope: dict[str, object],
) -> str:
    """
    ASGI 规范里，http/websocket 的 scope 都有一个 headers 字段，类型是：
    List[Tuple[bytes, bytes]]   # [(b"host", b"example.com"), (b"x-request-id", b"abc-123")]
    """
    headers = scope.get("headers", [])

    if isinstance(headers, list):
        # 遍历找目标 header
        for item in headers:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and item[0].lower() == b"x-request-id"
            ):
                # 值必须是 bytes，且能按 ASCII 解码
                raw_value = item[1]

                if not isinstance(raw_value, bytes):
                    break

                try:
                    value = raw_value.decode("ascii")
                except UnicodeDecodeError:
                    break

                # 正则全串校验
                if _REQUEST_ID_PATTERN.fullmatch(value):
                    return value

                break
    # 兜底：自己造一个
    return str(uuid4())

# 在 ASGI 协议层给每个 HTTP 请求发一张“身份证”（request_id），全程带着它跑，结束时打一条结构化访问日志，然后回收身份证。
class RequestContextMiddleware:
    """
    纯 ASGI middleware。

    使用纯 ASGI 而不是 BaseHTTPMiddleware，
    是为了让 ContextVar 覆盖整个 StreamingResponse 生成过程。
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
    ) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, object],
        receive,
        send,
    ) -> None:
        # 非 HTTP 请求直接透传
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        # 绑定 ContextVar
        request_id = _request_id_from_scope(scope)
        request_token = set_request_id(request_id)
        thread_token = set_thread_id(None)
        # 异常 / 取消 / 计时
        started_at = time.perf_counter()
        status_code: int | None = None
        cancelled = False
        fatal_error = False

        async def send_with_request_id(
            message: dict[str, object],
        ) -> None:
            nonlocal status_code
            # ASGI 里响应分两条消息：先 http.response.start（带 status + headers），再 N 条 http.response.body。
            if message.get("type") == "http.response.start":
                raw_status = message.get("status")
                
                if isinstance(raw_status, int):
                    status_code = raw_status

                raw_headers = message.get(
                    "headers",
                    [],
                )

                headers = (
                    list(raw_headers)
                    if isinstance(raw_headers, list)
                    else []
                )

                headers = [
                    (name, value)
                    for name, value in headers
                    if not (
                        isinstance(name, bytes)
                        and name.lower() == b"x-request-id"
                    )
                ]

                # 先删掉上游/应用可能已经塞的 x-request-id，再追加自己的。
                headers.append(
                    (
                        b"x-request-id",
                        request_id.encode("ascii"),
                    )
                )

                message = {
                    **message,
                    "headers": headers,
                }

            await send(message)

        try:
            await self.app(
                scope,
                receive,
                send_with_request_id,
            )
        except BaseException as error:
            if isinstance(
                error,
                (KeyboardInterrupt, SystemExit),
            ): # 进程要死了，别吞，别打“500 访问日志”
                fatal_error = True
                raise

            cancelled = isinstance(
                error,
                asyncio.CancelledError,
            )

            if status_code is None:
                status_code = (
                    499
                    if cancelled
                    else 500
                )

            raise
        finally:
            try:
                if not fatal_error:
                    if status_code is None:
                        status_code = 500

                    elapsed_ms = round(
                        (
                            time.perf_counter()
                            - started_at
                        )
                        * 1000,
                        3,
                    )

                    route = scope.get("route")
                    route_template = getattr(
                        route,
                        "path",
                        None,
                    )

                    if not isinstance(
                        route_template,
                        str,
                    ):
                        raw_path = scope.get("path", "")
                        route_template = (
                            raw_path
                            if isinstance(raw_path, str)
                            else ""
                        )

                    method = scope.get("method", "")
                    method = (
                        method
                        if isinstance(method, str)
                        else ""
                    )

                    if cancelled:
                        status = "cancelled"
                        level = logging.WARNING
                    elif status_code >= 500:
                        status = "failed"
                        level = logging.ERROR
                    else:
                        status = "completed"
                        level = logging.INFO

                    log_event(
                        logger,
                        level,
                        "http.request.completed",
                        request_id=request_id,
                        thread_id=get_thread_id(),
                        status=status,
                        elapsed_ms=elapsed_ms,
                        method=method,
                        route_template=route_template,
                        status_code=status_code,
                    )
            finally:
                reset_thread_id(thread_token)
                reset_request_id(request_token)