from contextvars import ContextVar, Token

# 定义上下文变量,ContextVar 是 Python 3.7+ 提供的机制，用于在异步任务（asyncio）或线程之间安全地隔离上下文数据——每个协程/线程看到的是自己独立的值，互不干扰；
_request_id: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

_thread_id: ContextVar[str | None] = ContextVar(
    "thread_id",
    default=None,
)

# 读取函数
def get_request_id() -> str | None:
    return _request_id.get()


def get_thread_id() -> str | None:
    return _thread_id.get()

# 设置函数
def set_request_id(
    request_id: str | None,
) -> Token[str | None]:
    return _request_id.set(request_id)


def set_thread_id(
    thread_id: str | None,
) -> Token[str | None]:
    return _thread_id.set(thread_id)

# 重置函数
def reset_request_id(
    token: Token[str | None],
) -> None:
    _request_id.reset(token)

def reset_thread_id(
    token: Token[str | None],
) -> None:
    _thread_id.reset(token)