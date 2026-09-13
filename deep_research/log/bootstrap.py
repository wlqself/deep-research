import atexit
import logging
import logging.config
import queue
import threading
from pathlib import Path
from logging.handlers import (
    RotatingFileHandler,
)
from ..settings import settings

from .log_handlers import (
    ContextFilter,
    SafeQueueHandler,
    SafeQueueListener,
)

from .log_audit import (
    AuditFormatter,
    AuditQueueListener,
)

"""
业务协程：logger.info("forget ok", extra={...})
   │
   ├─ logging 判断 level（logger.level=INFO，record 也是 INFO → 通过）
   ├─ 找 handlers → 只有 QueueHandler
   └─ QueueHandler.emit(record)
        └─ queue.put_nowait(record)      ← 内存队列，~1μs，协程不阻塞
业务协程继续干别的 ✅

─── 后台线程（QueueListener 起的 daemon thread）───
   while 没收到 stop：
     record = queue.get()
     for h in (console_sink, file_sink):
         h.emit(record)      ← 格式化字符串、写 stderr、写文件、检查 maxBytes
                              若超了 → close → rename app.log → app.log.1 → 新建
  收到 stop 且队列空 → 线程退出

进程退出：
   atexit → shutdown_logging()
     removeHandler → listener.stop() → sleep 排空 → close 文件 fd
"""
_LOG_LOCK = threading.RLock()
_LOG_STATE_ATTR = "_deep_research_logging_state"
_AUDIT_STATE_ATTR = "_deep_research_audit_state"
_ATEXIT_REGISTERED = False

# 程序退出或重载配置时，用它来安全地停掉后台监听线程，既确保剩余日志写完，又不会因为重复停止或状态异常而导致程序报错崩溃。
def _stop_listener(listener: object) -> None:
    thread = getattr(listener, "_thread", None)
    if thread is None:
        return
    try:
        listener.stop() # 通知后台线程“该收工了”，并且通常会等待队列里剩余的日志处理完再退出，确保不丢数据。
    except (AttributeError, RuntimeError):
        pass

# 判断日志的等级，看是否写入日志
def _logging_level_name() -> str:
    level_name = settings.log_level.strip().upper()

    if not isinstance(
        getattr(logging, level_name, None),
        int, # 判断取到的值是不是一个整数。因为 logging.DEBUG、logging.INFO 等标准级别本质上都是整数常量。
    ):
        raise ValueError(
            f"invalid log level: {settings.log_level}"
        )

    return level_name


def configure_logging() -> None:
    """
    初始化应用日志。

    业务 logger 只写入 QueueHandler；
    控制台和文件由 QueueListener 后台线程写入。
    """
    app_logger = logging.getLogger("deep_research")

    # 幂等守卫
    global _ATEXIT_REGISTERED
    with _LOG_LOCK:
        existing_state = getattr(
            app_logger,
            _LOG_STATE_ATTR,
            None,
        )

        if existing_state is not None:
            return

        # 算级别 + 建目录
        level_name = _logging_level_name()

        if settings.log_format.strip().lower() != "json":
            raise ValueError(
                "only json log format is supported"
            )

        log_path = Path(settings.log_dir) / settings.log_file
        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_queue: queue.Queue[
            logging.LogRecord
        ] = queue.Queue()

        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False, # 把已存在的所有 logger（除 root 外）的 handler 全清空、级别设为 NOTSET，关闭表示不进行
                "formatters": {
                    "json_console": {
                        "()": (
                            "deep_research.log.log_formatters."
                            "JsonFormatter"
                        ),
                        "include_traceback": (
                            settings.log_include_traceback
                        ),
                    },
                    "json_file": {
                        "()": (
                            "deep_research.log.log_formatters."
                            "JsonFormatter"
                        ),
                        "include_traceback": (
                            settings.log_include_traceback
                        ),
                    },
                },
                # 此刻这两个 sink handler 是“活的、但只是临时挂在 logger 上
                "handlers": {
                    "console_sink": {
                        "class": (
                            "logging.StreamHandler"
                        ),
                        "level": level_name,
                        "formatter": "json_console",
                        "stream": "ext://sys.stderr",
                    },
                    "file_sink": {
                        "class": (
                            "logging.handlers."
                            "RotatingFileHandler"
                        ),
                        "level": level_name,
                        "formatter": "json_file",
                        "filename": str(log_path),
                        "maxBytes": settings.log_max_bytes,
                        "backupCount": (
                            settings.log_backup_count
                        ),
                        "encoding": "utf-8",
                    },
                },
                "loggers": {
                    "deep_research": {
                        "handlers": [
                            "console_sink",
                            "file_sink",
                        ],
                        "level": level_name,
                        "propagate": False,
                    },
                },
            }
        )
        """
        为什么不能直接让 logger 同时挂着 console_sink + file_sink + QueueHandler？

        因为那样的话，一次 logger.info("xxx") 会走三遍：

        先走 console_sink → 写 stderr
        再走 file_sink → 写文件
        再走 QueueHandler → 进队列，后台线程再写一遍 stderr + 文件

        一条日志落盘 4 次。​ 而且顺序不可控，stderr 和文件里行与行之间会错位。
        """
        # “偷 handler”三步：clear → 校验 → 只挂 QueueHandler
        sink_handlers = list(app_logger.handlers) # ① 先把刚配好的 handler 引用存下来
        app_logger.handlers.clear() # ② 从 logger 上摘干净

        if not sink_handlers:
            raise RuntimeError(
                "logging sinks were not configured"
            )

        if not any(
            isinstance(
                handler,
                RotatingFileHandler,
            )
            for handler in sink_handlers
        ):
            raise RuntimeError(
                "RotatingFileHandler was not configured"
            )

        # QueueHandler + QueueListener：生产者和消费者 ,async 事件循环里的协程，永远只做一次内存 put，不会因磁盘 IO 被挂住。
        queue_handler = SafeQueueHandler(
            log_queue,
            include_traceback=settings.log_include_traceback,
        )
        queue_handler.setLevel(level_name)
        queue_handler.addFilter(ContextFilter()) # 在前台业务协程把 request/thread 上下文复制到 LogRecord，再放进队列。
        queue_handler._deep_research_managed = True # 纯标记，后面 shutdown 时认亲用的

        app_logger.addHandler(queue_handler)
        app_logger.setLevel(level_name)
        app_logger.propagate = False

        # 后台线程启动，不断从队列取日志，分发给 sink handler 写文件/写控制台。
        listener = SafeQueueListener(
            log_queue,
            *sink_handlers,
            respect_handler_level=True,
        )
        listener.start()

        # 获取 Logger 对象
        audit_logger = logging.getLogger(
            "deep_research.audit"
        )

        # 拼接路径
        audit_path = (
            Path(settings.log_dir)
            / settings.log_audit_file
        )

        # 创建日志目录
        audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 配置文件处理器
        audit_file_handler = RotatingFileHandler(
            filename=str(audit_path),
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )

        # 设置该处理器只处理指定级别（如 INFO、ERROR）及以上的日志；使用自定义的 AuditFormatter 来格式化日志内容。
        audit_file_handler.setLevel(level_name)
        audit_file_handler.setFormatter(
            AuditFormatter()
        )

        # 创建内存队列
        audit_queue: queue.Queue[
            logging.LogRecord
        ] = queue.Queue()

        # 配置队列处理器
        audit_queue_handler = SafeQueueHandler(
            audit_queue,
            include_traceback=settings.log_include_traceback,
        )

        # 获取日志等级，看待会是否过滤
        audit_queue_handler.setLevel(level_name)
        audit_queue_handler.addFilter(
            ContextFilter() # 过滤
        )

        audit_logger.handlers.clear()
        audit_logger.addHandler(
            audit_queue_handler
        )
        audit_logger.setLevel(level_name)
        audit_logger.propagate = False

        audit_listener = AuditQueueListener(
            audit_queue,
            audit_file_handler,
            respect_handler_level=True,
        )

        audit_listener.start()

        # 保存状态引用
        setattr(
            audit_logger,
            _AUDIT_STATE_ATTR,
            {
                "listener": audit_listener,
                "queue_handler": audit_queue_handler,
                "file_handler": audit_file_handler,
            },
        )

        # setattr 和开头 getattr 守卫是一对：写标记的地方，就是初始化完成的地方。​ 状态字典里存了三个句柄，shutdown 时全靠它们找回。
        setattr(
            app_logger,
            _LOG_STATE_ATTR,
            {
                "listener": listener,
                "queue_handler": queue_handler,
                "sink_handlers": sink_handlers,
            },
        )

        if not _ATEXIT_REGISTERED:
            atexit.register(shutdown_logging)
            _ATEXIT_REGISTERED = True




def shutdown_logging() -> None:
    """
    停止 QueueListener，并关闭文件 handler。
    可重复调用。
    """
    app_logger = logging.getLogger("deep_research")
    audit_logger = logging.getLogger(
        "deep_research.audit"
    )

    with _LOG_LOCK:
        state = getattr(
            app_logger,
            _LOG_STATE_ATTR,
            None,
        )

        audit_state = getattr(
            audit_logger,
            _AUDIT_STATE_ATTR,
            None,
        )
        if audit_state is not None:
            audit_queue_handler = (
                audit_state["queue_handler"]
            )
            audit_listener = audit_state["listener"]
            audit_file_handler = (
                audit_state["file_handler"]
            )

            audit_logger.removeHandler(
                audit_queue_handler
            )

            _stop_listener(audit_listener)
            audit_file_handler.close()

            delattr(
                audit_logger,
                _AUDIT_STATE_ATTR,
            )

        if state is None:
            return  # 没初始化过，或已经关过 → 幂等，直接走

        queue_handler = state["queue_handler"]
        listener = state["listener"]
        sink_handlers = state["sink_handlers"]

        app_logger.removeHandler(queue_handler)  # ① logger 不再接新日志

        _stop_listener(listener)  # ② 通知后台线程：停。线程会把队列剩的排空后再死

        for handler in sink_handlers:
            handler.close()   # ③ 关文件 fd / 关 stderr 流

        delattr(app_logger, _LOG_STATE_ATTR)