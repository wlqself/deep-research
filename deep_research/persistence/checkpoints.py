import logging

from ..log.logging_utils import log_event
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config import settings

logger = logging.getLogger(
    "deep_research.checkpointer"
)

@asynccontextmanager
async def sqlite_checkpointer() -> AsyncIterator[
    AsyncSqliteSaver
]:
    database_path = Path(
        settings.checkpoint_db_path
    )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        async with AsyncSqliteSaver.from_conn_string(
            str(database_path)
        ) as checkpointer:
            log_event(
                logger,
                logging.INFO,
                "store.checkpointer.started",
                status="started",
            )
            yield checkpointer
    finally:
        log_event(
            logger,
            logging.INFO,
            "store.checkpointer.stopped",
            status="completed",
        )