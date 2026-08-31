from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config import settings


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

    async with AsyncSqliteSaver.from_conn_string(
        str(database_path)
    ) as checkpointer:
        yield checkpointer