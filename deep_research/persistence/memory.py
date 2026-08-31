from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langgraph.store.sqlite import AsyncSqliteStore

from ..config import settings
from ..rag.factory import create_embeddings

"""
settings.memory_db_path
  -> Path
  -> AsyncSqliteStore.from_conn_string()
  -> await store.setup()
  -> 调用方使用 store
  -> 离开 async with
  -> SQLite 连接关闭
"""
@asynccontextmanager
async def sqlite_memory_store(
    embeddings: Embeddings | None = None,
) -> AsyncIterator[AsyncSqliteStore]:
    
    database_path = Path(settings.memory_db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    memory_embeddings = embeddings or create_embeddings(settings)

    async with AsyncSqliteStore.from_conn_string(
        str(database_path),
        index={
            "dims": settings.embedding_dimensions,
            "embed": memory_embeddings,
            "fields": ["title", "summary", "keywords"],
        },
    ) as store:
        await store.setup()
        yield store