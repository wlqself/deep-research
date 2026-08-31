from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .rag.factory import build_rag_service
from .rag.registry import DocumentRegistry
from . import agent as agent_module
from .handlers.research import (
    ResearchRequest,
    ResearchResponse,
    router as research_router,
)
from .agent.model import model
from .memory.extractor import MemoryExtractor
from .handlers.threads import (
    router as threads_router,
)

from .handlers.artifacts import (
    router as artifacts_router,
)
from .handlers.knowledge import (
    router as knowledge_router,
)
from .handlers.memory import (
    router as memory_router,
)
from .persistence import (
    sqlite_checkpointer,
    sqlite_memory_store,
)
from .memory.service import MemoryService

#FastAPI启动
@asynccontextmanager
async def lifespan(app: FastAPI):
    previous_agent = agent_module.agent

    registry = DocumentRegistry(
        settings.rag_registry_db_path
    )
    registry.initialize()

    rag_service = build_rag_service(settings)

    app.state.document_registry = registry
    app.state.rag_service = rag_service

    try:
        async with sqlite_checkpointer() as checkpointer:
            async with sqlite_memory_store() as memory_store:
                memory_service = MemoryService(
                    memory_store,
                    user_id=settings.memory_user_id,
                )
                memory_extractor = MemoryExtractor(model)

                app.state.memory_store = memory_store
                app.state.memory_service = memory_service
                app.state.memory_extractor = memory_extractor

                agent_module.agent = agent_module.build_agent(
                    checkpointer,
                    rag_service,
                    memory_service,
                )

                yield
    finally:
        agent_module.agent = previous_agent
        app.state.document_registry = None
        app.state.rag_service = None
        app.state.memory_store = None
        app.state.memory_service = None
        app.state.memory_extractor = None
        rag_service.close()

app = FastAPI(
    title="Deep Research Agent",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

app.include_router(research_router)
app.include_router(artifacts_router)
app.include_router(threads_router)
app.include_router(knowledge_router)
app.include_router(memory_router)

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
