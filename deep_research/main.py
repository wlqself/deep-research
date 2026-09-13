import mimetypes
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    configure_logging,
    settings,
)
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
from .handlers.hitl import (
    router as hitl_router,
)
from .handlers.memory import (
    router as memory_router,
)
from .handlers.publishing import (
    router as publishing_router,
)
from .persistence import (
    sqlite_checkpointer,
    sqlite_memory_store,
)
from .memory.service import MemoryService
from .middleware.request_context import (
    RequestContextMiddleware,
)
from .app_lifecycle import build_lifespan

configure_logging()


def mount_published_articles(
    application: FastAPI,
    published_site_dir: str | Path,
) -> Path:
    """Expose generated Markdown through a read-only StaticFiles mount."""

    mimetypes.add_type("text/markdown", ".md")
    articles_dir = (Path(published_site_dir) / "articles").resolve()
    articles_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/articles",
        StaticFiles(directory=articles_dir, html=False),
        name="published-articles",
    )
    return articles_dir


#FastAPI启动
lifespan = build_lifespan(
    settings=settings,
    agent_module=agent_module,
    document_registry_factory=lambda path: DocumentRegistry(path),
    rag_service_factory=lambda current_settings: (
        build_rag_service(current_settings)
    ),
    checkpointer_factory=lambda: sqlite_checkpointer(),
    memory_store_factory=lambda: sqlite_memory_store(),
    memory_service_factory=lambda store, **kwargs: (
        MemoryService(store, **kwargs)
    ),
    memory_extractor_factory=lambda current_model: (
        MemoryExtractor(current_model)
    ),
    model=model,
)


app = FastAPI(
    title="Deep Research Agent",
    lifespan=lifespan,
)

app.add_middleware(
    RequestContextMiddleware
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

PUBLISHED_ARTICLES_DIR = mount_published_articles(
    app,
    settings.published_site_dir,
)

app.include_router(research_router)
app.include_router(artifacts_router)
app.include_router(threads_router)
app.include_router(knowledge_router)
app.include_router(memory_router)
app.include_router(hitl_router)
app.include_router(publishing_router)

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
