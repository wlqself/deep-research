from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from ..config import Settings
from .lexical import SqliteBm25Index
from .query_rewrite import QueryRewriter
from .service import RagService

def create_qdrant_client(path: str) -> QdrantClient:
    storage_path = Path(path)
    storage_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=str(storage_path),
    )

def ensure_collection(
        client: QdrantClient,
        collection_name: str,
        dimensions: int,
) -> None:
    expected_distance = Distance.COSINE

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=expected_distance,
            ),
        )
        return

    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors

    if not isinstance(vectors, VectorParams):
        raise RuntimeError(
            "Collection 使用了不支持的向量配置，需要重新索引。"
        )

    if (
        vectors.size != dimensions
        or vectors.distance != expected_distance
    ):
        raise RuntimeError(
            "Collection 向量维度或距离类型不匹配，需要重新索引。"
        )



def create_embeddings(
    settings: Settings,
) -> Embeddings:
    kwargs = {
        "model": settings.embedding_model_name,
        "dimensions": settings.embedding_dimensions,
        "api_key": (
            settings.embedding_api_key
            or settings.model_api_key
        ),
        "chunk_size": settings.embedding_batch_size,
        "tiktoken_enabled": False,
        "check_embedding_ctx_length": False,
    }

    if settings.embedding_base_url:
        kwargs["base_url"] = settings.embedding_base_url
    elif settings.model_base_url:
        kwargs["base_url"] = settings.model_base_url

    return OpenAIEmbeddings(**kwargs)


def create_query_rewriter(settings: Settings) -> QueryRewriter:
    kwargs = {
        "model": settings.model_name,
        "api_key": settings.model_api_key,
        "temperature": 0,
        "timeout": 20,
        "max_retries": 0,
    }
    if settings.model_base_url:
        kwargs["base_url"] = settings.model_base_url
    return QueryRewriter(
        ChatOpenAI(**kwargs),
        structured_output=False,
    )

def build_rag_service(
    settings: Settings,
) -> RagService:
    client = create_qdrant_client(
        settings.rag_qdrant_path
    )

    try:
        ensure_collection(
            client=client,
            collection_name=settings.rag_collection_name,
            dimensions=settings.embedding_dimensions,
        )

        embeddings = create_embeddings(settings)
        bm25_index = SqliteBm25Index(settings.rag_bm25_db_path)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.rag_collection_name,
            embedding=embeddings,
            validate_collection_config=False,
        )

        rag_service = RagService(
            client=client,
            embeddings=embeddings,
            vector_store=vector_store,
            collection_name=settings.rag_collection_name,
            metadata_weight=settings.rag_metadata_weight,
            bm25_index=bm25_index,
            query_rewriter=(
                create_query_rewriter(settings)
                if settings.rag_query_rewrite_enabled
                else None
            ),
            candidate_k=settings.rag_candidate_k,
            rrf_k=settings.rag_rrf_k,
        )
        rag_service.rebuild_lexical_index()
        return rag_service

    except Exception:
        if "bm25_index" in locals():
            bm25_index.close()
        client.close()
        raise
