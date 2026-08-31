"""
完整流程：
SQLite pending
  ↓
SQLite processing
  ↓
解析
  ↓
切 Chunk
  ↓
写入 Qdrant，Chunk status=pending
  ↓
校验写入数量
  ↓
Qdrant status=indexed
  ↓
SQLite status=indexed
"""
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

class RagService:
    def __init__(
        self,
        *,
        client: QdrantClient,
        embeddings: Embeddings,
        vector_store: QdrantVectorStore,
        collection_name: str,
    ) -> None:
        self.client = client
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.collection_name = collection_name

    def close(self) -> None:
        self.client.close()

    # 在metadata设置status状态
    def mark_document_chunks_pending(
        self,
        document_id: str,
    ) -> None:
        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            raise ValueError("document_id must not be empty")

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"status": "pending"},       #  写 "pending"
            key="metadata",                      #  合并进 metadata 嵌套，不覆盖其他字段
            points=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=normalized_document_id),
                    )
                ]
            ),
            wait=True,
        )

    def index_chunks(
        self,
        chunks: list[Document],
    ) -> int:
        # 判断是否空chunk
        if not chunks:
            return 0

        ids: list[str] = []

        #循环取出chunks并
        for chunk in chunks:
            raw_chunk_id = chunk.metadata.get("chunk_id")
            if not isinstance(raw_chunk_id, str):
                raise ValueError("chunk metadata must contain chunk_id")
            ids.append(raw_chunk_id)

        written_ids = self.vector_store.add_documents(
            documents=chunks,
            ids=ids,
        )
        return len(written_ids)

    #  Qdrant 向量删除方法
    def delete_document_chunks(
        self,
        document_id: str,
    ) -> None:
        # 去掉首尾空格
        normalized_document_id = (
            document_id.strip()
        )
        # 空字符串拦截
        if not normalized_document_id:
            raise ValueError(
                "document_id must not be empty"
            )
        # 删除该文档所有 Chunk
        self.client.delete(
            collection_name=self.collection_name,
            # 精确匹配 payload 里的 document_id
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(
                            value=normalized_document_id
                        ),
                    )
                ]
            ),
            # 同步等删除完成才返回
            wait=True
        )

    def mark_document_chunks_indexed(
        self,
        document_id: str,
    ) -> None:
        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            raise ValueError("document_id must not be empty")

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"status": "indexed"},       # 写 "indexed"
            key="metadata",                      # 合并进 metadata 嵌套，不覆盖其他字段
            points=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=normalized_document_id),
                    )
                ]
            ),
            wait=True,
        )

    def search_chunks(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query must not be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )
        # 条件，找出那些status是indexed的chunk
        conditions = [
            FieldCondition(
                key="metadata.status",
                match=MatchValue(
                    value="indexed"
                ),
            )
        ]

        # 标准化 document_id
        if document_id is not None:
            normalized_document_id = (
                document_id.strip()
            )

            if not normalized_document_id:
                raise ValueError(
                    "document_id must not be empty"
                )

            # 只在指定文档的 chunk 里搜，而不是在所有文档里搜。
            conditions.append(
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(
                        value=normalized_document_id
                    ),
                )
            )

        return self.vector_store.similarity_search_with_score(
            query=normalized_query,
            k=top_k,
            filter=Filter(
                must=conditions
            ),
        )